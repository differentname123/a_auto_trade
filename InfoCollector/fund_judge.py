import os
import itertools
import math
import re
import glob
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from numba import njit
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
# ====================================================================
# 全局常量配置
# ====================================================================
TRADING_DAYS_PER_YEAR = 252
DEFAULT_REBALANCE_DAYS = 30
DEFAULT_MAX_HISTORY = 5 * TRADING_DAYS_PER_YEAR
ROLLING_WINDOW = TRADING_DAYS_PER_YEAR
ROLLING_STEP = 21

# VETO 否决阈值
VETO_MAX_MDD = 0.3
VETO_HURDLE_RATE = 0.2
VETO_AR1 = 0.35
VETO_VOL = 0.03
VETO_RECOVERY_DAYS = 180
VETO_MISSING_RATIO = 0.05
VETO_CONTINUOUS_ZEROS = 10
VETO_CALMAR_MULTIPLIER = 1.10
VETO_WORST_1Y = -0.15
VETO_ZOMBIE_DAYS = 35

# 打分参数
TAIL_RISK_SENSITIVITY = 2.5
TIME_SATURATION_DAYS = 756
SCORE_DC_HIGH = 0.8
SCORE_DC_MID = 0.5

# 多进程配置
BATCH_SIZE = 200000
CHUNK_SIZE = 2000
PERTURBATION_SEEDS = [1024, 2048, 4096]

# 输出列定义
OUTPUT_COLUMNS = [
    '组合文件名', 'Start_Date', 'End_Date', 'Total_Days',
    'CAGR', 'Max_Drawdown', 'Max_Recovery_Days', 'Worst_Rolling_1Y_R2',
    'AR1_Coefficient', 'Annualized_Volatility', 'Sharpe_Ratio',
    'Calmar_Ratio', 'Daily_Win_Rate', 'Downside_Correlation',
    'Avg_CAGR', 'Avg_Max_Drawdown', 'Calmar_Baseline', 'Worst_Rolling_1Y_Return',
    'error', 'Total_Score'
]

# 子进程全局容器
WORKER_MASTER_DF = None


def _init_worker(master_df):
    """子进程初始化:将主进程的大 DataFrame 锚定到子进程本地内存"""
    global WORKER_MASTER_DF
    WORKER_MASTER_DF = master_df


# ====================================================================
# 通用工具
# ====================================================================
def now_str():
    return datetime.now().strftime('%H:%M:%S')


def log(msg):
    print(f"[{now_str()}] {msg}")


def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def extract_fund_code(filepath):
    match = re.search(r'(\d{6})', os.path.basename(filepath))
    return match.group(1) if match else os.path.basename(filepath).replace('.csv', '')


def normalize_combo_tuple(combo_str, sep='_'):
    parts = [str(x).strip() for x in str(combo_str).split(sep)]
    return tuple(sorted(parts))


# ====================================================================
# 缓存自更新系统 (按要求维度泛化并持久化)
# ====================================================================
def load_or_init_computed_set(curr_dim):
    """
    加载或初始化已计算组合的全局缓存集合，文件名仅与维度相关。
    先加载缓存，再扫描目录中比缓存新的 parquet 文件进行增量更新。
    """
    cache_file = f'fund_data/computed_combos_{curr_dim}d_global_cache.pkl'
    parquet_files = glob.glob(f'fund_data/fof_evaluation_results_{curr_dim}d_*.parquet')

    computed_set = set()
    cache_time = 0

    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        try:
            with open(cache_file, 'rb') as f:
                computed_set = pickle.load(f)
            log(f"📦 命中全局缓存 | 规则: {curr_dim} 维 | 已加载 {len(computed_set):,} 个历史元组。")
        except Exception as e:
            log(f"⚠️ 缓存读取失败将重建: {e}")
            cache_time = 0

    log(f"🔍 扫描 Parquet 文件进行增量更新 | 规则: {curr_dim} 维 | 共发现 {len(parquet_files)} 个相关文件。")
    updated = False
    for pf in parquet_files:
        if os.path.getmtime(pf) <= cache_time:
            continue
        try:
            table = pq.read_table(pf, columns=['组合文件名'])
            raw_combos = table.column('组合文件名').to_pylist()
            new_tuples = {normalize_combo_tuple(c) for c in raw_combos if c}
            added = len(new_tuples - computed_set)
            computed_set.update(new_tuples)
            if added > 0:
                updated = True
                log(f"🔄 增量读取文件 | {os.path.basename(pf)} | 新增 {added:,} 个有效组合。")
        except Exception as e:
            log(f"❌ 读取文件异常跳过 | {os.path.basename(pf)} | 错误: {e}")

    if updated or (not os.path.exists(cache_file) and computed_set):
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(computed_set, f)
            log(f"✅ 缓存已持久化 | 规则: {curr_dim} 维 | 当前全局总库容量: {len(computed_set):,} 个。")
        except Exception as e:
            log(f"❌ 缓存持久化失败: {e}")
    elif not parquet_files and not computed_set:
        log(f"ℹ️ 空白初始化 | 规则: {curr_dim} 维 | 暂无历史计算数据。")

    return computed_set, cache_file


def _update_cache_from_new_file(output_file, cache_file, computed_set):
    """根据新生成的结果文件,动态追加更新组合缓存"""
    if not output_file or not os.path.exists(output_file): return
    cache_time = os.path.getmtime(cache_file) if os.path.exists(cache_file) else 0
    if os.path.getmtime(output_file) <= cache_time: return

    log("🌟 嗅探到新生成的结果文件更新于缓存,正在动态追加至持久化系统...")
    try:
        if os.path.getsize(output_file) <= 100: return
        table = pq.read_table(output_file, columns=['组合文件名'])
        raw_combos = table.column('组合文件名').to_pylist()
        added = 0
        for c in raw_combos:
            if c:
                combo_tup = normalize_combo_tuple(c)
                if combo_tup not in computed_set:
                    computed_set.add(combo_tup)
                    added += 1
        if added > 0:
            with open(cache_file, 'wb') as f:
                pickle.dump(computed_set, f)
            log(f"🔒 增量缓存注入完成,本次新增 {added} 个,库中最新包含 {len(computed_set):,} 个组合。")
    except Exception as e:
        log(f"❌ 动态更新缓存失败: {e}")


# ====================================================================
# Numba JIT 核心引擎 (严禁修改)
# ====================================================================
@njit(fastmath=True, cache=True)
def _fast_simulate_path(fund_rets_arr, w_init, current_reb_days, offset, n_days, n_funds):
    synth_ret_arr = np.zeros(n_days)
    w = np.copy(w_init)
    for t in range(n_days):
        daily_port_ret = np.sum(w * fund_rets_arr[t])
        synth_ret_arr[t] = daily_port_ret
        denom = 1.0 + daily_port_ret if (1.0 + daily_port_ret) > 1e-9 else 1e-9
        w = w * (1.0 + fund_rets_arr[t]) / denom
        if t >= current_reb_days and (t - offset) % current_reb_days == 0:
            w = np.copy(w_init)
    return synth_ret_arr


# ====================================================================
# 指标计算辅助函数 (严禁修改)
# ====================================================================
def _compute_max_continuous_zeros(fund_rets):
    zero_df = (fund_rets.abs() < 1e-8).astype(int)
    max_zeros = 0
    for col in zero_df.columns:
        consecutive = zero_df[col].groupby((zero_df[col] == 0).cumsum()).sum().max()
        max_zeros = max(max_zeros, consecutive)
    return max_zeros


def _compute_baseline_metrics(fund_rets_arr, n_days):
    fund_eq = np.cumprod(1.0 + fund_rets_arr, axis=0)
    fund_cagrs = np.power(fund_eq[-1, :], TRADING_DAYS_PER_YEAR / n_days) - 1.0
    avg_cagr = float(np.mean(fund_cagrs))
    cum_max = np.clip(np.maximum.accumulate(fund_eq, axis=0), a_min=1.0, a_max=None)
    drawdowns = (fund_eq - cum_max) / cum_max
    avg_mdd = float(np.mean(np.min(drawdowns, axis=0)))
    calmar_baseline = float(avg_cagr / abs(avg_mdd)) if abs(avg_mdd) > 1e-6 else 0.0
    return avg_cagr, avg_mdd, calmar_baseline


def _compute_worst_rolling_1y(synth_eq_arr, n_days):
    if n_days < TRADING_DAYS_PER_YEAR: return 0.0
    eq_padded = np.concatenate(([1.0], synth_eq_arr))
    rolling_ret = (eq_padded[TRADING_DAYS_PER_YEAR:] / eq_padded[:-TRADING_DAYS_PER_YEAR]) - 1.0
    return float(np.min(rolling_ret))


def _compute_worst_rolling_1y_r2(synth_eq_arr, n_days):
    log_eq = np.log(np.clip(synth_eq_arr, 1e-9, None))
    window = min(ROLLING_WINDOW, n_days)
    x_arr = np.arange(window)
    rolling_r2 = []
    for st in range(0, n_days - window + 1, ROLLING_STEP):
        y_sub = log_eq[st:st + window]
        r_mat = np.corrcoef(x_arr, y_sub)
        if r_mat.shape == (2, 2) and not np.isnan(r_mat[0, 1]):
            rolling_r2.append(r_mat[0, 1] ** 2)
        else:
            rolling_r2.append(0.0)
    return float(min(rolling_r2)) if rolling_r2 else 0.0


def _compute_downside_correlation(synth_ret, fund_rets):
    n_worst = max(5, int(len(synth_ret) * 0.05))
    worst_dates = synth_ret.nsmallest(n_worst).index
    worst_fund_rets = fund_rets.loc[worst_dates]
    if len(worst_fund_rets) <= 3: return 0.5
    corr_mat = worst_fund_rets.corr().values
    triu_idx = np.triu_indices_from(corr_mat, k=1)
    if len(triu_idx[0]) == 0: return 0.5
    with np.errstate(invalid='ignore'):
        max_corr = np.nanmax(corr_mat[triu_idx])
    return float(max_corr) if not np.isnan(max_corr) else 0.5


def _check_vetoes(metrics, vol_annual, max_missing_ratio, max_continuous_zeros, max_mdd_limit, hurdle_rate):
    return {
        "VETO_Hurdle_Rate": metrics['CAGR'] < hurdle_rate,
        "VETO_Drawdown_Crash": abs(metrics['Max_Drawdown']) > max_mdd_limit,
        "VETO_Fake_Smooth": (metrics['AR1_Coefficient'] > VETO_AR1) and (vol_annual < VETO_VOL),
        "VETO_Endless_Bleeding": metrics['Max_Recovery_Days'] > VETO_RECOVERY_DAYS,
        "VETO_Data_Distortion": (max_missing_ratio > VETO_MISSING_RATIO) or (
                max_continuous_zeros > VETO_CONTINUOUS_ZEROS),
        "VETO_Below_Calmar_Baseline": metrics['Calmar_Ratio'] <= (metrics['Calmar_Baseline'] * VETO_CALMAR_MULTIPLIER),
        "VETO_Worst_1Y_Crash": metrics['Worst_Rolling_1Y_Return'] < VETO_WORST_1Y,
    }


def _compute_total_score(metrics, n_days):
    excess_cagr = max(0.0, metrics['CAGR'])
    adj_mdd = max(abs(metrics['Max_Drawdown']), 0.01)
    base_calmar = excess_cagr / adj_mdd
    smoothness = metrics['Worst_Rolling_1Y_R2']
    worst_1y = min(0.0, metrics['Worst_Rolling_1Y_Return'])
    tail_discount = float(np.exp(worst_1y * TAIL_RISK_SENSITIVITY))
    time_mult = min(1.0, np.sqrt(n_days / TIME_SATURATION_DAYS))
    dc = metrics['Downside_Correlation']
    if dc > SCORE_DC_HIGH:
        p_corr = 0.1
    elif dc > SCORE_DC_MID:
        p_corr = float(np.clip(1.0 - (dc - SCORE_DC_MID) * 1.0, 0.5, 1.0))
    else:
        p_corr = 1.0
    base_score = base_calmar * smoothness * tail_discount
    return max(0.0, float(base_score * time_mult * p_corr))


# ====================================================================
# 核心评估引擎 (严禁修改)
# ====================================================================
def evaluate_fof_portfolio_fast(merged_nav, rebalance_days=DEFAULT_REBALANCE_DAYS,
                                max_history_days=DEFAULT_MAX_HISTORY,
                                max_mdd_limit=VETO_MAX_MDD, hurdle_rate=VETO_HURDLE_RATE):
    n_funds = merged_nav.shape[1]
    raw_index = merged_nav.index
    valid_mask = merged_nav.notna().all(axis=1)
    if not valid_mask.any(): return {"error": "No overlapping data", "Total_Score": 0.0}
    merged_nav = merged_nav.loc[valid_mask.idxmax():valid_mask[::-1].idxmax()]
    if len(merged_nav) > max_history_days: merged_nav = merged_nav.iloc[-max_history_days:]
    if len(merged_nav) < TRADING_DAYS_PER_YEAR: return {"error": "Common data length < 252 days", "Total_Score": 0.0}

    max_missing_ratio = float((merged_nav.isna().sum() / len(merged_nav)).max())
    if (raw_index[-1] - merged_nav.index[-1]).days > VETO_ZOMBIE_DAYS:
        return {"error": f"Zombie fund detected. Cutoff at {merged_nav.index[-1].strftime('%Y-%m-%d')}",
                "Total_Score": 0.0}

    fund_rets = merged_nav.ffill().pct_change().fillna(0.0)
    max_continuous_zeros = _compute_max_continuous_zeros(fund_rets)
    n_days = len(fund_rets)
    fund_rets_arr = np.ascontiguousarray(fund_rets.values, dtype=np.float64)
    avg_cagr, avg_mdd, calmar_baseline = _compute_baseline_metrics(fund_rets_arr, n_days)

    def _evaluate_single_path(reb_days, offset=0, w_init=None):
        w_init_arr = (np.ones(n_funds, dtype=np.float64) / n_funds
                      if w_init is None else np.array(w_init, dtype=np.float64))
        synth_ret_arr = _fast_simulate_path(fund_rets_arr, w_init_arr, reb_days, offset, n_days, n_funds)
        synth_ret = pd.Series(synth_ret_arr, index=fund_rets.index)
        synth_eq = (1 + synth_ret).cumprod()

        cagr = float(synth_eq.iloc[-1] ** (TRADING_DAYS_PER_YEAR / n_days) - 1)
        cum_max = synth_eq.cummax().clip(lower=1.0)
        drawdowns = (synth_eq - cum_max) / cum_max
        max_dd = float(drawdowns.min())
        is_dd = drawdowns < 0
        max_recovery_days = int(is_dd.groupby((~is_dd).cumsum()).sum().max())
        vol_annual = synth_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = float(cagr / vol_annual) if vol_annual > 1e-6 else 0.0
        calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0
        ar1 = synth_ret.autocorr(lag=1)
        ar1 = float(ar1) if pd.notna(ar1) else 1.0

        metrics = {
            'Start_Date': merged_nav.index[0].strftime('%Y-%m-%d'),
            'End_Date': merged_nav.index[-1].strftime('%Y-%m-%d'),
            'Total_Days': n_days, 'n_funds': n_funds, 'Avg_CAGR': avg_cagr, 'Avg_Max_Drawdown': avg_mdd,
            'Calmar_Baseline': calmar_baseline, 'CAGR': cagr, 'Max_Drawdown': max_dd,
            'Max_Recovery_Days': max_recovery_days,
            'Worst_Rolling_1Y_Return': _compute_worst_rolling_1y(synth_eq.values, n_days),
            'Worst_Rolling_1Y_R2': _compute_worst_rolling_1y_r2(synth_eq.values, n_days),
            'AR1_Coefficient': ar1, 'Annualized_Volatility': float(vol_annual), 'Sharpe_Ratio': sharpe,
            'Calmar_Ratio': calmar, 'Daily_Win_Rate': float((synth_ret > 0).sum() / n_days) if n_days > 0 else 0.0,
            'Downside_Correlation': _compute_downside_correlation(synth_ret, fund_rets),
        }

        vetoes = _check_vetoes(metrics, vol_annual, max_missing_ratio, max_continuous_zeros, max_mdd_limit, hurdle_rate)
        metrics.update(vetoes)
        if any(vetoes.values()): return metrics, 0.0
        return metrics, _compute_total_score(metrics, n_days)

    final_metrics, score_base = _evaluate_single_path(rebalance_days, offset=0)
    if score_base == 0.0:
        final_metrics['Total_Score'] = 0.0
        final_metrics['VETO_Perturbation_Death'] = False
        return final_metrics

    half_offset = rebalance_days // 2
    score_weight = float('inf')
    metrics_w = None

    for seed in PERTURBATION_SEEDS:
        np.random.seed(seed)
        shift = np.random.uniform(-0.05, 0.05, n_funds)
        w_perturbed = np.clip(np.ones(n_funds) / n_funds + shift, 0.01, 1.0)
        w_perturbed = w_perturbed / np.sum(w_perturbed)
        m_w, s_w = _evaluate_single_path(rebalance_days, offset=half_offset, w_init=w_perturbed)
        if s_w < score_weight: score_weight, metrics_w = s_w, m_w

    if score_weight == float('inf'): score_weight = 0.0
    final_metrics['Total_Score'] = min(score_base, score_weight)
    final_metrics['VETO_Perturbation_Death'] = (final_metrics['Total_Score'] == 0.0)

    if final_metrics['VETO_Perturbation_Death'] and metrics_w is not None and score_weight == 0.0:
        for k, v in metrics_w.items():
            if k.startswith("VETO_") and v is True:
                final_metrics[k + "_in_Perturb"] = True

    return final_metrics


# ====================================================================
# Worker 工作进程函数 (优化:消灭多进程内的 re 正则解析)
# ====================================================================
def _worker_process_combo(combo_codes):
    """单组合处理:传入的已是纯粹的 6 位代码元组，极速执行"""
    global WORKER_MASTER_DF
    combo_name = "_".join(combo_codes)
    try:
        merged_nav = WORKER_MASTER_DF[list(combo_codes)]
        result = evaluate_fof_portfolio_fast(merged_nav)
        result['组合文件名'] = combo_name
        return result
    except Exception as e:
        return {'组合文件名': combo_name, 'error': f"处理异常: {str(e)}", 'Total_Score': 0.0}


def _worker_process_chunk(combo_chunk):
    return [_worker_process_combo(combo) for combo in combo_chunk]


# ====================================================================
# 基金净值并行加载 & 相关性双矩阵 & 漏斗 (包含详细日志修改)
# ====================================================================
def _load_single_nav(filepath):
    try:
        if not os.path.exists(filepath): return None
        df = pd.read_csv(filepath)
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        df = df.set_index('净值日期').sort_index()
        df = df[~df.index.duplicated(keep='last')]
        return df['复权净值'].rename(filepath)
    except Exception:
        return None


def _parallel_load_navs(files, max_workers, desc):
    series_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_single_nav, f) for f in files]
        for future in tqdm(as_completed(futures), total=len(files), desc=desc):
            res = future.result()
            if res is not None: series_list.append(res)
    return series_list


def _build_downside_corr_matrix(downside_csv):
    try:
        df_2d = pd.read_csv(downside_csv)
    except Exception as e:
        log(f"提取 Downside_Correlation 矩阵失败: {e}")
        return pd.DataFrame()
    if '组合文件名' not in df_2d.columns or 'Downside_Correlation' not in df_2d.columns: return pd.DataFrame()
    records = []
    for _, row in df_2d.iterrows():
        combo = row['组合文件名']
        if pd.isna(combo): continue
        parts = [p.strip() for p in str(combo).split('+')]
        if len(parts) != 2: continue
        f1, f2 = f'fund_data/nav/{parts[0]}', f'fund_data/nav/{parts[1]}'
        dc = row['Downside_Correlation']
        records.append({'f1': f1, 'f2': f2, 'dc': dc})
        records.append({'f1': f2, 'f2': f1, 'dc': dc})
    if not records: return pd.DataFrame()
    return pd.DataFrame(records).groupby(['f1', 'f2'])['dc'].mean().unstack()


def precompute_correlations(result_csv='fund_data/all_funds_result.csv', corr_csv='fund_data/fund_correlations.csv',
                            downside_csv='fund_data/fof_evaluation_results_2d_pool3917.csv', max_workers=10,
                            downside_corr_csv='fund_data/fund_downside_correlations.csv'):
    print(f"\n[{now_str()}] ---------------- 开始准备全市场相关性双矩阵 ----------------")
    if os.path.exists(downside_corr_csv):
        log(f"已发现现有现行相关性文件 {downside_corr_csv},直接加载缓存...")
        downside_corr_matrix = pd.read_csv(downside_corr_csv, index_col=0)
    elif os.path.exists(downside_csv):
        log(f"正在从 2 维历史组合文件 {downside_csv} 提取 Downside_Correlation 矩阵...")
        downside_corr_matrix = _build_downside_corr_matrix(downside_csv)
        if not downside_corr_matrix.empty:
            log(f"成功构建下行相关性矩阵,涵盖 {len(downside_corr_matrix)} 只标的。")
            ensure_dir(downside_corr_csv)
            downside_corr_matrix.to_csv(downside_corr_csv)
    else:
        log(f"未发现 2 维组合文件 {downside_csv},跳过下行相关性提取。")
        downside_corr_matrix = pd.DataFrame()

    if os.path.exists(corr_csv):
        log(f"已发现现有全天候相关性文件 {corr_csv},直接加载缓存...")
        return pd.read_csv(corr_csv, index_col=0), downside_corr_matrix

    if not os.path.exists(result_csv): return pd.DataFrame(), downside_corr_matrix
    df_results = pd.read_csv(result_csv)
    files = df_results['adj_nav_file'].dropna().tolist()
    log(f"正在并行读取 {len(files)} 只基金...")
    nav_series_list = _parallel_load_navs(files, max_workers, "并行加载数据")
    if not nav_series_list: return pd.DataFrame(), downside_corr_matrix
    merged_nav = pd.concat(nav_series_list, axis=1, join='outer').ffill()
    corr_matrix = merged_nav.pct_change().corr()
    ensure_dir(corr_csv)
    corr_matrix.to_csv(corr_csv)
    return corr_matrix, downside_corr_matrix


def _extract_target_codes(df_filtered):
    if '基金代码' in df_filtered.columns: return df_filtered['基金代码'].astype(str).str.strip().str.zfill(6)
    if 'fund_code' in df_filtered.columns: return df_filtered['fund_code'].astype(str).str.strip().str.zfill(6)
    return df_filtered['adj_nav_file'].apply(
        lambda x: re.search(r'(\d{6})', str(x)).group(1) if pd.notna(x) and re.search(r'(\d{6})', str(x)) else "000000")


def filter_fund_pool(df_results, active_cache='temp/active_fund_codes.csv', min_annual_return=0.15, min_day=1000):
    if df_results is None or df_results.empty: return pd.DataFrame()
    original_count = len(df_results)
    df_filtered = df_results.copy()
    active_filtered_count = original_count

    log(f"🔎 开始对初始池 ({original_count}只) 执行基础过滤，要求: 年化>{min_annual_return * 100}%，天数>{min_day}天")

    if os.path.exists(active_cache):
        try:
            df_active = pd.read_csv(active_cache, dtype=str)
            if '基金代码' in df_active.columns:
                active_codes = set(df_active['基金代码'].str.strip().str.zfill(6).tolist())
                target_codes = _extract_target_codes(df_filtered)
                df_filtered = df_filtered[target_codes.isin(active_codes)].copy()
                active_filtered_count = len(df_filtered)
                log(f"已校验基金申购状态，剔除了 {original_count - active_filtered_count} 只暂不可申购/下架基金。")
        except Exception as e:
            log(f"⚠️ 可申购状态文件读取失败: {e}")

    if df_filtered.empty: return df_filtered

    condition = ((df_filtered['total_active_days'] > min_day) & (df_filtered['annualized_return'] > min_annual_return) &
                 (df_filtered['missing_ratio'] <= 0.05) & (df_filtered['max_zeros'] < 10) & (
                         df_filtered['max_drawdown'] > -0.7))
    df_final = df_filtered[condition].sort_values(by='annualized_return', ascending=False)

    print("\n" + "=" * 65)
    print("🎯 基金池初筛漏斗统计:")
    print(f"  1. 初始输入总数           : {original_count} 只")
    print(f"  2. 可申购状态通过         : {active_filtered_count} 只")
    print(f"  3. 财务与质量硬性达标     : {len(df_final)} 只 (保留作为高优候选池)")
    print("=" * 65 + "\n")
    return df_final


def _is_too_correlated(fund_f, selected_files, corr_matrix, downside_corr_matrix, corr_threshold,
                       downside_corr_threshold):
    has_downside = downside_corr_matrix is not None and not downside_corr_matrix.empty
    for sel_f in selected_files:
        if sel_f in corr_matrix.columns:
            corr_val = corr_matrix.loc[fund_f, sel_f]
            if pd.notna(corr_val) and corr_val > corr_threshold: return True
        if has_downside and fund_f in downside_corr_matrix.index and sel_f in downside_corr_matrix.columns:
            dc_val = downside_corr_matrix.loc[fund_f, sel_f]
            if pd.notna(dc_val) and dc_val > downside_corr_threshold: return True
    return False


def _greedy_correlation_filter(df_filtered, corr_matrix, downside_corr_matrix, corr_threshold, downside_corr_threshold):
    selected = []
    for f in df_filtered['adj_nav_file'].dropna():
        if f not in corr_matrix.columns: continue
        if not _is_too_correlated(f, selected, corr_matrix, downside_corr_matrix, corr_threshold,
                                  downside_corr_threshold): selected.append(f)
    return selected


def _prepare_results_table(batch_results):
    df = pd.DataFrame(batch_results).reindex(columns=OUTPUT_COLUMNS)
    df['组合文件名'] = df['组合文件名'].astype(str)
    df['error'] = df['error'].fillna("").astype(str)
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    for int_col in ['Total_Days', 'Max_Recovery_Days']:
        if int_col in df.columns: df[int_col] = df[int_col].fillna(0).astype(np.int16)
    return pa.Table.from_pandas(df)


def _warmup_numba():
    log("正在预热 Numba 核心引擎,防止多进程缓存踩踏...")
    _fast_simulate_path(np.zeros((100, 2), dtype=np.float64), np.array([0.5, 0.5], dtype=np.float64), 30, 0, 100, 2)


def _build_master_matrix(target_files, max_workers):
    """
    重构：强制将 master_df 的列名在拼接后直接映射为 6 位基金代码，
    后续 Worker 不再需要执行昂贵的正则解析操作。
    """
    log("🚀 正在构建全局 Master Matrix...")
    nav_series_list = _parallel_load_navs(target_files, max_workers, "构建 Master 列数据")
    master_df = pd.concat(nav_series_list, axis=1, join='outer')
    del nav_series_list

    # 将文件路径列名替换为 6 位代码
    master_df.columns = [extract_fund_code(col) for col in master_df.columns]
    sorted_codes = sorted(master_df.columns.tolist())
    master_df = master_df[sorted_codes]

    log(f"✅ 全局 Master Matrix 构建完毕,行数: {len(master_df)},列数: {len(sorted_codes)}")
    return master_df, sorted_codes


def _print_final_report(final_fund_count, combo_size, total_combos, skipped, computed, output_parquet, has_output):
    skip_ratio = (skipped / total_combos * 100) if total_combos > 0 else 0.0
    comp_ratio = (computed / total_combos * 100) if total_combos > 0 else 0.0
    print("\n" + "★" * 60)
    log(f"📊 【批处理计算任务复盘报告】 (池: {final_fund_count}只 | {combo_size}维)")
    print(f"  ➤ 理论总组合数量 : {total_combos:,} 组")
    print(f"  ➤ 命中了缓存跳过 : {skipped:,} 组 ({skip_ratio:.2f}%) 🚀")
    print(f"  ➤ 实际提交计算量 : {computed:,} 组 ({comp_ratio:.2f}%)")
    if has_output:
        print(f"  ➤ 最终文件保存至 : {output_parquet}")
    else:
        print("  ➤ 执行动作结语   : 完美跳过,本次未产生任何新计算数据。")
    print("★" * 60 + "\n")


# ====================================================================
# 新增: 升维组合引擎辅助函数
# ====================================================================
def get_previous_good_combos(prev_dim, base_pool_codes_set):
    """提取上一维度中表现优秀的组合(确保资金在 Base Pool 内)"""
    # 匹配条件: 不限制pool数量，匹配当前维度的所有parquet历史数据
    parquet_files = glob.glob(f'fund_data/fof_evaluation_results_{prev_dim}d_*.parquet')
    good_combos = set()
    for pf in parquet_files:
        try:
            df = pq.read_table(pf, columns=['组合文件名', 'Calmar_Ratio', 'Calmar_Baseline']).to_pandas()
            # 过滤条件仅为: Calmar > Calmar_Baseline
            mask = df['Calmar_Ratio'] > df['Calmar_Baseline']
            df_good = df[mask]

            for combo_str in df_good['组合文件名']:
                combo_tuple = normalize_combo_tuple(combo_str)
                if all(f in base_pool_codes_set for f in combo_tuple):
                    good_combos.add(combo_tuple)
        except Exception as e:
            log(f"读取上一维文件异常跳过 | {os.path.basename(pf)} | 错误: {e}")

    return good_combos

def generate_next_dimension_combos_apriori(N, good_prev_combos, computed_set, strict_mode=False):
    """
    高维组合极速裂变引擎 (基于 Apriori 算法原理)

    参数:
    - N: 当前要生成的目标维度 (必须 >= 3)
    - good_prev_combos: 上一维度 (N-1) 表现优秀的组合集合 (Set)
    - computed_set: 全局已计算过的组合缓存 (Set)，用于跳过历史进度
    - strict_mode: 严苛模式(默认False)。若开启，则生成的 N 维组合的所有子 N-1 组合都必须优秀才会放行。

    返回:
    - uncomputed_combos (List): 最终需要下发给多进程计算的新组合列表
    - total_combos (int): 理论生成的有效组合总数
    - global_skipped (int): 命中缓存被跳过的组合数
    """
    uncomputed_combos = []
    global_skipped = 0
    total_combos = 0

    # 将上一代组合转换为集合(如果是List)，用于严苛模式的 O(1) 极速查找
    if strict_mode and not isinstance(good_prev_combos, set):
        good_prev_combos = set(good_prev_combos)

    # 1. 建立前缀家族字典
    prefix_map = defaultdict(list)
    prefix_len = N - 2  # 共同前缀长度。例如 N=3，前缀长度为 1

    # 2. 将优秀种子按“前缀”进行家族归类
    for combo in good_prev_combos:
        prefix = combo[:prefix_len]
        prefix_map[prefix].append(combo)

    # 3. 在同前缀家族内部进行两两繁衍 (极速且无重复)
    for prefix, combos in prefix_map.items():
        n_combos = len(combos)
        # 如果该前缀下只有一个组合，形不成新的多维组合，直接基因淘汰
        if n_combos < 2:
            continue

        for i in range(n_combos):
            for j in range(i + 1, n_combos):
                # 提取两个组合的尾部基因
                tail1 = combos[i][-1]
                tail2 = combos[j][-1]

                # 强制排序，确保哈希一致性，替代昂贵的 tuple(sorted())
                if tail1 < tail2:
                    new_combo = prefix + (tail1, tail2)
                else:
                    new_combo = prefix + (tail2, tail1)

                # 【可选】严苛模式：要求 new_combo 的所有 N-1 维子集都必须是优秀的
                if strict_mode:
                    # 比如 (A, B, C, D)，那么 (A, B, C), (A, B, D), (A, C, D), (B, C, D) 必须全在上一代里
                    is_elite = True
                    for sub_combo in itertools.combinations(new_combo, N - 1):
                        if sub_combo not in good_prev_combos:
                            is_elite = False
                            break
                    if not is_elite:
                        continue

                total_combos += 1

                # 4. 查漏补缺：比对全局缓存，实现断点跳过
                if computed_set is not None and new_combo in computed_set:
                    global_skipped += 1
                else:
                    uncomputed_combos.append(new_combo)

    return uncomputed_combos, total_combos, global_skipped


# ====================================================================
# 程序入口: 升维裂变爬坡引擎
# ====================================================================
if __name__ == '__main__':
    GLOBAL_MAX_WORKERS = 25
    RESULT_CSV = 'fund_data/all_funds_result.csv'
    CORR_CSV = 'fund_data/fund_correlations.csv'
    DOWNSIDE_CSV = 'fund_data/fof_evaluation_results_2d_pool3917.csv'

    # 【配置参数】固定单次 Base Pool 筛选的 min_day 阈值，及最高计算维度
    FIXED_MIN_DAY = DEFAULT_MAX_HISTORY
    MAX_DIMENSION = 10

    _warmup_numba()

    # 步骤 1: 全局唯一初始化 (生成固定的 Base Pool)
    log("【全局初始化】正在扫描汇总文件生成唯一基础池 (Base Pool)...")
    if not os.path.exists(RESULT_CSV):
        log(f"错误: 未找到汇总文件 {RESULT_CSV}")
        exit(0)

    df_results = pd.read_csv(RESULT_CSV)
    df_filtered = filter_fund_pool(df_results, min_annual_return=0.05, min_day=FIXED_MIN_DAY)
    if df_filtered.empty:
        log("符合基础要求的基金数量为 0,退出流程。")
        exit(0)

    # 如果传入的 downside_csv 有统配符，获取第一个
    matched_downside = glob.glob(DOWNSIDE_CSV)
    ACTUAL_DOWNSIDE_CSV = matched_downside[
        0] if matched_downside else 'fund_data/fof_evaluation_results_2d_downside.csv'

    global_corr_matrix, global_downside_corr_matrix = precompute_correlations(
        result_csv=RESULT_CSV, corr_csv=CORR_CSV, downside_csv=ACTUAL_DOWNSIDE_CSV,
        max_workers=GLOBAL_MAX_WORKERS
    )

    if global_corr_matrix.empty:
        base_pool_files = df_filtered['adj_nav_file'].dropna().tolist()
    else:
        log("启动双重相关性优胜劣汰过滤...")
        base_pool_files = _greedy_correlation_filter(
            df_filtered, global_corr_matrix, global_downside_corr_matrix,
            0.8, 0.2)

    final_fund_count = len(base_pool_files)
    print("\n" + "=" * 50)
    log(f"🎯 唯一基础基金池 (Base Pool) 确定, 最终保留: {final_fund_count} 只")
    print("=" * 50 + "\n")

    if final_fund_count < 2:
        log("基础池基金数量不足以生成 2 维组合, 流程退出。")
        exit(0)

    # 将固定的 Base Pool 常驻内存，返回值已优化为纯 6 位代码
    master_df, base_pool_codes = _build_master_matrix(base_pool_files, GLOBAL_MAX_WORKERS)
    base_pool_codes_set = set(base_pool_codes)

    # 步骤 2: 维度递增循环 (N = 2 开始往上爬坡)
    N = 2
    while N <= MAX_DIMENSION:
        print(f"\n{'#' * 70}")
        log(f"【维度爬坡引擎】 🚀 正在生成和评估 {N} 维 FOF 组合")
        print(f"{'#' * 70}")

        # 2.1 获取上一维度(N-1)优秀的组合种子
        good_prev_combos = set()
        if N > 2:
            good_prev_combos = get_previous_good_combos(N - 1, base_pool_codes_set)
            log(f"自 {N - 1} 维成功提取了 {len(good_prev_combos)} 个优秀组合种子。")
            if not good_prev_combos:
                log(f"⚠️ 没有符合条件的 {N - 1} 维优秀组合, 失去升维裂变能力, 算法自然终止。")
                break

        # 2.2 加载持久化缓存，扫描当前维度(N)已存在的历史计算组合
        computed_set, cache_file = load_or_init_computed_set(N)

        # 2.3 依据基因裂变逻辑生成新维度的组合 (边裂变、边去重、边过滤)
        log(f"正在交叉组合生成 {N} 维的待测序列池(启用极速剪枝模式)...")
        uncomputed_combos = []
        global_skipped = 0
        total_combos = 0

        if N == 2:
            # 2 维: 从 Base Pool 中进行两两全量组合
            for c in itertools.combinations(base_pool_codes, 2):
                # c 默认即已排序
                total_combos += 1
                if c not in computed_set:
                    uncomputed_combos.append(c)
                else:
                    global_skipped += 1
        else:
            uncomputed_combos, total_combos, global_skipped = generate_next_dimension_combos_apriori(
                N=N,
                good_prev_combos=good_prev_combos,
                computed_set=computed_set,
                strict_mode=True  # 设置为 True 会进一步大幅压缩数量，保留最顶尖的硬核组合
            )

        log(f"总计生成了 {total_combos:,} 个符合条件的 {N} 维理论组合。 其中 {global_skipped:,} 个命中缓存被跳过，剩余 {len(uncomputed_combos):,} 个待评估组合将进入下一阶段。占比跳过 {global_skipped / total_combos * 100:.2f}%")

        if total_combos == 0:
            log("生成的待评估组合数量为 0, 流程结束。")
            break

        if not uncomputed_combos:
            log(f"✅ {N} 维理论组合共 {total_combos:,} 个，已全部在缓存中，直接进入下一维度。")
            N += 1
            continue

        # 2.4 设置输出目标文件(名称带上 pool 和 min_day 严格符合原规则)
        output_parquet = f'fund_data/fof_evaluation_results_{N}d_pool{final_fund_count}_min_day_{FIXED_MIN_DAY}.parquet'
        if os.path.exists(output_parquet):
            log(f"已发现现有文件 {output_parquet},将直接跳过计算并进入下一维度。")
            N += 1
            continue
        ensure_dir(output_parquet)

        writer = None
        global_computed = 0

        # 2.5 执行批处理评估下发
        with ProcessPoolExecutor(max_workers=GLOBAL_MAX_WORKERS,
                                 initializer=_init_worker,
                                 initargs=(master_df,)) as executor:
            with tqdm(total=total_combos, desc=f"计算 {N} 维", unit="组") as pbar:

                # 提前更新已经被缓存跳过的进度条
                if global_skipped > 0:
                    pbar.update(global_skipped)

                # 直接下发代码元组，无需还原为文件路径 (性能大幅提升)
                job_list = uncomputed_combos

                # 嵌套分片下发 (降低 IPC 序列化开销)
                for i in range(0, len(job_list), BATCH_SIZE):
                    batch = job_list[i:i + BATCH_SIZE]
                    chunks = [batch[j:j + CHUNK_SIZE] for j in range(0, len(batch), CHUNK_SIZE)]

                    futures = [executor.submit(_worker_process_chunk, chunk) for chunk in chunks]

                    batch_results = []
                    for future in as_completed(futures):
                        chunk_res = future.result()
                        batch_results.extend(chunk_res)
                        pbar.update(len(chunk_res))

                    # Batch结果落盘
                    if batch_results:
                        global_computed += len(batch_results)
                        table = _prepare_results_table(batch_results)
                        if writer is None:
                            writer = pq.ParquetWriter(output_parquet, table.schema, compression='zstd')
                        writer.write_table(table)

                    del futures, batch_results, batch, chunks

        # 2.6 当前维度收尾操作
        has_output = writer is not None
        if has_output:
            writer.close()

        _print_final_report(final_fund_count, N, total_combos,
                            global_skipped, global_computed, output_parquet, has_output)

        # 2.7 动态将本轮运算出的新 Parquet 文件结果更新到全局缓存 pkl
        _update_cache_from_new_file(output_parquet, cache_file, computed_set)

        # 维数 + 1 准备向更高维度发起挑战
        N += 1