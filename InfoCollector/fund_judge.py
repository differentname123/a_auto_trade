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
VETO_MAX_MDD = 0.5
VETO_HURDLE_RATE = 0.05
VETO_AR1 = 0.55
VETO_VOL = 0.03
VETO_RECOVERY_DAYS = 350
VETO_MISSING_RATIO = 0.05
VETO_CONTINUOUS_ZEROS = 10
VETO_CALMAR_MULTIPLIER = 1.01
VETO_WORST_1Y = -0.15
VETO_ZOMBIE_DAYS = 35

# 打分参数
TAIL_RISK_SENSITIVITY = 2.5
TIME_SATURATION_DAYS = DEFAULT_MAX_HISTORY
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
    # 【核心修改】：字符串拆分后立刻降级为基础整型，极大节约内存
    parts = [int(str(x).strip()) for x in str(combo_str).split(sep)]
    return tuple(sorted(parts))


# ====================================================================
# 缓存自更新系统 (按要求维度泛化并持久化) - 已升级至极致压缩的 Parquet + Int 引擎
# ====================================================================
def _save_set_to_parquet(combo_set, dim, filepath):
    try:
        if not combo_set: return

        # 【修复】：将 Set 转为 List 固化，然后通过推导式按列剥离，彻底避开 * 解包
        combo_list = list(combo_set)
        cols = [[combo[i] for combo in combo_list] for i in range(dim)]

        arrays = [pa.array(col, type=pa.int32()) for col in cols]
        names = [f'dim_{i}' for i in range(dim)]
        table = pa.Table.from_arrays(arrays, names=names)
        pq.write_table(table, filepath, compression='zstd')

        # 显式释放巨大内存
        del combo_list, cols, arrays, table
    except Exception as e:
        log(f"❌ 缓存 Parquet 持久化失败: {e}")


def load_or_init_computed_set(curr_dim):
    """
    【重构】：加载或初始化全局缓存，支持读取新版 Parquet 并自动兼容升级旧版 pkl
    """
    cache_file = f'fund_data/computed_combos_{curr_dim}d_global_cache.parquet'
    old_pkl_file = f'fund_data/computed_combos_{curr_dim}d_global_cache.pkl'
    parquet_files = glob.glob(f'fund_data/fof_evaluation_results_{curr_dim}d_*.parquet')

    computed_set = set()
    cache_time = 0

    # 1. 优先尝试加载全新版 Parquet 缓存
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        try:
            table = pq.read_table(cache_file)
            # 还原为 Set[Tuple[int]]
            cols = [table.column(f'dim_{i}').to_pylist() for i in range(curr_dim)]
            computed_set = set(zip(*cols))
            log(f"📦 命中全局 Parquet 缓存 | 规则: {curr_dim} 维 | 已加载 {len(computed_set):,} 个历史元组。")
        except Exception as e:
            log(f"⚠️ Parquet 缓存读取失败将重建: {e}")
            cache_time = 0

    # 2. 嗅探并静默升级历史的老 PKL 文件
    elif os.path.exists(old_pkl_file):
        try:
            with open(old_pkl_file, 'rb') as f:
                old_set = pickle.load(f)
            log(f"♻️ 发现历史 PKL 缓存，正在执行内存降维与 Parquet 转换升级...")
            # 循环遍历，将历史字符串 Tuple 强制转换压缩为 Int Tuple
            for combo in old_set:
                computed_set.add(tuple(int(x) for x in combo))

            # 立刻以全新 Parquet 格式落盘，并封存历史 pkl
            _save_set_to_parquet(computed_set, curr_dim, cache_file)
            os.rename(old_pkl_file, old_pkl_file + ".bak")
            cache_time = os.path.getmtime(cache_file)
            log(f"✅ PKL 缓存已成功升级为 Parquet | 规则: {curr_dim} 维 | 容量: {len(computed_set):,}。")
        except Exception as e:
            log(f"⚠️ 历史 PKL 缓存升级失败: {e}")

    log(f"🔍 扫描 Parquet 文件进行增量更新 | 规则: {curr_dim} 维 | 共发现 {len(parquet_files)} 个相关文件。")
    updated = False
    for pf in parquet_files:
        if os.path.getmtime(pf) <= cache_time:
            continue
        try:
            parquet_file = pq.ParquetFile(pf)
            file_temp_set = set()

            for batch in parquet_file.iter_batches(batch_size=100000, columns=['组合文件名']):
                raw_combos = batch.column('组合文件名').to_pylist()
                file_temp_set.update(normalize_combo_tuple(c) for c in raw_combos if c)
                del raw_combos, batch
            del parquet_file

            before_len = len(computed_set)
            computed_set.update(file_temp_set)
            added = len(computed_set) - before_len
            del file_temp_set

            if added > 0:
                updated = True
                log(f"🔄 增量读取文件 | {os.path.basename(pf)} | 新增 {added:,} 个有效组合。")
        except Exception as e:
            log(f"❌ 读取文件异常跳过 | {os.path.basename(pf)} | 错误: {e}")

    if updated or (not os.path.exists(cache_file) and computed_set):
        _save_set_to_parquet(computed_set, curr_dim, cache_file)
        log(f"✅ 缓存已持久化 (Parquet) | 规则: {curr_dim} 维 | 当前全局总库容量: {len(computed_set):,} 个。")
    elif not parquet_files and not computed_set:
        log(f"ℹ️ 空白初始化 | 规则: {curr_dim} 维 | 暂无历史计算数据。")

    return computed_set, cache_file


def _update_cache_from_new_file(output_file, cache_file, computed_set):
    """【重构】：根据新生成的结果文件,动态追加更新组合缓存 (写出为Parquet)"""
    if not output_file or not os.path.exists(output_file): return
    cache_time = os.path.getmtime(cache_file) if os.path.exists(cache_file) else 0
    if os.path.getmtime(output_file) <= cache_time: return

    log("🌟 嗅探到新生成的结果文件更新于缓存,正在动态追加至持久化系统...")
    try:
        if os.path.getsize(output_file) <= 100: return

        parquet_file = pq.ParquetFile(output_file)
        added = 0

        for batch in parquet_file.iter_batches(batch_size=100000, columns=['组合文件名']):
            raw_combos = batch.column('组合文件名').to_pylist()
            for c in raw_combos:
                if c:
                    combo_tup = normalize_combo_tuple(c)
                    if combo_tup not in computed_set:
                        computed_set.add(combo_tup)
                        added += 1
            del raw_combos, batch
        del parquet_file

        if added > 0:
            dim = len(next(iter(computed_set))) if computed_set else 0
            if dim > 0:
                _save_set_to_parquet(computed_set, dim, cache_file)
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

        # 初始化带有全部 VETO 标志的底座，为了无缝衔接外部的原有跟踪日志
        metrics = {
            'Start_Date': merged_nav.index[0].strftime('%Y-%m-%d'),
            'End_Date': merged_nav.index[-1].strftime('%Y-%m-%d'),
            'Total_Days': n_days, 'n_funds': n_funds, 'Avg_CAGR': avg_cagr, 'Avg_Max_Drawdown': avg_mdd,
            'Calmar_Baseline': calmar_baseline, 'CAGR': 0.0, 'Max_Drawdown': 0.0,
            'Max_Recovery_Days': 0, 'Worst_Rolling_1Y_Return': 0.0, 'Worst_Rolling_1Y_R2': 0.0,
            'AR1_Coefficient': 1.0, 'Annualized_Volatility': 0.0, 'Sharpe_Ratio': 0.0,
            'Calmar_Ratio': 0.0, 'Daily_Win_Rate': 0.0, 'Downside_Correlation': 1.0,
            "VETO_Hurdle_Rate": False, "VETO_Drawdown_Crash": False, "VETO_Fake_Smooth": False,
            "VETO_Endless_Bleeding": False, "VETO_Data_Distortion": False,
            "VETO_Below_Calmar_Baseline": False, "VETO_Worst_1Y_Crash": False
        }

        # --- 第 0 层拦截：极端数据缺失 ---
        if (max_missing_ratio > VETO_MISSING_RATIO) or (max_continuous_zeros > VETO_CONTINUOUS_ZEROS):
            metrics["VETO_Data_Distortion"] = True
            return metrics, 0.0

        # --- 第 1 层短路：基础收益与回撤拦截 ---
        cagr = float(synth_eq.iloc[-1] ** (TRADING_DAYS_PER_YEAR / n_days) - 1)
        cum_max = synth_eq.cummax().clip(lower=1.0)
        drawdowns = (synth_eq - cum_max) / cum_max
        max_dd = float(drawdowns.min())
        calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

        metrics.update({'CAGR': cagr, 'Max_Drawdown': max_dd, 'Calmar_Ratio': calmar})

        v_hurdle = cagr < hurdle_rate
        v_dd = abs(max_dd) > max_mdd_limit
        v_calmar = calmar <= (calmar_baseline * VETO_CALMAR_MULTIPLIER)

        if v_hurdle or v_dd or v_calmar:
            metrics["VETO_Hurdle_Rate"] = v_hurdle
            metrics["VETO_Drawdown_Crash"] = v_dd
            metrics["VETO_Below_Calmar_Baseline"] = v_calmar
            return metrics, 0.0

        # --- 第 2 层短路：恢复天数与波动平滑度拦截 ---
        is_dd = drawdowns < 0
        max_recovery_days = int(is_dd.groupby((~is_dd).cumsum()).sum().max())
        vol_annual = float(synth_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        ar1 = synth_ret.autocorr(lag=1)
        ar1 = float(ar1) if pd.notna(ar1) else 1.0

        metrics.update(
            {'Max_Recovery_Days': max_recovery_days, 'Annualized_Volatility': vol_annual, 'AR1_Coefficient': ar1})

        v_bleeding = max_recovery_days > VETO_RECOVERY_DAYS
        v_fake = (ar1 > VETO_AR1) and (vol_annual < VETO_VOL)

        if v_bleeding or v_fake:
            metrics["VETO_Endless_Bleeding"] = v_bleeding
            metrics["VETO_Fake_Smooth"] = v_fake
            return metrics, 0.0

        # --- 第 3 层短路：滚动 1 年表现拦截 ---
        worst_1y = _compute_worst_rolling_1y(synth_eq.values, n_days)
        metrics.update({'Worst_Rolling_1Y_Return': worst_1y})

        v_worst_1y = worst_1y < VETO_WORST_1Y
        if v_worst_1y:
            metrics["VETO_Worst_1Y_Crash"] = True
            return metrics, 0.0

        # --- 第 4 层：只有活下来的幸存者，才执行极致耗时的指标计算 ---
        worst_1y_r2 = _compute_worst_rolling_1y_r2(synth_eq.values, n_days)
        downside_corr = _compute_downside_correlation(synth_ret, fund_rets)
        sharpe = float(cagr / vol_annual) if vol_annual > 1e-6 else 0.0
        win_rate = float((synth_ret > 0).sum() / n_days) if n_days > 0 else 0.0

        metrics.update({
            'Worst_Rolling_1Y_R2': worst_1y_r2,
            'Downside_Correlation': downside_corr,
            'Sharpe_Ratio': sharpe,
            'Daily_Win_Rate': win_rate
        })

        return metrics, _compute_total_score(metrics, n_days)

    # ==========================
    # 以下为你原有的外层扰动测试逻辑，完全未修改
    # ==========================
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

    # final_metrics['Total_Score'] = score_base


    return final_metrics


# ====================================================================
# Worker 工作进程函数 (优化:消灭多进程内的 re 正则解析)
# ====================================================================
def _worker_process_combo(combo_codes):
    """单组合处理:传入的已是极简的 Tuple[int]，在最终结果呈现时还原为 6 位代码"""
    global WORKER_MASTER_DF
    # 【核心修改】：通过格式化补齐 0，将 (1, 2) 还原成 "000001_000002"
    combo_name = "_".join([f"{c:06d}" for c in combo_codes])
    try:
        # WORKER_MASTER_DF 的列名现已是 int，直接通过整数列表取列
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

        nav_series = df['复权净值'].rename(filepath)
        recent_series = nav_series.iloc[-DEFAULT_MAX_HISTORY:] if len(nav_series) > DEFAULT_MAX_HISTORY else nav_series

        # 严格使用 ffill 防止 NaN 吞噬异常涨跌
        daily_returns = recent_series.ffill().pct_change().dropna()

        if not daily_returns.empty:
            max_jump = float(daily_returns.max())

            # ---------------------------------------------------------
            # 🛡️ 终极双轨制拦截 (Dual-Track Filter)
            # ---------------------------------------------------------

            # 轨道 1：绝对物理极限拦截 (31.5%)
            # 涵盖A股主板、创业板、科创板、北交所(30%)及美股纳斯达克的物理极限。
            # 超过此值，连爱因斯坦来了这也是脏数据或严重赎回，无脑击杀。
            if max_jump > 0.315:
                return None

            # 轨道 2：固收类资产暗病拦截 (隐式分类)
            # 如果涨幅没有超过 31.5%，但超过了 6% (0.06)
            if max_jump > 0.06:
                # 我们将最大的 3 天涨幅剔除，看看它"平时"是个什么脾气
                # 这样可以防止单日暴涨本身把波动率污染了
                normal_returns = daily_returns.sort_values().iloc[:-3]
                normal_vol = float(normal_returns.std())

                # 如果它平时的日波动率小于 0.006 (0.6%)，这绝对是一只固收类/偏债基金
                # 一只偏债基金单日涨超 6%，100% 是遭遇了巨额赎回，精准斩首。
                # (而股票基金平时的波动率通常在 1.0%~2.5% 之间，不会触发此条件)
                if normal_vol < 0.006:
                    return None

        return nav_series
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


def filter_fund_pool(df_results, active_cache='temp/active_fund_codes_1741_剔除增强指数_剔除主动债_剔除细分行业_剔地方信用债.csv', min_annual_return=0.15, min_day=1000):
    if df_results is None or df_results.empty: return pd.DataFrame()
    original_count = len(df_results)
    df_filtered = df_results.copy()
    active_filtered_count = original_count

    log(f"🔎 开始对初始池 ({original_count}只) 执行基础过滤，要求: 年化>{min_annual_return * 100}%，天数>{min_day}天")

    rejection_records = []

    if os.path.exists(active_cache):
        try:
            df_active = pd.read_csv(active_cache, dtype=str)
            if '基金代码' in df_active.columns:
                active_codes = set(df_active['基金代码'].str.strip().str.zfill(6).tolist())
                target_codes = _extract_target_codes(df_filtered)
                is_active = target_codes.isin(active_codes)

                # 记录因申购状态被剔除的基金
                rejected_active = df_filtered[~is_active]
                for idx, row in rejected_active.iterrows():
                    code = target_codes.loc[idx]
                    rejection_records.append({
                        '基金代码': code,
                        '文件': row.get('adj_nav_file', ''),
                        '筛选阶段': '初筛-申购状态',
                        '剔除原因': '该基金暂不可申购或已下架，未在可申购白名单(active_cache)内。'
                    })

                df_filtered = df_filtered[is_active].copy()
                active_filtered_count = len(df_filtered)
                log(f"已校验基金申购状态，剔除了 {original_count - active_filtered_count} 只暂不可申购/下架基金。")
        except Exception as e:
            log(f"⚠️ 可申购状态文件读取失败: {e}")

    if df_filtered.empty:
        ensure_dir('fund_data/base_pool_rejection_reasons.csv')
        if rejection_records:
            pd.DataFrame(rejection_records).to_csv('fund_data/base_pool_rejection_reasons.csv', index=False,
                                                   encoding='utf-8-sig')
        return df_filtered

    # 短路逻辑生成具体剔除原因，利用 not 操作自动处理 NaN 情况，与原布尔逻辑严格等价
    def _get_reject_reason(row):
        if not (row['total_active_days'] > min_day):
            return f"运行天数不足，当前天数{row['total_active_days']}天，目标阈值>{min_day}天。"
        if not (row['annualized_return'] > min_annual_return):
            return f"年化收益不达标，当前年化{row['annualized_return'] * 100:.2f}%，目标阈值>{min_annual_return * 100:.2f}%。"
        if not (row['missing_ratio'] <= 0.05):
            return f"数据缺失率过高，当前缺失率{row['missing_ratio'] * 100:.2f}%，目标阈值<=5.00%。"
        if not (row['max_zeros'] < 10):
            return f"连续零收益异常，当前最大连续{row['max_zeros']}天，目标阈值<10天。"
        if not (row['max_drawdown'] > -0.7):
            return f"最大回撤已击穿底线，当前回撤{row['max_drawdown'] * 100:.2f}%，目标阈值>-70.00%。"
        return ""

    df_filtered['剔除原因'] = df_filtered.apply(_get_reject_reason, axis=1)

    # === 新增代码：统计5种财务硬性指标的驳回个数 ===
    reject_counts = {
        "运行天数不足": df_filtered['剔除原因'].str.startswith("运行天数不足").sum(),
        "年化收益不达标": df_filtered['剔除原因'].str.startswith("年化收益不达标").sum(),
        "数据缺失率过高": df_filtered['剔除原因'].str.startswith("数据缺失率过高").sum(),
        "连续零收益异常": df_filtered['剔除原因'].str.startswith("连续零收益异常").sum(),
        "最大回撤已击穿底线": df_filtered['剔除原因'].str.startswith("最大回撤已击穿底线").sum()
    }
    # ==========================================

    rejected_metrics = df_filtered[df_filtered['剔除原因'] != ""]
    target_codes_filtered = _extract_target_codes(df_filtered)

    for idx, row in rejected_metrics.iterrows():
        code = target_codes_filtered.loc[idx]
        rejection_records.append({
            '基金代码': code,
            '文件': row.get('adj_nav_file', ''),
            '筛选阶段': '初筛-财务硬性指标',
            '剔除原因': row['剔除原因']
        })

    # 初筛是全新流程的起点，每次执行此函数时全新覆盖写入文件，确保记录文件干净且为最新版本
    reason_file = 'fund_data/base_pool_rejection_reasons.csv'
    ensure_dir(reason_file)
    if rejection_records:
        pd.DataFrame(rejection_records).to_csv(reason_file, index=False, encoding='utf-8-sig')
    else:
        pd.DataFrame(columns=['基金代码', '文件', '筛选阶段', '剔除原因']).to_csv(reason_file, index=False,
                                                                                  encoding='utf-8-sig')

    condition = df_filtered['剔除原因'] == ""
    df_final = df_filtered[condition].drop(columns=['剔除原因']).sort_values(by='annualized_return', ascending=False)

    # === 修改代码：在漏斗日志中追加5种驳回分布的打印 ===
    print("\n" + "=" * 65)
    print("🎯 基金池初筛漏斗统计:")
    print(f"  1. 初始输入总数           : {original_count} 只")
    print(f"  2. 可申购状态通过         : {active_filtered_count} 只 (剔除 {original_count - active_filtered_count}只)")
    print(f"  3. 财务与质量硬性达标     : {len(df_final)} 只 (保留作为高优候选池)")
    print("     [财务硬性指标驳回分布]")
    print(f"      - 运行天数不足        : {reject_counts['运行天数不足']} 只")
    print(f"      - 年化收益不达标      : {reject_counts['年化收益不达标']} 只")
    print(f"      - 数据缺失率过高      : {reject_counts['数据缺失率过高']} 只")
    print(f"      - 连续零收益异常      : {reject_counts['连续零收益异常']} 只")
    print(f"      - 最大回撤已击穿底线  : {reject_counts['最大回撤已击穿底线']} 只")
    print("=" * 65 + "\n")
    # ==========================================

    return df_final


def _is_too_correlated(fund_f, selected_files, corr_matrix, downside_corr_matrix, corr_threshold,
                       downside_corr_threshold):
    has_downside = downside_corr_matrix is not None and not downside_corr_matrix.empty
    for sel_f in selected_files:
        if sel_f in corr_matrix.columns:
            corr_val = corr_matrix.loc[fund_f, sel_f]
            if pd.notna(corr_val) and corr_val > corr_threshold:
                f2_code = extract_fund_code(sel_f)
                return True, f"全天候相关性超标，与已保留标的[{f2_code}]的相关性达{corr_val:.4f}，目标阈值<={corr_threshold}。"
        if has_downside and fund_f in downside_corr_matrix.index and sel_f in downside_corr_matrix.columns:
            dc_val = downside_corr_matrix.loc[fund_f, sel_f]
            if pd.notna(dc_val) and dc_val > downside_corr_threshold:
                f2_code = extract_fund_code(sel_f)
                return True, f"下行相关性超标，与已保留标的[{f2_code}]的下行相关性达{dc_val:.4f}，目标阈值<={downside_corr_threshold}。"
    return False, ""


def _greedy_correlation_filter(df_filtered, corr_matrix, downside_corr_matrix, corr_threshold, downside_corr_threshold):
    selected = []
    action_records = []  # 改名：不仅记录 rejection，也记录 selection

    for f in df_filtered['adj_nav_file'].dropna():
        if f not in corr_matrix.columns:
            # action_records.append({
            #     '基金代码': extract_fund_code(f),
            #     '文件': f,
            #     '筛选阶段': '次筛-相关性过滤',
            #     '剔除原因': '未能在全局相关性矩阵中找到对应数据，无法执行安全评估，强制剔除。'
            # })

            selected.append(f)
            action_records.append({
                '基金代码': extract_fund_code(f),
                '文件': f,
                '筛选阶段': '次筛-相关性过滤',
                '剔除原因': '【成功入选】未能在全局相关性矩阵中找到对应数据，无法执行安全评估，但考虑到不应过于苛刻，暂时保留进入下一轮。'
            })

            continue

        is_corr, reason = _is_too_correlated(f, selected, corr_matrix, downside_corr_matrix, corr_threshold,
                                             downside_corr_threshold)

        if not is_corr:
            selected.append(f)
            # 【核心修复】：补齐全链路闭环，将最终存活进入 Base Pool 的基金也记录下来
            action_records.append({
                '基金代码': extract_fund_code(f),
                '文件': f,
                '筛选阶段': '最终结果',
                '剔除原因': '【成功入选】顺利通过财务初筛与相关性测试，已进入 Base Pool。'
            })
        else:
            action_records.append({
                '基金代码': extract_fund_code(f),
                '文件': f,
                '筛选阶段': '次筛-相关性过滤',
                '剔除原因': reason
            })

    # 追加记录到初筛创建的原因收集文件中，实现全链路闭环审计
    if action_records:
        reason_file = 'fund_data/base_pool_rejection_reasons.csv'  # 注: 文件虽叫rejection, 现已包含完整漏斗结果
        df_action = pd.DataFrame(action_records)
        ensure_dir(reason_file)

        # 为了保证成功入选的基金排在文件的最前面，我们可以对本次追加的数据做个排序
        # 让“最终结果”阶段的记录置顶
        df_action['is_selected'] = df_action['筛选阶段'] == '最终结果'
        df_action = df_action.sort_values(by=['is_selected', '基金代码'], ascending=[False, True]).drop(
            columns=['is_selected'])

        if os.path.exists(reason_file):
            df_action.to_csv(reason_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_action.to_csv(reason_file, index=False, encoding='utf-8-sig')

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
    重构：强制将 master_df 的列名直接映射为 Int 整型，
    配合缓存层的纯整型升级，剥离所有字符串开销。
    """
    log("🚀 正在构建全局 Master Matrix...")
    nav_series_list = _parallel_load_navs(target_files, max_workers, "构建 Master 列数据")
    master_df = pd.concat(nav_series_list, axis=1, join='outer')
    del nav_series_list

    # 【核心修改】：解析出的基金代码提取数字部分，作为整型列名
    master_df.columns = [int(extract_fund_code(col)) for col in master_df.columns]
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


def fast_estimate_combos(N, seeds_list, strict_mode=True):
    """纯内存极速测算给定种子在 N 维情况下的理论生成量 (无文件I/O)"""
    seeds_set = set(seeds_list)
    prefix_map = defaultdict(list)
    prefix_len = N - 2

    for combo in seeds_list:
        prefix = combo[:prefix_len]
        prefix_map[prefix].append(combo)

    total_count = 0
    for prefix, combos in prefix_map.items():
        n_combos = len(combos)
        if n_combos < 2:
            continue
        for i in range(n_combos):
            for j in range(i + 1, n_combos):
                tail1 = combos[i][-1]
                tail2 = combos[j][-1]

                if tail1 < tail2:
                    new_combo = prefix + (tail1, tail2)
                else:
                    new_combo = prefix + (tail2, tail1)

                if strict_mode:
                    is_elite = True
                    for idx in range(N):
                        sub_combo = new_combo[:idx] + new_combo[idx + 1:]
                        if sub_combo not in seeds_set:
                            is_elite = False
                            break
                    if not is_elite:
                        continue
                total_count += 1
    return total_count



def calculate_dynamic_k_by_binary_search(N, candidate_list, target_min=45000000, target_max=55000000, strict_mode=True):
    """基于对数空间插值与安全阻尼步进的高速测算"""
    if len(candidate_list) < 2:
        return len(candidate_list), 0

    high_total = len(candidate_list)
    target_mid = (target_min + target_max) // 2
    log(f"🧠 [测算引擎] 启动非线性测算器，全量候补: {high_total:,}，目标区间 [{target_min:,}, {target_max:,}]")

    # ==========================================
    # 第一步：初始探测 (建立基准锚点)
    # ==========================================
    init_k = min(high_total // 2, 1000000)
    init_k = max(2, init_k)  # <--- 加上这一行，保证种子数量至少为2

    # 强烈建议: 如果 fast_estimate_combos 支持传入 limit 参数，不要用切片!
    # count_init = fast_estimate_combos(N, candidate_list, strict_mode, limit=init_k)
    count_init = fast_estimate_combos(N, candidate_list[:init_k], strict_mode)

    log(f"🧠 [测算引擎] 初始锚点: K={init_k:,} -> 生成量: {count_init:,}")

    best_diff = abs(count_init - target_mid)
    best_k = init_k
    best_count = count_init

    if target_min <= count_init <= target_max:
        return init_k, count_init

    # ==========================================
    # 第二步：寻找安全上下界 (防止掉入复杂度黑洞)
    # ==========================================
    if count_init < target_mid:
        low, count_low = init_k, count_init
        high, count_high = high_total, None
    else:
        # 初始估值过大，补测绝对物理下界以获取真实锚点
        low = 2
        count_low = fast_estimate_combos(N, candidate_list[:low], strict_mode)
        high, count_high = init_k, count_init

        # 🌟 终态修复：物理下界拦截。如果连 k=2 都已经超标，说明目标区间设定得不切实际，直接返回物理最小值。
        if count_low > target_max:
            log(f"⚠️ [绝对下限] 物理极限 K=2 的生成量({count_low:,})仍大于上限，强制收敛。")
            return low, count_low
        elif target_min <= count_low <= target_max:
            return low, count_low

    # 如果我们只有下界没有上界，绝不使用全量求中值！使用“基于多项式预估”的安全步进
    prev_k = None
    prev_count = None

    while count_high is None and low < high_total:
        shortfall_ratio = target_mid / max(1, count_low)

        # 引入“自适应斜率（动态求导）”
        is_dynamic = False
        if prev_k is not None and count_low > prev_count and low > prev_k:
            try:
                # 🌟 修复 1：防止 math.log(0) 导致 ValueError 崩溃
                log_c_diff = math.log(max(1, count_low)) - math.log(max(1, prev_count))
                log_k_diff = math.log(low) - math.log(prev_k)

                E = log_c_diff / log_k_diff
                E = max(1.1, min(E, float(N)))

                # 🌟 修复 2：加入越界偏置 (*1.05) 和 强制最小步进 (max(1.15, ...))
                # 目的：我们是为了寻找上限(Overshoot)，必须打破逼近停滞！
                predicted_multiplier = math.pow(shortfall_ratio, 1 / E)
                safe_multiplier = min(3.5, max(1.15, predicted_multiplier * 1.05))
                is_dynamic = True
            except (ZeroDivisionError, ValueError):  # 🌟 补全异常捕获
                # 异常时回退，同样赋予越界偏置和最低步幅
                safe_multiplier = min(2.0, max(1.15, math.pow(shortfall_ratio, 1 / max(1, N)) * 1.05))
        else:
            # 初始数据不足时，用稍保守但不至于停滞的倍率探路 (上限调高到2.0防止初期走得太慢)
            safe_multiplier = min(2.0, max(1.15, math.pow(shortfall_ratio, 1 / max(1, N)) * 1.05))

        # 保证至少向前推进 10 步，防止 low+1 导致死循环
        next_k = min(high_total, max(low + 10, int(low * safe_multiplier)))

        if is_dynamic:
            log(f"🛡️ [自适应步进] 尚未找到上限！当前感知维度 E={E:.2f}，动态预估倍率: {safe_multiplier:.2f}x，向 K={next_k:,} 探索...")
        else:
            log(f"🛡️ [安全步进] 尚未找到上限！数学预估倍率: {safe_multiplier:.2f}x，向 K={next_k:,} 探索...")

        count = fast_estimate_combos(N, candidate_list[:next_k], strict_mode)
        log(f"✅ [测算完成] K={next_k:,} -> 生成量: {count:,}")

        # 更新最佳记录
        if abs(count - target_mid) < best_diff:
            best_diff, best_k, best_count = abs(count - target_mid), next_k, count

        if target_min <= count <= target_max:
            return next_k, count

        if count > target_max:
            high, count_high = next_k, count
            break  # 成功跨越界限，夹逼出安全区间！交给光速的第三步！
        else:
            # 更新历史记录用于下一轮求导
            prev_k, prev_count = low, count_low
            low, count_low = next_k, count

    # 🌟 修复点 1：拦截算力枯竭黑天鹅
    if count_high is None:
        log(f"⚠️ [算力枯竭] 全量候补池 {high_total:,} 的生成量仍低于目标中值，停止测算。")
        return best_k, best_count

    # ==========================================
    # 第三步：在安全的 [low, high] 区间内进行【对数插值查找】
    # ==========================================
    log(f"🎯 [锁定区间] 已夹逼出安全测算区间 K ∈ [{low:,}, {high:,}]，开始精细对数插值...")

    while low <= high:
        # 精度控制：区间足够小，提前结束
        if high - low < 1000:
            log(f"⚠️ [精度达标] 上下限差值极小，停止探索。")
            break

        # 【核心优化】：对数空间插值，完美契合组合数非线性增长的特性
        try:
            log_low_k, log_high_k = math.log(low), math.log(high)
            log_low_c = math.log(max(1, count_low))
            log_high_c = math.log(max(1, count_high))
            log_target = math.log(target_mid)

            # 在对数空间计算斜率和预测值
            ratio = (log_target - log_low_c) / (log_high_c - log_low_c)
            log_mid_k = log_low_k + ratio * (log_high_k - log_low_k)
            mid_guess = int(math.exp(log_mid_k))

            # 防止浮点误差导致的越界
            mid = max(low + 1, min(high - 1, mid_guess))
        except (ValueError, ZeroDivisionError):  # 🌟 修复点 2：补全除零异常防御
            # 异常回退为标准二分
            mid = (low + high) // 2

        log(f"🔬 [对数插值] 预测最佳 K={mid:,} (当前区间: [{low:,}, {high:,}])...")
        count = fast_estimate_combos(N, candidate_list[:mid], strict_mode)

        diff = abs(count - target_mid)
        if diff < best_diff:
            best_diff, best_k, best_count = diff, mid, count

        if target_min <= count <= target_max:
            log(f"🎉 [测算命中] 精准落入目标区间！最终锁定 K={mid:,}，生成量: {count:,}")
            return mid, count

        if count > target_max:
            high, count_high = mid, count  # 🌟 修复点 3：保持真实坐标绑定，防止插值畸变
        else:
            low, count_low = mid, count  # 🌟 修复点 3：保持真实坐标绑定，防止插值畸变

    log(f"🏁 [测算结束] 采用最佳逼近 K={best_k:,}，生成量: {best_count:,}")
    return best_k, best_count


def get_previous_good_combos(prev_dim, base_pool_codes_set, target_dim, target_max_combos=1000000):
    # 🌟 核心杀手锏：自带短路拦截的极速转换函数 (一次 Split，终身受用)
    def fast_filter_and_normalize(combo_str):
        # 【核心修改】：因为外部的 base_pool_codes_set 已经全部是整数，这里需保持一致判断
        items = [int(x.strip()) for x in str(combo_str).split('_')]
        if not base_pool_codes_set.issuperset(items):
            return None
        return tuple(sorted(items))

    parquet_files = glob.glob(f'fund_data/fof_evaluation_results_{prev_dim}d_*.parquet')
    all_clean_records = []

    total_read_rows = 0
    passed_calmar_count = 0

    for pf in parquet_files:
        try:
            # 修改1：额外读取 Total_Score 列 (读取时已包含 CAGR)
            df = pq.read_table(pf, columns=['组合文件名', 'Calmar_Ratio', 'Calmar_Baseline', 'Total_Score', 'CAGR']).to_pandas()
            # 兼容处理：防止有些老文件缺失 Total_Score，填充为 0
            if 'Total_Score' not in df.columns:
                df['Total_Score'] = 0
            else:
                df['Total_Score'] = df['Total_Score'].fillna(0)

            total_read_rows += len(df)

            # 门槛 1：卡玛基准过滤
            df = df[df['Calmar_Ratio'] > df['Calmar_Baseline']]
            passed_calmar_count += len(df)

            if df.empty:
                continue

            # 🚀 门槛 2 & 3：短路过滤 + 乱序归一化
            df['Combo_Tuple'] = df['组合文件名'].apply(fast_filter_and_normalize)

            # 瞬间剔除所有被返回 None 的历史孤儿
            df = df[df['Combo_Tuple'].notna()]

            if df.empty:
                continue

            # 门槛 4：单文件局部去重 (修改：Total_Score 优先，CAGR 其次)
            df = df.sort_values(['Total_Score', 'CAGR'], ascending=[False, False]).drop_duplicates(
                subset=['Combo_Tuple'])

            # 极致压缩内存，仅保留核心字段 (修改：保留 Total_Score 和后续排序必须的 CAGR)
            all_clean_records.append(df[['Combo_Tuple', 'Calmar_Ratio', 'Total_Score', 'CAGR']])

        except Exception as e:
            log(f"读取上一维文件异常跳过 | {os.path.basename(pf)} | 错误: {e}")

    if not all_clean_records:
        log("⚠️ 未读取到任何符合基础条件的上一维历史数据。")
        return set()

    # 5. 合并并输出漏斗日志
    combined_df = pd.concat(all_clean_records, ignore_index=True)
    log(f"🔍 [漏斗 1] 物理扫描 {total_read_rows:,} 行，其中 {passed_calmar_count:,} 行符合卡玛门槛。")

    # 6. 全局终极去重 (修改：Total_Score 优先，CAGR 其次保证优者存活)
    pool_only_df = combined_df.sort_values(['Total_Score', 'CAGR'], ascending=[False, False]).drop_duplicates(
        subset=['Combo_Tuple'])
    in_pool_unique_count = len(pool_only_df)
    log(f"🔍 [漏斗 2] 短路剔除杂质并全局去重后，剩余 {in_pool_unique_count:,} 个纯净组合。")

    # 7. 降维打击与优先级提取
    # 分化为两个互斥子池，并在子池内部按 Total_Score 优先、CAGR 其次降序排列
    score_gt_0_df = pool_only_df[pool_only_df['Total_Score'] > 0].sort_values(['Total_Score', 'CAGR'],
                                                                              ascending=[False, False])
    score_lte_0_df = pool_only_df[pool_only_df['Total_Score'] <= 0].sort_values(['Total_Score', 'CAGR'],
                                                                                ascending=[False, False])

    total_score_gt_0_count = len(score_gt_0_df)

    # 将两个子池首尾拼接，形成完全严格按优先级排序的候补长队
    full_candidate_df = pd.concat([score_gt_0_df, score_lte_0_df], ignore_index=True)
    full_candidate_list = full_candidate_df['Combo_Tuple'].tolist()

    # 设置允许 10% 误差的动态目标区间
    target_mid = target_max_combos
    target_min = int(target_mid * 0.9)
    target_max = int(target_mid * 1.1)

    # 8. 绝对硬控二分查找限速配额
    dynamic_k, expected_count = calculate_dynamic_k_by_binary_search(
        N=target_dim,
        candidate_list=full_candidate_list,
        target_min=target_min,
        target_max=target_max,
        strict_mode=True
    )

    log(f"🧠 算力评估: 目标爬坡={target_dim}维.")
    log(f"🧠 限额保护: 二分法硬控成功截留 Top {dynamic_k:,} 精英，预期生成 {expected_count:,} 个新组合。")

    # 逻辑判断：由于队列已经排好序，直接提取头部前 dynamic_k 个
    elite_df = full_candidate_df.head(dynamic_k)

    if total_score_gt_0_count >= dynamic_k:
        selected_gt_0_count = dynamic_k
        selected_fallback_count = 0
    else:
        selected_gt_0_count = total_score_gt_0_count
        selected_fallback_count = dynamic_k - total_score_gt_0_count

    log(f"📈 [质量甄别] 当前可用池中，Total_Score > 0 的基金组合总计: {total_score_gt_0_count:,} 个。")
    log(f"⚖️ [优先级选拔] 优先录用了 {selected_gt_0_count:,} 个 Total_Score > 0 的组合。")
    if selected_fallback_count > 0:
        log(f"⚖️ [补充选拔] 额外补充了 {selected_fallback_count:,} 个仅凭 CAGR 的组合来凑数。")

    good_combos = set(elite_df['Combo_Tuple'])

    log(f"✅ [漏斗 3] 最终向高维输出纯血种子: {len(good_combos):,} 个。")
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
                    is_elite = True
                    # 手动快速剥离一个元素构建子组合，比 itertools 快 30%
                    for idx in range(N):
                        sub_combo = new_combo[:idx] + new_combo[idx + 1:]
                        if sub_combo not in good_prev_combos:
                            is_elite = False
                            break

                    # 【核心修复】：如果发现任何一个 N-1 维子集不优秀，直接拦截并丢弃该组合
                    if not is_elite:
                        continue

                total_combos += 1

                # 4. 查漏补缺：比对全局缓存，实现断点跳过
                if computed_set is not None and new_combo in computed_set:
                    global_skipped += 1
                else:
                    uncomputed_combos.append(new_combo)

    return uncomputed_combos, total_combos, global_skipped


if __name__ == '__main__':
    GLOBAL_MAX_WORKERS = 30
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
    df_filtered = filter_fund_pool(df_results, min_annual_return=0.02, min_day=FIXED_MIN_DAY)
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
            0.9, 0.75)


    # 将固定的 Base Pool 常驻内存，返回值已优化为纯 6 位代码
    master_df, base_pool_codes = _build_master_matrix(base_pool_files, GLOBAL_MAX_WORKERS)
    base_pool_codes_set = set(base_pool_codes)
    print(base_pool_codes)
    final_fund_count = len(base_pool_codes)
    print("\n" + "=" * 50)
    log(f"🎯 唯一基础基金池 (Base Pool) 确定, 最终保留: {final_fund_count} 只")
    print("=" * 50 + "\n")

    if final_fund_count < 2:
        log("基础池基金数量不足以生成 2 维组合, 流程退出。")
        exit(0)
    # 步骤 2: 维度递增循环 (N = 2 开始往上爬坡)
    N = 2
    while N <= MAX_DIMENSION:
        print(f"\n{'#' * 70}")
        log(f"【维度爬坡引擎】 🚀 正在生成和评估 {N} 维 FOF 组合")
        print(f"{'#' * 70}")
        # 2.4 设置输出目标文件(名称带上 pool 和 min_day 严格符合原规则)
        output_parquet = f'fund_data/fof_evaluation_results_{N}d_pool{final_fund_count}_min_day_{FIXED_MIN_DAY}.parquet'
        if os.path.exists(output_parquet):
            log(f"已发现现有文件 {output_parquet},将直接跳过计算并进入下一维度。")
            N += 1
            continue
        # 2.1 获取上一维度(N-1)优秀的组合种子
        good_prev_combos = set()
        if N > 2:
            # 🎯 【唯一修改处】: 传入 target_dim=N, target_max_combos=1000000 触发动态 K 算法
            good_prev_combos = get_previous_good_combos(N - 1, base_pool_codes_set, target_dim=N,
                                                        target_max_combos=50000000)

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