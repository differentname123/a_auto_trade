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
    """当前时间格式化字符串"""
    return datetime.now().strftime('%H:%M:%S')


def log(msg):
    """统一日志格式打印"""
    print(f"[{now_str()}] {msg}")


def ensure_dir(filepath):
    """确保给定文件所在目录存在"""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def extract_fund_code(filepath):
    """从文件路径提取 6 位基金代码,若无则去掉 .csv 后缀"""
    match = re.search(r'(\d{6})', os.path.basename(filepath))
    return match.group(1) if match else os.path.basename(filepath).replace('.csv', '')


def normalize_combo_tuple(combo_str, sep='_'):
    """将组合字符串规范化为已排序字符串元组(防乱序穿透 & 防 OOM)"""
    parts = [str(x).strip() for x in str(combo_str).split(sep)]
    return tuple(sorted(parts))


# ====================================================================
# 全局缓存系统
# ====================================================================
def load_or_init_computed_set(combo_size):
    """
    加载或初始化已计算组合的全局缓存集合。
    跨 min_day 全局共享,扩大基金池时可完美跳过历史组合。
    """
    cache_file = f'fund_data/computed_combos_{combo_size}d_global_cache.pkl'
    parquet_files = glob.glob(f'fund_data/fof_evaluation_results_{combo_size}d_*.parquet')

    computed_set = set()
    cache_time = 0

    # 1) 加载现有缓存
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        try:
            with open(cache_file, 'rb') as f:
                computed_set = pickle.load(f)
            log(f"📦 命中全局缓存 | 规则: {combo_size} 维 | 已加载 {len(computed_set):,} 个历史元组。")
        except Exception as e:
            log(f"⚠️ 缓存读取失败将重建: {e}")
            cache_time = 0

    # 2) 增量扫描 Parquet 文件
    log(f"🔍 扫描 Parquet 文件进行增量更新 | 规则: {combo_size} 维 | 共发现 {len(parquet_files)} 个相关文件。")
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

    # 3) 持久化更新的缓存
    if updated or (not os.path.exists(cache_file) and computed_set):
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(computed_set, f)
            log(f"✅ 缓存已持久化 | 规则: {combo_size} 维 | 当前全局总库容量: {len(computed_set):,} 个。")
        except Exception as e:
            log(f"❌ 缓存持久化失败: {e}")
    elif not parquet_files and not computed_set:
        log(f"ℹ️ 空白初始化 | 规则: {combo_size} 维 | 暂无历史计算数据。")

    return computed_set, cache_file


# ====================================================================
# Numba JIT 核心引擎
# ====================================================================
@njit(fastmath=True, cache=True)
def _fast_simulate_path(fund_rets_arr, w_init, current_reb_days, offset, n_days, n_funds):
    """JIT 编译的路径演化循环:组合净值 + 周期性仓位再平衡"""
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
# 指标计算辅助函数
# ====================================================================
def _compute_max_continuous_zeros(fund_rets):
    """计算各基金最大连续零收益天数(数据质量检测)"""
    zero_df = (fund_rets.abs() < 1e-8).astype(int)
    max_zeros = 0
    for col in zero_df.columns:
        consecutive = zero_df[col].groupby((zero_df[col] == 0).cumsum()).sum().max()
        max_zeros = max(max_zeros, consecutive)
    return max_zeros


def _compute_baseline_metrics(fund_rets_arr, n_days):
    """向量化计算零协同基线 (Avg CAGR, Avg MDD, Calmar Baseline)"""
    fund_eq = np.cumprod(1.0 + fund_rets_arr, axis=0)

    fund_cagrs = np.power(fund_eq[-1, :], TRADING_DAYS_PER_YEAR / n_days) - 1.0
    avg_cagr = float(np.mean(fund_cagrs))

    cum_max = np.clip(np.maximum.accumulate(fund_eq, axis=0), a_min=1.0, a_max=None)
    drawdowns = (fund_eq - cum_max) / cum_max
    avg_mdd = float(np.mean(np.min(drawdowns, axis=0)))

    calmar_baseline = float(avg_cagr / abs(avg_mdd)) if abs(avg_mdd) > 1e-6 else 0.0
    return avg_cagr, avg_mdd, calmar_baseline


def _compute_worst_rolling_1y(synth_eq_arr, n_days):
    """计算最差滚动 1 年收益率(纯 NumPy)"""
    if n_days < TRADING_DAYS_PER_YEAR:
        return 0.0
    # 在序列前补 1.0,防止 n_days == 252 时空切片崩溃
    eq_padded = np.concatenate(([1.0], synth_eq_arr))
    rolling_ret = (eq_padded[TRADING_DAYS_PER_YEAR:] / eq_padded[:-TRADING_DAYS_PER_YEAR]) - 1.0
    return float(np.min(rolling_ret))


def _compute_worst_rolling_1y_r2(synth_eq_arr, n_days):
    """计算最差滚动 1 年净值曲线 R² (平滑度判定)"""
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
    """计算最差 5% 交易日的下行最大相关性"""
    n_worst = max(5, int(len(synth_ret) * 0.05))
    worst_dates = synth_ret.nsmallest(n_worst).index
    worst_fund_rets = fund_rets.loc[worst_dates]

    if len(worst_fund_rets) <= 3:
        return 0.5

    corr_mat = worst_fund_rets.corr().values
    triu_idx = np.triu_indices_from(corr_mat, k=1)
    if len(triu_idx[0]) == 0:
        return 0.5

    with np.errstate(invalid='ignore'):
        max_corr = np.nanmax(corr_mat[triu_idx])
    return float(max_corr) if not np.isnan(max_corr) else 0.5


def _check_vetoes(metrics, vol_annual, max_missing_ratio, max_continuous_zeros,
                  max_mdd_limit, hurdle_rate):
    """整合所有 VETO 否决条件 (Iron VETO)"""
    return {
        "VETO_Hurdle_Rate": metrics['CAGR'] < hurdle_rate,
        "VETO_Drawdown_Crash": abs(metrics['Max_Drawdown']) > max_mdd_limit,
        "VETO_Fake_Smooth": (metrics['AR1_Coefficient'] > VETO_AR1) and (vol_annual < VETO_VOL),
        "VETO_Endless_Bleeding": metrics['Max_Recovery_Days'] > VETO_RECOVERY_DAYS,
        "VETO_Data_Distortion": (max_missing_ratio > VETO_MISSING_RATIO) or (max_continuous_zeros > VETO_CONTINUOUS_ZEROS),
        "VETO_Below_Calmar_Baseline": metrics['Calmar_Ratio'] <= (metrics['Calmar_Baseline'] * VETO_CALMAR_MULTIPLIER),
        "VETO_Worst_1Y_Crash": metrics['Worst_Rolling_1Y_Return'] < VETO_WORST_1Y,
    }


def _compute_total_score(metrics, n_days):
    """第一性原理综合打分 = 基础卡玛 × 平滑度 × 尾部风险 × 时间饱和度 × 下行相关性"""
    # 1) 风险调整后收益(基础卡玛)
    excess_cagr = max(0.0, metrics['CAGR'])
    adj_mdd = max(abs(metrics['Max_Drawdown']), 0.01)
    base_calmar = excess_cagr / adj_mdd

    # 2) 净值平滑度
    smoothness = metrics['Worst_Rolling_1Y_R2']

    # 3) 尾部灾难指数级惩罚
    worst_1y = min(0.0, metrics['Worst_Rolling_1Y_Return'])
    tail_discount = float(np.exp(worst_1y * TAIL_RISK_SENSITIVITY))

    # 4) 时间饱和度
    time_mult = min(1.0, np.sqrt(n_days / TIME_SATURATION_DAYS))

    # 5) 下行相关性惩罚因子
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
# 核心评估引擎
# ====================================================================
def evaluate_fof_portfolio_fast(merged_nav, rebalance_days=DEFAULT_REBALANCE_DAYS,
                                max_history_days=DEFAULT_MAX_HISTORY,
                                max_mdd_limit=VETO_MAX_MDD, hurdle_rate=VETO_HURDLE_RATE):
    """第一性原理 FOF 绝对收益评估引擎 (极速切片版)"""
    n_funds = merged_nav.shape[1]
    raw_index = merged_nav.index

    # 1) 找到全部基金重叠的有效区间
    valid_mask = merged_nav.notna().all(axis=1)
    if not valid_mask.any():
        return {"error": "No overlapping data", "Total_Score": 0.0}

    merged_nav = merged_nav.loc[valid_mask.idxmax():valid_mask[::-1].idxmax()]

    if len(merged_nav) > max_history_days:
        merged_nav = merged_nav.iloc[-max_history_days:]

    if len(merged_nav) < TRADING_DAYS_PER_YEAR:
        return {"error": "Common data length < 252 days", "Total_Score": 0.0}

    # 2) 数据质量与僵尸基金检测
    max_missing_ratio = float((merged_nav.isna().sum() / len(merged_nav)).max())

    if (raw_index[-1] - merged_nav.index[-1]).days > VETO_ZOMBIE_DAYS:
        return {
            "error": f"Zombie fund detected. Cutoff at {merged_nav.index[-1].strftime('%Y-%m-%d')}",
            "Total_Score": 0.0
        }

    # 3) 计算各基金日收益率
    fund_rets = merged_nav.ffill().pct_change().fillna(0.0)
    max_continuous_zeros = _compute_max_continuous_zeros(fund_rets)
    n_days = len(fund_rets)

    # 强制 C-连续内存(防 Numba 崩溃 / 掉速)
    fund_rets_arr = np.ascontiguousarray(fund_rets.values, dtype=np.float64)

    # 4) 零协同基线指标(向量化)
    avg_cagr, avg_mdd, calmar_baseline = _compute_baseline_metrics(fund_rets_arr, n_days)

    # 5) 单路径评估闭包
    def _evaluate_single_path(reb_days, offset=0, w_init=None):
        w_init_arr = (np.ones(n_funds, dtype=np.float64) / n_funds
                      if w_init is None
                      else np.array(w_init, dtype=np.float64))

        synth_ret_arr = _fast_simulate_path(fund_rets_arr, w_init_arr,
                                            reb_days, offset, n_days, n_funds)
        synth_ret = pd.Series(synth_ret_arr, index=fund_rets.index)
        synth_eq = (1 + synth_ret).cumprod()

        # 基础指标计算
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

        # 组装指标
        metrics = {
            'Start_Date': merged_nav.index[0].strftime('%Y-%m-%d'),
            'End_Date': merged_nav.index[-1].strftime('%Y-%m-%d'),
            'Total_Days': n_days,
            'n_funds': n_funds,
            'Avg_CAGR': avg_cagr,
            'Avg_Max_Drawdown': avg_mdd,
            'Calmar_Baseline': calmar_baseline,
            'CAGR': cagr,
            'Max_Drawdown': max_dd,
            'Max_Recovery_Days': max_recovery_days,
            'Worst_Rolling_1Y_Return': _compute_worst_rolling_1y(synth_eq.values, n_days),
            'Worst_Rolling_1Y_R2': _compute_worst_rolling_1y_r2(synth_eq.values, n_days),
            'AR1_Coefficient': ar1,
            'Annualized_Volatility': float(vol_annual),
            'Sharpe_Ratio': sharpe,
            'Calmar_Ratio': calmar,
            'Daily_Win_Rate': float((synth_ret > 0).sum() / n_days) if n_days > 0 else 0.0,
            'Downside_Correlation': _compute_downside_correlation(synth_ret, fund_rets),
        }

        # VETO 否决判定
        vetoes = _check_vetoes(metrics, vol_annual, max_missing_ratio, max_continuous_zeros,
                               max_mdd_limit, hurdle_rate)
        metrics.update(vetoes)
        if any(vetoes.values()):
            return metrics, 0.0

        return metrics, _compute_total_score(metrics, n_days)

    # 6) 基线路径评估
    final_metrics, score_base = _evaluate_single_path(rebalance_days, offset=0)
    if score_base == 0.0:
        final_metrics['Total_Score'] = 0.0
        final_metrics['VETO_Perturbation_Death'] = False
        return final_metrics

    # 7) 多维扰动验证(取最差路径作为最终分数)
    half_offset = rebalance_days // 2
    score_weight = float('inf')
    metrics_w = None

    for seed in PERTURBATION_SEEDS:
        np.random.seed(seed)
        shift = np.random.uniform(-0.05, 0.05, n_funds)
        w_perturbed = np.clip(np.ones(n_funds) / n_funds + shift, 0.01, 1.0)
        w_perturbed = w_perturbed / np.sum(w_perturbed)

        m_w, s_w = _evaluate_single_path(rebalance_days, offset=half_offset, w_init=w_perturbed)
        if s_w < score_weight:
            score_weight, metrics_w = s_w, m_w

    if score_weight == float('inf'):
        score_weight = 0.0

    final_metrics['Total_Score'] = min(score_base, score_weight)
    final_metrics['VETO_Perturbation_Death'] = (final_metrics['Total_Score'] == 0.0)

    # 记录扰动路径下的 VETO 标志
    if final_metrics['VETO_Perturbation_Death'] and metrics_w is not None and score_weight == 0.0:
        for k, v in metrics_w.items():
            if k.startswith("VETO_") and v is True:
                final_metrics[k + "_in_Perturb"] = True

    return final_metrics


# ====================================================================
# Worker 工作进程函数
# ====================================================================
def _worker_process_combo(combo_files):
    """单组合处理:从 Master DataFrame 切片并评估"""
    global WORKER_MASTER_DF
    combo_name = "_".join(extract_fund_code(f) for f in combo_files)

    try:
        merged_nav = WORKER_MASTER_DF[list(combo_files)]
        result = evaluate_fof_portfolio_fast(merged_nav)
        result['组合文件名'] = combo_name
        return result
    except Exception as e:
        return {
            '组合文件名': combo_name,
            'error': f"处理异常: {str(e)}",
            'Total_Score': 0.0
        }


def _worker_process_chunk(combo_chunk):
    """批量处理:消灭 IPC 进程间通信开销"""
    return [_worker_process_combo(combo) for combo in combo_chunk]


# ====================================================================
# 基金净值并行加载
# ====================================================================
def _load_single_nav(filepath):
    """加载并清洗单只基金净值数据"""
    try:
        if not os.path.exists(filepath):
            return None
        df = pd.read_csv(filepath)
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        df = df.set_index('净值日期').sort_index()
        df = df[~df.index.duplicated(keep='last')]
        return df['复权净值'].rename(filepath)
    except Exception:
        return None


def _parallel_load_navs(files, max_workers, desc):
    """并行加载多只基金净值"""
    series_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_single_nav, f) for f in files]
        for future in tqdm(as_completed(futures), total=len(files), desc=desc):
            res = future.result()
            if res is not None:
                series_list.append(res)
    return series_list


# ====================================================================
# 相关性双矩阵预计算
# ====================================================================
def _build_downside_corr_matrix(downside_csv):
    """从 2 维 FOF 历史结果中提取下行相关性矩阵"""
    try:
        df_2d = pd.read_csv(downside_csv)
    except Exception as e:
        log(f"提取 Downside_Correlation 矩阵失败: {e}")
        return pd.DataFrame()

    if '组合文件名' not in df_2d.columns or 'Downside_Correlation' not in df_2d.columns:
        return pd.DataFrame()

    records = []
    for _, row in df_2d.iterrows():
        combo = row['组合文件名']
        if pd.isna(combo):
            continue
        parts = [p.strip() for p in str(combo).split('+')]
        if len(parts) != 2:
            continue
        f1, f2 = f'fund_data/nav/{parts[0]}', f'fund_data/nav/{parts[1]}'
        dc = row['Downside_Correlation']
        records.append({'f1': f1, 'f2': f2, 'dc': dc})
        records.append({'f1': f2, 'f2': f1, 'dc': dc})

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).groupby(['f1', 'f2'])['dc'].mean().unstack()


def precompute_correlations(result_csv='fund_data/all_funds_result.csv',
                            corr_csv='fund_data/fund_correlations.csv',
                            downside_csv='fund_data/fof_evaluation_results_2d_pool3917.csv',
                            max_workers=10,
                            downside_corr_csv='fund_data/fund_downside_correlations.csv'):
    """预计算全市场相关性双矩阵 (全天候 + 下行)"""
    print(f"\n[{now_str()}] ---------------- 开始准备全市场相关性双矩阵 ----------------")

    # 1) 下行相关性矩阵
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

    # 2) 全天候相关性矩阵
    if os.path.exists(corr_csv):
        log(f"已发现现有全天候相关性文件 {corr_csv},直接加载缓存...")
        return pd.read_csv(corr_csv, index_col=0), downside_corr_matrix

    if not os.path.exists(result_csv):
        log(f"错误: 未找到汇总文件 {result_csv}")
        return pd.DataFrame(), downside_corr_matrix

    df_results = pd.read_csv(result_csv)
    files = df_results['adj_nav_file'].dropna().tolist()

    log(f"正在并行读取 {len(files)} 只基金...")
    nav_series_list = _parallel_load_navs(files, max_workers, "并行加载数据")
    if not nav_series_list:
        return pd.DataFrame(), downside_corr_matrix

    merged_nav = pd.concat(nav_series_list, axis=1, join='outer').ffill()
    corr_matrix = merged_nav.pct_change().corr()

    ensure_dir(corr_csv)
    corr_matrix.to_csv(corr_csv)
    return corr_matrix, downside_corr_matrix


# ====================================================================
# 基金池漏斗过滤
# ====================================================================
def _extract_target_codes(df_filtered):
    """从过滤后 DataFrame 中按优先级提取目标基金 6 位代码"""
    if '基金代码' in df_filtered.columns:
        return df_filtered['基金代码'].astype(str).str.strip().str.zfill(6)
    if 'fund_code' in df_filtered.columns:
        return df_filtered['fund_code'].astype(str).str.strip().str.zfill(6)
    return df_filtered['adj_nav_file'].apply(
        lambda x: re.search(r'(\d{6})', str(x)).group(1)
        if pd.notna(x) and re.search(r'(\d{6})', str(x)) else "000000"
    )


def filter_fund_pool(df_results, active_cache='temp/active_fund_codes.csv',
                     min_annual_return=0.15, min_day=1000):
    """基金池漏斗过滤"""
    if df_results is None or df_results.empty:
        return pd.DataFrame()

    original_count = len(df_results)
    df_filtered = df_results.copy()
    active_filtered_count = original_count

    # 1) 可申购状态过滤
    if os.path.exists(active_cache):
        try:
            df_active = pd.read_csv(active_cache, dtype=str)
            if '基金代码' in df_active.columns:
                active_codes = set(df_active['基金代码'].str.strip().str.zfill(6).tolist())
                target_codes = _extract_target_codes(df_filtered)
                df_filtered = df_filtered[target_codes.isin(active_codes)].copy()
                active_filtered_count = len(df_filtered)
        except Exception:
            pass

    if df_filtered.empty:
        return df_filtered

    # 2) 财务与质量过滤
    condition = (
        (df_filtered['total_active_days'] > min_day) &
        (df_filtered['annualized_return'] > min_annual_return) &
        (df_filtered['missing_ratio'] <= 0.05) &
        (df_filtered['max_zeros'] < 10) &
        (df_filtered['max_drawdown'] > -0.7)
    )
    df_final = df_filtered[condition].sort_values(by='annualized_return', ascending=False)

    print("\n" + "=" * 55)
    print("🎯 基金池初筛漏斗统计:")
    print(f"  1. 初始输入总数    : {original_count} 只")
    print(f"  2. 可申购状态通过  : {active_filtered_count} 只")
    print(f"  3. 财务与质量达标  : {len(df_final)} 只 (作为高优候选池)")
    print("=" * 55 + "\n")
    return df_final


# ====================================================================
# 优胜劣汰相关性过滤
# ====================================================================
def _is_too_correlated(fund_f, selected_files, corr_matrix,
                       downside_corr_matrix, corr_threshold, downside_corr_threshold):
    """检查 fund_f 与已选基金是否过于相关(全天候 + 下行)"""
    has_downside = downside_corr_matrix is not None and not downside_corr_matrix.empty

    for sel_f in selected_files:
        # 全天候相关性
        if sel_f in corr_matrix.columns:
            corr_val = corr_matrix.loc[fund_f, sel_f]
            if pd.notna(corr_val) and corr_val > corr_threshold:
                return True

        # 下行相关性
        if has_downside and fund_f in downside_corr_matrix.index and sel_f in downside_corr_matrix.columns:
            dc_val = downside_corr_matrix.loc[fund_f, sel_f]
            if pd.notna(dc_val) and dc_val > downside_corr_threshold:
                return True

    return False


def _greedy_correlation_filter(df_filtered, corr_matrix, downside_corr_matrix,
                               corr_threshold, downside_corr_threshold):
    """贪心过滤:按收益率排序优先保留,剔除高相关性基金"""
    selected = []
    for f in df_filtered['adj_nav_file'].dropna():
        if f not in corr_matrix.columns:
            continue
        if not _is_too_correlated(f, selected, corr_matrix, downside_corr_matrix,
                                  corr_threshold, downside_corr_threshold):
            selected.append(f)
    return selected


# ====================================================================
# Parquet 结果序列化
# ====================================================================
def _prepare_results_table(batch_results):
    """将结果列表转换为压缩后的 PyArrow Table"""
    df = pd.DataFrame(batch_results).reindex(columns=OUTPUT_COLUMNS)
    df['组合文件名'] = df['组合文件名'].astype(str)
    df['error'] = df['error'].fillna("").astype(str)

    # 浮点压缩
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # 整数压缩
    for int_col in ['Total_Days', 'Max_Recovery_Days']:
        if int_col in df.columns:
            df[int_col] = df[int_col].fillna(0).astype(np.int16)

    return pa.Table.from_pandas(df)


# ====================================================================
# 主流程辅助函数
# ====================================================================
def _warmup_numba():
    """预热 Numba 编译器,防止多进程缓存踩踏"""
    log("正在预热 Numba 核心引擎,防止多进程缓存踩踏...")
    _fast_simulate_path(np.zeros((100, 2), dtype=np.float64),
                        np.array([0.5, 0.5], dtype=np.float64),
                        30, 0, 100, 2)


def _build_master_matrix(target_files, max_workers):
    """构建全局 Master Matrix 并按基金代码对齐排序"""
    log("🚀 正在构建全局 Master Matrix...")
    nav_series_list = _parallel_load_navs(target_files, max_workers, "构建 Master 列数据")
    master_df = pd.concat(nav_series_list, axis=1, join='outer')
    del nav_series_list

    sorted_files = sorted(master_df.columns.tolist(), key=extract_fund_code)
    master_df = master_df[sorted_files]
    log(f"✅ 全局 Master Matrix 构建完毕,行数: {len(master_df)},列数: {len(sorted_files)}")
    return master_df, sorted_files


def _iter_uncomputed_batch(combos_iter, computed_set, file_to_code_map, batch_size):
    """从组合生成器中提取一个 batch 的未计算组合,同时统计跳过数量"""
    batch_combos = []
    batch_skipped = 0
    try:
        while len(batch_combos) < batch_size:
            combo = next(combos_iter)
            combo_tuple = tuple(file_to_code_map[f] for f in combo)
            if computed_set is not None and combo_tuple in computed_set:
                batch_skipped += 1
            else:
                batch_combos.append(combo)
    except StopIteration:
        pass
    return batch_combos, batch_skipped


def _print_final_report(final_fund_count, combo_size, total_combos,
                        skipped, computed, output_parquet, has_output):
    """打印最终复盘报告"""
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
# 主流程
# ====================================================================
def run_batch_evaluation(result_csv='fund_data/all_funds_result.csv', combo_size=4,
                         max_workers=10, corr_threshold=0.9, corr_matrix=None,
                         downside_corr_matrix=None, downside_corr_threshold=0.0,
                         min_day=1000, computed_set=None):
    """批量评估 FOF 组合主流程"""
    _warmup_numba()

    # 步骤 1: 基础绩效筛选
    log("步骤 1: 扫描汇总文件进行基础绩效条件筛选...")
    if not os.path.exists(result_csv):
        return None
    df_results = pd.read_csv(result_csv)
    df_filtered = filter_fund_pool(df_results, min_annual_return=0.05, min_day=min_day)
    if df_filtered.empty:
        log("符合要求的基金数量为 0,退出流程。")
        return None

    # 步骤 2: 准备相关性矩阵
    if corr_matrix is None:
        corr_matrix, ds_matrix = precompute_correlations(result_csv, max_workers=max_workers)
        if downside_corr_matrix is None:
            downside_corr_matrix = ds_matrix

    # 步骤 3: 优胜劣汰过滤
    if corr_matrix.empty:
        target_files = df_filtered['adj_nav_file'].dropna().tolist()
    else:
        log("步骤 3: 启动优胜劣汰过滤...")
        target_files = _greedy_correlation_filter(
            df_filtered, corr_matrix, downside_corr_matrix,
            corr_threshold, downside_corr_threshold)

    final_fund_count = len(target_files)
    print("\n" + "=" * 50)
    log(f"🎯 基金池双矩阵排雷过滤后保留: {final_fund_count} 只")
    print("=" * 50 + "\n")

    if final_fund_count < combo_size:
        log("过滤后基金不足以生成组合。")
        return None

    # 步骤 4: 检查输出文件
    output_parquet = f'fund_data/fof_evaluation_results_{combo_size}d_pool{final_fund_count}_min_day_{min_day}.parquet'
    if os.path.exists(output_parquet):
        log(f"⏭️ 目标文件 [{output_parquet}] 已存在,直接跳过计算。")
        return output_parquet

    # 步骤 5: 构建 Master Matrix
    master_df, target_files = _build_master_matrix(target_files, max_workers)
    final_fund_count = len(target_files)
    file_to_code_map = {f: extract_fund_code(f) for f in target_files}

    # 步骤 6: 分批并行评估
    total_combos = math.comb(final_fund_count, combo_size)
    combos = itertools.combinations(target_files, combo_size)
    ensure_dir(output_parquet)

    writer = None
    global_skipped = 0
    global_computed = 0

    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=_init_worker,
                             initargs=(master_df,)) as executor:
        with tqdm(total=total_combos, desc=f"评估 {combo_size} 维组合", unit="组") as pbar:
            while True:
                batch_combos, batch_skipped = _iter_uncomputed_batch(
                    combos, computed_set, file_to_code_map, BATCH_SIZE)

                if batch_skipped > 0:
                    pbar.update(batch_skipped)
                    global_skipped += batch_skipped

                if not batch_combos:
                    break

                # 分块提交,降低 IPC 开销
                chunks = [batch_combos[i:i + CHUNK_SIZE]
                          for i in range(0, len(batch_combos), CHUNK_SIZE)]
                futures = [executor.submit(_worker_process_chunk, c) for c in chunks]

                batch_results = []
                for future in as_completed(futures):
                    chunk_res = future.result()
                    batch_results.extend(chunk_res)
                    pbar.update(len(chunk_res))

                if batch_results:
                    global_computed += len(batch_results)
                    table = _prepare_results_table(batch_results)
                    if writer is None:
                        writer = pq.ParquetWriter(output_parquet, table.schema, compression='zstd')
                    writer.write_table(table)

                del futures, batch_results, batch_combos, chunks

    # 关闭写入并报告
    has_output = writer is not None
    if has_output:
        writer.close()

    _print_final_report(final_fund_count, combo_size, total_combos,
                        global_skipped, global_computed, output_parquet, has_output)

    return output_parquet if has_output else None


# ====================================================================
# 缓存自更新
# ====================================================================
def _update_cache_from_new_file(output_file, cache_file, computed_set):
    """根据新生成的结果文件,动态追加更新组合缓存"""
    if not output_file or not os.path.exists(output_file):
        return

    cache_time = os.path.getmtime(cache_file) if os.path.exists(cache_file) else 0
    if os.path.getmtime(output_file) <= cache_time:
        return

    log("🌟 嗅探到新生成的结果文件更新于缓存,正在动态追加至持久化系统...")
    try:
        if os.path.getsize(output_file) <= 100:
            return
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
        else:
            log("⚡ 本次文件中所有组合均已在库中,无需更新。")
    except Exception as e:
        log(f"❌ 动态更新缓存失败: {e}")


# ====================================================================
# 程序入口
# ====================================================================
if __name__ == '__main__':
    GLOBAL_MAX_WORKERS = 25
    COMBO_SIZE = 3

    RESULT_CSV = 'fund_data/all_funds_result.csv'
    CORR_CSV = 'fund_data/fund_correlations.csv'
    DOWNSIDE_CSV = 'fund_data/fof_evaluation_results_2d_pool3917.csv'

    # 加载历史缓存
    computed_set, cache_file = load_or_init_computed_set(COMBO_SIZE)

    # 预计算全局相关性矩阵
    log("【全局初始化】正在准备全市场相关性双重矩阵...")
    global_corr_matrix, global_downside_corr_matrix = precompute_correlations(
        result_csv=RESULT_CSV, corr_csv=CORR_CSV, downside_csv=DOWNSIDE_CSV,
        max_workers=GLOBAL_MAX_WORKERS
    )

    # 主循环:递减 min_day 逐步扩大基金池
    for i in range(10):
        min_day = 1300 - i * 100
        print(f"\n{'#' * 70}")
        log(f"【启动引擎】正在评估 {COMBO_SIZE} 维 FOF 组合")
        print(f"{'#' * 70}")

        output_file = run_batch_evaluation(
            result_csv=RESULT_CSV,
            combo_size=COMBO_SIZE,
            max_workers=GLOBAL_MAX_WORKERS,
            corr_matrix=global_corr_matrix,
            downside_corr_matrix=global_downside_corr_matrix,
            min_day=min_day,
            computed_set=computed_set
        )

        _update_cache_from_new_file(output_file, cache_file, computed_set)