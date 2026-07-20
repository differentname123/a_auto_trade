# =====================================================================================
# 【功能摘要】
#   全市场基金组合(FOF)多维度择优引擎。从单只基金净值出发,逐维(2维 → N维)裂变生成组合,
#   经风险/收益指标评估与多重 VETO 否决后,沉淀出稳健的高分基金组合。
#
# 【输入数据】
#   1. fund_data/all_funds_result.csv : 全市场基金汇总表(含 adj_nav_file 路径/年化/天数/回撤等)。
#   2. fund_data/nav/*.csv            : 单只基金净值明细(列: 净值日期 + 复权净值)。
#   3. fund_data/*.parquet (历史)     : 各维度既往评估结果,用作断点缓存与升维种子。
#
# 【数据流转/交互】
#   汇总表 ─filter_fund_pool→ 财务达标池
#         ─precompute_correlations→ 全天候相关性矩阵
#         ─_greedy_correlation_filter→ 低相关 Base Pool
#         ─_build_master_matrix→ 常驻内存 Master 净值矩阵(整型列名,零字符串开销)
#   随后进入【维度爬坡循环 N = 2..MAX】:
#     上一维优秀种子(get_previous_good_combos) ─Apriori 前缀裂变→ 新维理论组合
#         ─比对 computed_set 缓存去重→ 待评估组合
#         ─ProcessPoolExecutor 并行→ evaluate_fof_portfolio_fast (Numba 模拟 + 分层VETO + 打分)
#
# 【输出数据】
#   1. fund_data/fof_evaluation_results_{N}d_pool*_min_day_*.parquet : 每维评估明细结果(副作用:落盘)。
#   2. fund_data/computed_combos_{N}d_global_cache.parquet           : 全局已算组合缓存(断点续算)。
#   3. fund_data/base_pool_rejection_reasons.csv                     : 全链路筛选审计记录(可追溯)。
# =====================================================================================

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

# 多进程与批处理
BATCH_SIZE = 200000
CHUNK_SIZE = 2000
PERTURBATION_SEEDS = [1024, 2048, 4096]

# 单只基金净值加载的脏数据拦截阈值
JUMP_HARD_LIMIT = 0.315   # 绝对物理涨幅上限(覆盖各板块 30% + 缓冲)
JUMP_SOFT_LIMIT = 0.06    # 固收类可疑涨幅阈值
FIXED_INCOME_VOL = 0.006  # 固收类日波动率判定线

# 输出列定义
OUTPUT_COLUMNS = [
    '组合文件名', 'Start_Date', 'End_Date', 'Total_Days',
    'CAGR', 'Max_Drawdown', 'Max_Recovery_Days', 'Worst_Rolling_1Y_R2',
    'AR1_Coefficient', 'Annualized_Volatility', 'Sharpe_Ratio',
    'Calmar_Ratio', 'Daily_Win_Rate', 'Downside_Correlation',
    'Avg_CAGR', 'Avg_Max_Drawdown', 'Calmar_Baseline', 'Worst_Rolling_1Y_Return',
    'error', 'Total_Score'
]

# 子进程全局容器(由 initializer 锚定,避免主进程大 DataFrame 反复序列化)
WORKER_MASTER_DF = None


def _init_worker(master_df):
    """子进程初始化: 将主进程构建好的 Master 净值矩阵锚定到本地内存,供后续组合直接切列。"""
    global WORKER_MASTER_DF
    WORKER_MASTER_DF = master_df


# ====================================================================
# 通用工具 & 人性化日志
# ====================================================================
def now_str():
    return datetime.now().strftime('%H:%M:%S')


def log(msg, level=None):
    """统一日志出口。level 传入 'WARN'/'ERR' 时会打上语义前缀,便于排查时一眼分级。"""
    tag = f"[{level}] " if level else ""
    print(f"[{now_str()}] {tag}{msg}")


def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def extract_fund_code(filepath):
    """从文件路径中提取 6 位基金代码,失败则退化为去后缀的文件名。"""
    match = re.search(r'(\d{6})', os.path.basename(filepath))
    return match.group(1) if match else os.path.basename(filepath).replace('.csv', '')


def normalize_combo_tuple(combo_str, sep='_'):
    """将组合字符串归一化为"升序整型元组"。整型化可极大降低缓存 Set 的内存占用。"""
    parts = [int(str(x).strip()) for x in str(combo_str).split(sep)]
    return tuple(sorted(parts))


# ====================================================================
# 全局缓存系统 (Parquet + 纯整型引擎, 支持断点续算与旧版 PKL 静默升级)
# ====================================================================
def _save_set_to_parquet(combo_set, dim, filepath):
    """将组合 Set 按维度列式拆解后以 zstd 压缩落盘。列式剥离避开 * 解包,兼顾内存与速度。"""
    try:
        if not combo_set:
            return
        combo_list = list(combo_set)
        cols = [[combo[i] for combo in combo_list] for i in range(dim)]
        arrays = [pa.array(col, type=pa.int32()) for col in cols]
        names = [f'dim_{i}' for i in range(dim)]
        table = pa.Table.from_arrays(arrays, names=names)
        pq.write_table(table, filepath, compression='zstd')
        del combo_list, cols, arrays, table  # 显式释放大内存
    except Exception as e:
        log(f"[缓存/落盘] 持久化 Parquet 失败 | 文件:[{os.path.basename(filepath)}] "
            f"| 原因:[{e}] (可能磁盘空间不足或路径无写权限)", level='ERR')


def load_or_init_computed_set(curr_dim):
    """
    加载/初始化当前维度的全局已算组合缓存。
    优先读新版 Parquet;若只存在旧版 PKL 则静默降维升级为 Parquet;最后增量扫描结果文件补全。
    """
    cache_file = f'fund_data/computed_combos_{curr_dim}d_global_cache.parquet'
    old_pkl_file = f'fund_data/computed_combos_{curr_dim}d_global_cache.pkl'
    parquet_files = glob.glob(f'fund_data/fof_evaluation_results_{curr_dim}d_*.parquet')

    computed_set = set()
    cache_time = 0

    # 1) 优先命中新版 Parquet 缓存
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        try:
            table = pq.read_table(cache_file)
            cols = [table.column(f'dim_{i}').to_pylist() for i in range(curr_dim)]
            computed_set = set(zip(*cols))
            log(f"[缓存/命中] 加载全局 Parquet 缓存成功 | 维度:【{curr_dim}D】 "
                f"| 历史组合:[{len(computed_set):,}] 个")
        except Exception as e:
            log(f"[缓存/命中] Parquet 缓存读取失败将重建 | 维度:【{curr_dim}D】 "
                f"| 原因:[{e}] (缓存文件可能损坏或版本不兼容)", level='WARN')
            cache_time = 0

    # 2) 无新版缓存时, 嗅探并静默升级历史 PKL
    elif os.path.exists(old_pkl_file):
        try:
            with open(old_pkl_file, 'rb') as f:
                old_set = pickle.load(f)
            log(f"[缓存/升级] 发现历史 PKL, 正在执行整型降维并转换为 Parquet | 维度:【{curr_dim}D】")
            for combo in old_set:
                computed_set.add(tuple(int(x) for x in combo))
            _save_set_to_parquet(computed_set, curr_dim, cache_file)
            os.rename(old_pkl_file, old_pkl_file + ".bak")  # 封存历史 PKL
            cache_time = os.path.getmtime(cache_file)
            log(f"[缓存/升级] PKL → Parquet 升级完成 | 维度:【{curr_dim}D】 "
                f"| 容量:[{len(computed_set):,}]")
        except Exception as e:
            log(f"[缓存/升级] 历史 PKL 升级失败 | 维度:【{curr_dim}D】 "
                f"| 原因:[{e}] (旧缓存格式异常, 本次将忽略并重新累积)", level='WARN')

    # 3) 增量扫描当前维度的结果文件, 补齐缓存尚未收录的组合
    log(f"[缓存/扫描] 增量扫描结果文件 | 维度:【{curr_dim}D】 | 相关文件:[{len(parquet_files)}] 个")
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
                log(f"[缓存/增量] 读取历史结果 | 文件:[{os.path.basename(pf)}] "
                    f"| 新增组合:[{added:,}]")
        except Exception as e:
            log(f"[缓存/增量] 读取结果文件异常跳过 | 文件:[{os.path.basename(pf)}] "
                f"| 原因:[{e}] (文件可能正在写入或已损坏)", level='ERR')

    if updated or (not os.path.exists(cache_file) and computed_set):
        _save_set_to_parquet(computed_set, curr_dim, cache_file)
        log(f"[缓存/落盘] 缓存已持久化 | 维度:【{curr_dim}D】 | 全局总库容:[{len(computed_set):,}]")
    elif not parquet_files and not computed_set:
        log(f"[缓存/初始] 空白初始化 | 维度:【{curr_dim}D】 | 暂无历史计算数据")

    return computed_set, cache_file


def _update_cache_from_new_file(output_file, cache_file, computed_set):
    """本轮新结果落盘后, 将其中的组合动态追加进全局缓存并回写 Parquet, 保证下次断点续算精准。"""
    if not output_file or not os.path.exists(output_file):
        return
    cache_time = os.path.getmtime(cache_file) if os.path.exists(cache_file) else 0
    if os.path.getmtime(output_file) <= cache_time:
        return

    log("[缓存/回写] 嗅探到新结果文件, 正在动态追加至全局缓存...")
    try:
        if os.path.getsize(output_file) <= 100:  # 近似空文件, 无需处理
            return

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
            log(f"[缓存/回写] 增量注入完成 | 本次新增:[{added:,}] | 全局总库容:[{len(computed_set):,}]")
    except Exception as e:
        log(f"[缓存/回写] 动态更新缓存失败 | 原因:[{e}] "
            f"(结果文件可能未完整落盘, 下轮启动会自动重扫补齐)", level='ERR')


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
    if n_days < TRADING_DAYS_PER_YEAR:
        return 0.0
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
    if len(worst_fund_rets) <= 3:
        return 0.5
    corr_mat = worst_fund_rets.corr().values
    triu_idx = np.triu_indices_from(corr_mat, k=1)
    if len(triu_idx[0]) == 0:
        return 0.5
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
#   职责: 输入合并净值 → 清洗/裁剪 → Numba 模拟组合净值 → 分层 VETO 短路 → 打分
# ====================================================================
def evaluate_fof_portfolio_fast(merged_nav, rebalance_days=DEFAULT_REBALANCE_DAYS,
                                max_history_days=DEFAULT_MAX_HISTORY,
                                max_mdd_limit=VETO_MAX_MDD, hurdle_rate=VETO_HURDLE_RATE):
    n_funds = merged_nav.shape[1]
    raw_index = merged_nav.index
    valid_mask = merged_nav.notna().all(axis=1)
    if not valid_mask.any():
        return {"error": "No overlapping data", "Total_Score": 0.0}
    merged_nav = merged_nav.loc[valid_mask.idxmax():valid_mask[::-1].idxmax()]
    if len(merged_nav) > max_history_days:
        merged_nav = merged_nav.iloc[-max_history_days:]
    if len(merged_nav) < TRADING_DAYS_PER_YEAR:
        return {"error": "Common data length < 252 days", "Total_Score": 0.0}

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

        # 第 0 层拦截: 极端数据缺失 / 长期零收益
        if (max_missing_ratio > VETO_MISSING_RATIO) or (max_continuous_zeros > VETO_CONTINUOUS_ZEROS):
            metrics["VETO_Data_Distortion"] = True
            return metrics, 0.0

        # 第 1 层短路: 基础收益 + 回撤 + 卡玛基准
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

        # 第 2 层短路: 恢复天数 + 虚假平滑
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

        # 第 3 层短路: 滚动 1 年最差表现
        worst_1y = _compute_worst_rolling_1y(synth_eq.values, n_days)
        metrics.update({'Worst_Rolling_1Y_Return': worst_1y})
        if worst_1y < VETO_WORST_1Y:
            metrics["VETO_Worst_1Y_Crash"] = True
            return metrics, 0.0

        # 第 4 层: 仅存活者才执行昂贵的平滑度/下行相关性/夏普/胜率计算
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

    # 主路径评估 + 权重扰动稳健性测试 (取最坏得分作为最终分, 防过拟合)
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
        if s_w < score_weight:
            score_weight, metrics_w = s_w, m_w

    if score_weight == float('inf'):
        score_weight = 0.0
    final_metrics['Total_Score'] = min(score_base, score_weight)
    final_metrics['VETO_Perturbation_Death'] = (final_metrics['Total_Score'] == 0.0)

    if final_metrics['VETO_Perturbation_Death'] and metrics_w is not None and score_weight == 0.0:
        for k, v in metrics_w.items():
            if k.startswith("VETO_") and v is True:
                final_metrics[k + "_in_Perturb"] = True

    return final_metrics


# ====================================================================
# Worker 工作进程函数 (直接吃整型元组, 消灭子进程内的正则解析开销)
# ====================================================================
def _worker_process_combo(combo_codes):
    """单组合评估: 传入极简整型 Tuple, 切列送入引擎; 结果落地时把代码补零还原为 6 位组合名。"""
    global WORKER_MASTER_DF
    combo_name = "_".join([f"{c:06d}" for c in combo_codes])
    try:
        merged_nav = WORKER_MASTER_DF[list(combo_codes)]  # Master 列名已是 int, 直接整型取列
        result = evaluate_fof_portfolio_fast(merged_nav)
        result['组合文件名'] = combo_name
        return result
    except Exception as e:
        return {'组合文件名': combo_name, 'error': f"处理异常: {str(e)}", 'Total_Score': 0.0}


def _worker_process_chunk(combo_chunk):
    return [_worker_process_combo(combo) for combo in combo_chunk]


# ====================================================================
# 基金净值并行加载 & 相关性矩阵 & 初筛漏斗
# ====================================================================
def _load_single_nav(filepath):
    """
    读取单只基金复权净值并做脏数据双轨拦截。
    轨道1: 绝对涨幅超物理极限直接淘汰; 轨道2: 疑似固收类却暴涨(赎回污染)则精准斩首。
    """
    try:
        if not os.path.exists(filepath):
            return None
        df = pd.read_csv(filepath)
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        df = df.set_index('净值日期').sort_index()
        df = df[~df.index.duplicated(keep='last')]

        nav_series = df['复权净值'].rename(filepath)
        recent_series = nav_series.iloc[-DEFAULT_MAX_HISTORY:] if len(nav_series) > DEFAULT_MAX_HISTORY else nav_series
        daily_returns = recent_series.ffill().pct_change().dropna()  # ffill 防止 NaN 吞噬异常涨跌

        if not daily_returns.empty:
            max_jump = float(daily_returns.max())

            # 轨道1: 绝对物理极限(覆盖各板块涨停 + 缓冲), 超限即脏数据/巨额赎回, 无脑击杀
            if max_jump > JUMP_HARD_LIMIT:
                return None

            # 轨道2: 固收类暗病拦截。剔除最大 3 天涨幅后看"平时脾气", 波动极低却单日暴涨必为赎回
            if max_jump > JUMP_SOFT_LIMIT:
                normal_returns = daily_returns.sort_values().iloc[:-3]
                normal_vol = float(normal_returns.std())
                if normal_vol < FIXED_INCOME_VOL:
                    return None

        return nav_series
    except Exception:
        return None


def _parallel_load_navs(files, max_workers, desc):
    """多进程并行加载净值列表, 过滤掉加载失败/被拦截的 None。"""
    series_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_single_nav, f) for f in files]
        for future in tqdm(as_completed(futures), total=len(files), desc=desc):
            res = future.result()
            if res is not None:
                series_list.append(res)
    return series_list


def _build_downside_corr_matrix(downside_csv):
    """从 2 维历史结果中还原下行相关性对称矩阵(供贪心去相关使用)。"""
    try:
        df_2d = pd.read_csv(downside_csv)
    except Exception as e:
        log(f"[相关性/下行] 提取 Downside_Correlation 失败 | 原因:[{e}]", level='WARN')
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


def precompute_correlations(result_csv='fund_data/all_funds_result.csv', corr_csv='fund_data/fund_correlations.csv',
                            downside_csv='fund_data/fof_evaluation_results_2d_pool3917.csv', max_workers=10,
                            downside_corr_csv='fund_data/fund_downside_correlations_300.csv', df_filtered=None):
    """
    构建全天候相关性矩阵(命中缓存则直接读取)。
    注: 当前流程下行相关性恒为空矩阵(与原逻辑保持一致), 去相关阶段将仅依赖全天候相关性。
    """
    print(f"\n[{now_str()}] ---------------- 开始准备全市场相关性矩阵 ----------------")
    log(f"[相关性/下行] 未启用 2 维下行相关性提取 | 来源:[{downside_csv}] | 结果:[跳过, 使用空矩阵]")
    downside_corr_matrix = pd.DataFrame()

    if os.path.exists(corr_csv):
        log(f"[相关性/全天候] 命中缓存直接加载 | 文件:[{os.path.basename(corr_csv)}]")
        return pd.read_csv(corr_csv, index_col=0), downside_corr_matrix

    if not os.path.exists(result_csv):
        return pd.DataFrame(), downside_corr_matrix

    df_results = pd.read_csv(result_csv)
    if df_filtered is not None and not df_filtered.empty:
        df_results = df_filtered
        log(f"[相关性/全天候] 已套用最新初筛结果 | 计算基数:[{len(df_results)}] 只")

    files = df_results['adj_nav_file'].dropna().tolist()
    log(f"[相关性/全天候] 并行读取净值中 | 待读:[{len(files)}] 只")
    nav_series_list = _parallel_load_navs(files, max_workers, "并行加载数据")
    if not nav_series_list:
        return pd.DataFrame(), downside_corr_matrix

    merged_nav = pd.concat(nav_series_list, axis=1, join='outer').ffill()
    corr_matrix = merged_nav.pct_change().corr()
    ensure_dir(corr_csv)
    corr_matrix.to_csv(corr_csv)
    log(f"[相关性/全天候] 矩阵构建完成并落盘 | 覆盖标的:[{len(corr_matrix)}] 只")
    return corr_matrix, downside_corr_matrix


def _extract_target_codes(df_filtered):
    """统一从多种可能的列(基金代码/fund_code/文件名)中提取标准 6 位代码。"""
    if '基金代码' in df_filtered.columns:
        return df_filtered['基金代码'].astype(str).str.strip().str.zfill(6)
    if 'fund_code' in df_filtered.columns:
        return df_filtered['fund_code'].astype(str).str.strip().str.zfill(6)
    return df_filtered['adj_nav_file'].apply(
        lambda x: re.search(r'(\d{6})', str(x)).group(1) if pd.notna(x) and re.search(r'(\d{6})', str(x)) else "000000")


def filter_fund_pool(df_results, active_cache='temp/active_fund_codes.csv', min_annual_return=0.15, min_day=1000):
    """
    基金池初筛: 先过可申购状态白名单, 再过财务硬性指标(天数/年化/缺失率/连续零/回撤)。
    全程记录每只基金的剔除原因, 全新覆盖写入审计文件, 保证可追溯。
    """
    if df_results is None or df_results.empty:
        return pd.DataFrame()
    original_count = len(df_results)
    df_filtered = df_results.copy()
    active_filtered_count = original_count

    log(f"[初筛/启动] 基础过滤开始 | 初始池:[{original_count}] 只 "
        f"| 门槛: 年化>[{min_annual_return * 100:.0f}%] & 天数>[{min_day}]天")

    rejection_records = []

    # 阶段一: 可申购状态白名单过滤
    if os.path.exists(active_cache):
        try:
            df_active = pd.read_csv(active_cache, dtype=str)
            if '基金代码' in df_active.columns:
                active_codes = set(df_active['基金代码'].str.strip().str.zfill(6).tolist())
                target_codes = _extract_target_codes(df_filtered)
                is_active = target_codes.isin(active_codes)

                rejected_active = df_filtered[~is_active]
                for idx, row in rejected_active.iterrows():
                    rejection_records.append({
                        '基金代码': target_codes.loc[idx],
                        '文件': row.get('adj_nav_file', ''),
                        '筛选阶段': '初筛-申购状态',
                        '剔除原因': '该基金暂不可申购或已下架, 未在可申购白名单(active_cache)内。'
                    })

                df_filtered = df_filtered[is_active].copy()
                active_filtered_count = len(df_filtered)
                log(f"[初筛/申购] 申购状态校验完成 "
                    f"| 剔除:[{original_count - active_filtered_count}] 只(不可申购/下架)")
        except Exception as e:
            log(f"[初筛/申购] 白名单文件读取失败, 本步跳过 | 原因:[{e}]", level='WARN')

    if df_filtered.empty:
        ensure_dir('fund_data/base_pool_rejection_reasons.csv')
        if rejection_records:
            pd.DataFrame(rejection_records).to_csv('fund_data/base_pool_rejection_reasons.csv', index=False,
                                                   encoding='utf-8-sig')
        return df_filtered

    # 阶段二: 财务硬性指标过滤。短路取首个不达标原因, not 天然兼容 NaN, 与原布尔逻辑严格等价
    def _get_reject_reason(row):
        if not (row['total_active_days'] > min_day):
            return f"运行天数不足, 当前{row['total_active_days']}天, 目标>{min_day}天。"
        if not (row['annualized_return'] > min_annual_return):
            return f"年化收益不达标, 当前{row['annualized_return'] * 100:.2f}%, 目标>{min_annual_return * 100:.2f}%。"
        if not (row['missing_ratio'] <= 0.05):
            return f"数据缺失率过高, 当前{row['missing_ratio'] * 100:.2f}%, 目标<=5.00%。"
        if not (row['max_zeros'] < 10):
            return f"连续零收益异常, 当前最大连续{row['max_zeros']}天, 目标<10天。"
        if not (row['max_drawdown'] > -0.7):
            return f"最大回撤击穿底线, 当前{row['max_drawdown'] * 100:.2f}%, 目标>-70.00%。"
        return ""

    df_filtered['剔除原因'] = df_filtered.apply(_get_reject_reason, axis=1)

    reject_counts = {
        "运行天数不足": df_filtered['剔除原因'].str.startswith("运行天数不足").sum(),
        "年化收益不达标": df_filtered['剔除原因'].str.startswith("年化收益不达标").sum(),
        "数据缺失率过高": df_filtered['剔除原因'].str.startswith("数据缺失率过高").sum(),
        "连续零收益异常": df_filtered['剔除原因'].str.startswith("连续零收益异常").sum(),
        "最大回撤已击穿底线": df_filtered['剔除原因'].str.startswith("最大回撤击穿底线").sum()
    }

    rejected_metrics = df_filtered[df_filtered['剔除原因'] != ""]
    target_codes_filtered = _extract_target_codes(df_filtered)
    for idx, row in rejected_metrics.iterrows():
        rejection_records.append({
            '基金代码': target_codes_filtered.loc[idx],
            '文件': row.get('adj_nav_file', ''),
            '筛选阶段': '初筛-财务硬性指标',
            '剔除原因': row['剔除原因']
        })

    # 初筛是全新流程起点, 每次全新覆盖写入, 确保审计文件干净且为最新
    reason_file = 'fund_data/base_pool_rejection_reasons.csv'
    ensure_dir(reason_file)
    if rejection_records:
        pd.DataFrame(rejection_records).to_csv(reason_file, index=False, encoding='utf-8-sig')
    else:
        pd.DataFrame(columns=['基金代码', '文件', '筛选阶段', '剔除原因']).to_csv(
            reason_file, index=False, encoding='utf-8-sig')

    df_final = df_filtered[df_filtered['剔除原因'] == ""].drop(columns=['剔除原因']).sort_values(
        by='annualized_return', ascending=False)

    print("\n" + "=" * 65)
    print("🎯 基金池初筛漏斗统计:")
    print(f"  1. 初始输入总数           : {original_count} 只")
    print(f"  2. 可申购状态通过         : {active_filtered_count} 只 (剔除 {original_count - active_filtered_count} 只)")
    print(f"  3. 财务与质量硬性达标     : {len(df_final)} 只 (保留作为高优候选池)")
    print("     [财务硬性指标驳回分布]")
    print(f"      - 运行天数不足        : {reject_counts['运行天数不足']} 只")
    print(f"      - 年化收益不达标      : {reject_counts['年化收益不达标']} 只")
    print(f"      - 数据缺失率过高      : {reject_counts['数据缺失率过高']} 只")
    print(f"      - 连续零收益异常      : {reject_counts['连续零收益异常']} 只")
    print(f"      - 最大回撤已击穿底线  : {reject_counts['最大回撤已击穿底线']} 只")
    print("=" * 65 + "\n")

    return df_final


def _is_too_correlated(fund_f, selected_files, corr_matrix, downside_corr_matrix,
                       corr_threshold, downside_corr_threshold):
    """判断候选基金与已入选者是否相关性超标, 并返回可读的超标原因。"""
    has_downside = downside_corr_matrix is not None and not downside_corr_matrix.empty
    for sel_f in selected_files:
        if sel_f in corr_matrix.columns:
            corr_val = corr_matrix.loc[fund_f, sel_f]
            if pd.notna(corr_val) and corr_val > corr_threshold:
                f2_code = extract_fund_code(sel_f)
                return True, (f"全天候相关性超标, 与已保留标的[{f2_code}]相关性达[{corr_val:.4f}], "
                              f"目标<=[{corr_threshold}]。")
        if has_downside and fund_f in downside_corr_matrix.index and sel_f in downside_corr_matrix.columns:
            dc_val = downside_corr_matrix.loc[fund_f, sel_f]
            if pd.notna(dc_val) and dc_val > downside_corr_threshold:
                f2_code = extract_fund_code(sel_f)
                return True, (f"下行相关性超标, 与已保留标的[{f2_code}]下行相关性达[{dc_val:.4f}], "
                              f"目标<=[{downside_corr_threshold}]。")
    return False, ""


def _greedy_correlation_filter(df_filtered, corr_matrix, downside_corr_matrix,
                               corr_threshold, downside_corr_threshold):
    """
    贪心去相关: 按年化降序逐只考察, 与已入选者相关性均达标才纳入 Base Pool。
    矩阵中缺数据的基金采取"从宽保留"策略, 并把全部入选/剔除动作追加进审计文件形成闭环。
    """
    selected = []
    action_records = []

    for f in df_filtered['adj_nav_file'].dropna():
        # 相关性矩阵缺该基金数据: 无法评估, 从宽保留进入下一轮
        if f not in corr_matrix.columns:
            selected.append(f)
            action_records.append({
                '基金代码': extract_fund_code(f), '文件': f, '筛选阶段': '次筛-相关性过滤',
                '剔除原因': '【成功入选】未在全局相关性矩阵中找到数据, 无法安全评估, 从宽保留进入下一轮。'
            })
            continue

        is_corr, reason = _is_too_correlated(f, selected, corr_matrix, downside_corr_matrix,
                                             corr_threshold, downside_corr_threshold)
        if not is_corr:
            selected.append(f)
            action_records.append({
                '基金代码': extract_fund_code(f), '文件': f, '筛选阶段': '最终结果',
                '剔除原因': '【成功入选】顺利通过财务初筛与相关性测试, 已进入 Base Pool。'
            })
        else:
            action_records.append({
                '基金代码': extract_fund_code(f), '文件': f, '筛选阶段': '次筛-相关性过滤',
                '剔除原因': reason
            })

    # 追加写入初筛创建的审计文件, 成功入选记录置顶, 实现全链路可追溯
    if action_records:
        reason_file = 'fund_data/base_pool_rejection_reasons.csv'
        df_action = pd.DataFrame(action_records)
        ensure_dir(reason_file)
        df_action['is_selected'] = df_action['筛选阶段'] == '最终结果'
        df_action = df_action.sort_values(by=['is_selected', '基金代码'],
                                          ascending=[False, True]).drop(columns=['is_selected'])
        if os.path.exists(reason_file):
            df_action.to_csv(reason_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_action.to_csv(reason_file, index=False, encoding='utf-8-sig')

    return selected


# ====================================================================
# 结果落盘 & 引擎预热 & Master 矩阵构建
# ====================================================================
def _prepare_results_table(batch_results):
    """将一批评估结果规整为对齐列、压缩类型的 Arrow Table, 便于低开销追加落盘。"""
    df = pd.DataFrame(batch_results).reindex(columns=OUTPUT_COLUMNS)
    df['组合文件名'] = df['组合文件名'].astype(str)
    df['error'] = df['error'].fillna("").astype(str)
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    for int_col in ['Total_Days', 'Max_Recovery_Days']:
        if int_col in df.columns:
            df[int_col] = df[int_col].fillna(0).astype(np.int16)
    return pa.Table.from_pandas(df)


def _warmup_numba():
    """提前触发 Numba 编译, 避免多进程启动瞬间集中编译造成缓存踩踏。"""
    log("[引擎/预热] 正在预热 Numba 核心引擎, 防止多进程缓存踩踏...")
    _fast_simulate_path(np.zeros((100, 2), dtype=np.float64),
                        np.array([0.5, 0.5], dtype=np.float64), 30, 0, 100, 2)


def _build_master_matrix(target_files, max_workers):
    """构建常驻内存的全局净值矩阵, 列名直接映射为整型基金代码, 剥离字符串开销以配合整型缓存。"""
    log("[Master/构建] 开始构建全局净值矩阵...")
    nav_series_list = _parallel_load_navs(target_files, max_workers, "构建 Master 列数据")
    master_df = pd.concat(nav_series_list, axis=1, join='outer')
    del nav_series_list

    master_df.columns = [int(extract_fund_code(col)) for col in master_df.columns]
    sorted_codes = sorted(master_df.columns.tolist())
    master_df = master_df[sorted_codes]

    log(f"[Master/构建] 构建完毕 | 行数:[{len(master_df)}] | 列数(基金):[{len(sorted_codes)}]")
    return master_df, sorted_codes


def _print_final_report(final_fund_count, combo_size, total_combos, skipped, computed, output_parquet, has_output):
    """打印单维度批处理任务复盘报告(理论量/缓存跳过/实算量/落盘路径)。"""
    skip_ratio = (skipped / total_combos * 100) if total_combos > 0 else 0.0
    comp_ratio = (computed / total_combos * 100) if total_combos > 0 else 0.0
    print("\n" + "★" * 60)
    log(f"📊 【批处理复盘报告】 池:[{final_fund_count}]只 | 维度:【{combo_size}D】")
    print(f"  ➤ 理论总组合数量 : {total_combos:,} 组")
    print(f"  ➤ 命中缓存跳过   : {skipped:,} 组 ({skip_ratio:.2f}%) 🚀")
    print(f"  ➤ 实际提交计算量 : {computed:,} 组 ({comp_ratio:.2f}%)")
    if has_output:
        print(f"  ➤ 最终文件保存至 : {output_parquet}")
    else:
        print("  ➤ 执行动作结语   : 完美跳过, 本次未产生任何新计算数据。")
    print("★" * 60 + "\n")


# ====================================================================
# 升维测算引擎 (纯内存快速估量 + 对数插值动态限额)
# ====================================================================
def fast_estimate_combos(N, seeds_list, strict_mode=True):
    """纯内存极速测算: 给定种子在 N 维下按 Apriori 前缀裂变的理论生成量(无任何文件 I/O)。"""
    seeds_set = set(seeds_list)
    prefix_map = defaultdict(list)
    prefix_len = N - 2

    for combo in seeds_list:
        prefix_map[combo[:prefix_len]].append(combo)

    total_count = 0
    for prefix, combos in prefix_map.items():
        n_combos = len(combos)
        if n_combos < 2:
            continue
        for i in range(n_combos):
            for j in range(i + 1, n_combos):
                tail1, tail2 = combos[i][-1], combos[j][-1]
                new_combo = prefix + (tail1, tail2) if tail1 < tail2 else prefix + (tail2, tail1)
                if strict_mode:
                    is_elite = True
                    for idx in range(N):
                        if new_combo[:idx] + new_combo[idx + 1:] not in seeds_set:
                            is_elite = False
                            break
                    if not is_elite:
                        continue
                total_count += 1
    return total_count


def calculate_dynamic_k_by_binary_search(N, candidate_list, target_min=45000000,
                                         target_max=55000000, strict_mode=True):
    """
    在候补种子队列上寻找最优截留数 K, 使升维后的理论生成量落入目标区间。
    采用"对数空间插值 + 自适应阻尼步进", 契合组合数非线性爆炸特性, 避免全量试算的算力黑洞。
    """
    if len(candidate_list) < 2:
        return len(candidate_list), 0

    high_total = len(candidate_list)
    target_mid = (target_min + target_max) // 2
    log(f"[测算/启动] 非线性测算器就绪 | 全量候补:[{high_total:,}] "
        f"| 目标区间:[{target_min:,} ~ {target_max:,}]")

    # 第一步: 初始锚点探测
    init_k = max(2, min(high_total // 2, 1000000))
    count_init = fast_estimate_combos(N, candidate_list[:init_k], strict_mode)
    log(f"[测算/锚点] 初始锚点 | K=[{init_k:,}] → 生成量:[{count_init:,}]")

    best_diff = abs(count_init - target_mid)
    best_k, best_count = init_k, count_init
    if target_min <= count_init <= target_max:
        return init_k, count_init

    # 第二步: 夹逼出安全上下界
    if count_init < target_mid:
        low, count_low = init_k, count_init
        high, count_high = high_total, None
    else:
        low = 2
        count_low = fast_estimate_combos(N, candidate_list[:low], strict_mode)
        high, count_high = init_k, count_init
        if count_low > target_max:
            log(f"[测算/下限] 物理极限 K=2 生成量([{count_low:,}])已超上限, 强制收敛。", level='WARN')
            return low, count_low
        if target_min <= count_low <= target_max:
            return low, count_low

    prev_k, prev_count = None, None
    while count_high is None and low < high_total:
        shortfall_ratio = target_mid / max(1, count_low)
        is_dynamic = False
        E = None
        if prev_k is not None and count_low > prev_count and low > prev_k:
            try:
                log_c_diff = math.log(max(1, count_low)) - math.log(max(1, prev_count))
                log_k_diff = math.log(low) - math.log(prev_k)
                E = max(1.1, min(log_c_diff / log_k_diff, float(N)))
                predicted_multiplier = math.pow(shortfall_ratio, 1 / E)
                safe_multiplier = min(3.5, max(1.15, predicted_multiplier * 1.05))  # 越界偏置, 打破逼近停滞
                is_dynamic = True
            except (ZeroDivisionError, ValueError):
                safe_multiplier = min(2.0, max(1.15, math.pow(shortfall_ratio, 1 / max(1, N)) * 1.05))
        else:
            safe_multiplier = min(2.0, max(1.15, math.pow(shortfall_ratio, 1 / max(1, N)) * 1.05))

        next_k = min(high_total, max(low + 10, int(low * safe_multiplier)))  # 至少推进 10 步防死循环
        if is_dynamic:
            log(f"[测算/步进] 自适应探索上限 | 感知维度 E=[{E:.2f}] "
                f"| 倍率:[{safe_multiplier:.2f}x] → K=[{next_k:,}]")
        else:
            log(f"[测算/步进] 数学预估探索上限 | 倍率:[{safe_multiplier:.2f}x] → K=[{next_k:,}]")

        count = fast_estimate_combos(N, candidate_list[:next_k], strict_mode)
        log(f"[测算/结果] K=[{next_k:,}] → 生成量:[{count:,}]")

        if abs(count - target_mid) < best_diff:
            best_diff, best_k, best_count = abs(count - target_mid), next_k, count
        if target_min <= count <= target_max:
            return next_k, count

        if count > target_max:
            high, count_high = next_k, count
            break  # 成功跨越, 移交第三步光速插值
        prev_k, prev_count = low, count_low
        low, count_low = next_k, count

    if count_high is None:
        log(f"[测算/枯竭] 全量候补池([{high_total:,}])生成量仍低于目标中值, 采用最佳逼近。", level='WARN')
        return best_k, best_count

    # 第三步: 在安全区间内对数插值精细查找
    log(f"[测算/锁定] 已夹逼安全区间 K∈[{low:,} ~ {high:,}], 开始对数插值精查...")
    while low <= high:
        if high - low < 1000:
            log("[测算/精度] 上下限差值极小, 停止探索。")
            break
        try:
            log_low_k, log_high_k = math.log(low), math.log(high)
            log_low_c, log_high_c = math.log(max(1, count_low)), math.log(max(1, count_high))
            ratio = (math.log(target_mid) - log_low_c) / (log_high_c - log_low_c)
            mid_guess = int(math.exp(log_low_k + ratio * (log_high_k - log_low_k)))
            mid = max(low + 1, min(high - 1, mid_guess))
        except (ValueError, ZeroDivisionError):
            mid = (low + high) // 2

        log(f"[测算/插值] 预测最佳 K=[{mid:,}] (当前区间 [{low:,} ~ {high:,}])")
        count = fast_estimate_combos(N, candidate_list[:mid], strict_mode)
        diff = abs(count - target_mid)
        if diff < best_diff:
            best_diff, best_k, best_count = diff, mid, count
        if target_min <= count <= target_max:
            log(f"[测算/命中] 精准落入目标区间 | K=[{mid:,}] → 生成量:[{count:,}]")
            return mid, count
        if count > target_max:
            high, count_high = mid, count
        else:
            low, count_low = mid, count

    log(f"[测算/结束] 采用最佳逼近 | K=[{best_k:,}] → 生成量:[{best_count:,}]")
    return best_k, best_count


def get_previous_good_combos(prev_dim, base_pool_codes_set, target_dim, target_max_combos=1000000):
    """
    从上一维(prev_dim)历史结果中提炼优质种子供升维。
    漏斗: 卡玛门槛 → Base Pool 归属过滤+乱序归一化 → 全局去重 → Total_Score/CAGR 优先级排序
         → 动态 K 限额截留, 输出可控规模的纯血种子集。
    """
    def fast_filter_and_normalize(combo_str):
        """自带短路的极速转换: 一次 split, 仅保留完全落在 Base Pool 内的组合并升序归一。"""
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
            df = pq.read_table(
                pf, columns=['组合文件名', 'Calmar_Ratio', 'Calmar_Baseline', 'Total_Score', 'CAGR']).to_pandas()
            df['Total_Score'] = df['Total_Score'].fillna(0) if 'Total_Score' in df.columns else 0
            total_read_rows += len(df)

            df = df[df['Calmar_Ratio'] > df['Calmar_Baseline']]  # 门槛1: 卡玛基准
            passed_calmar_count += len(df)
            if df.empty:
                continue

            df['Combo_Tuple'] = df['组合文件名'].apply(fast_filter_and_normalize)  # 门槛2&3: 归属+归一
            df = df[df['Combo_Tuple'].notna()]
            if df.empty:
                continue

            # 门槛4: 单文件局部去重(Total_Score 优先, CAGR 次之)
            df = df.sort_values(['Total_Score', 'CAGR'], ascending=[False, False]).drop_duplicates(subset=['Combo_Tuple'])
            all_clean_records.append(df[['Combo_Tuple', 'Calmar_Ratio', 'Total_Score', 'CAGR']])
        except Exception as e:
            log(f"[升维/读取] 上一维文件异常跳过 | 文件:[{os.path.basename(pf)}] "
                f"| 原因:[{e}]", level='WARN')

    if not all_clean_records:
        log("[升维/读取] 未读到任何符合基础条件的上一维历史数据。", level='WARN')
        return set()

    combined_df = pd.concat(all_clean_records, ignore_index=True)
    log(f"[升维/漏斗1] 物理扫描:[{total_read_rows:,}]行 | 通过卡玛门槛:[{passed_calmar_count:,}]行")

    # 全局终极去重
    pool_only_df = combined_df.sort_values(['Total_Score', 'CAGR'],
                                           ascending=[False, False]).drop_duplicates(subset=['Combo_Tuple'])
    log(f"[升维/漏斗2] 剔杂质+全局去重后剩余纯净组合:[{len(pool_only_df):,}]")

    # 按 Total_Score>0 优先、其余凑数的优先级拼接候补长队
    score_gt_0_df = pool_only_df[pool_only_df['Total_Score'] > 0].sort_values(
        ['Total_Score', 'CAGR'], ascending=[False, False])
    score_lte_0_df = pool_only_df[pool_only_df['Total_Score'] <= 0].sort_values(
        ['Total_Score', 'CAGR'], ascending=[False, False])
    total_score_gt_0_count = len(score_gt_0_df)

    full_candidate_df = pd.concat([score_gt_0_df, score_lte_0_df], ignore_index=True)
    full_candidate_list = full_candidate_df['Combo_Tuple'].tolist()

    target_mid = target_max_combos
    target_min = int(target_mid * 0.9)
    target_max = int(target_mid * 1.1)

    dynamic_k, expected_count = calculate_dynamic_k_by_binary_search(
        N=target_dim, candidate_list=full_candidate_list,
        target_min=target_min, target_max=target_max, strict_mode=True)

    log(f"[升维/限额] 目标爬坡 [{target_dim}D] | 二分法截留 Top:[{dynamic_k:,}] 精英 "
        f"| 预期生成新组合:[{expected_count:,}]")

    elite_df = full_candidate_df.head(dynamic_k)
    if total_score_gt_0_count >= dynamic_k:
        selected_gt_0_count, selected_fallback_count = dynamic_k, 0
    else:
        selected_gt_0_count = total_score_gt_0_count
        selected_fallback_count = dynamic_k - total_score_gt_0_count

    log(f"[升维/甄别] 可用池 Total_Score>0 组合:[{total_score_gt_0_count:,}] "
        f"| 优先录用:[{selected_gt_0_count:,}]")
    if selected_fallback_count > 0:
        log(f"[升维/补充] 额外补充仅凭 CAGR 的组合凑数:[{selected_fallback_count:,}]")

    good_combos = set(elite_df['Combo_Tuple'])
    log(f"[升维/漏斗3] 最终向高维输出纯血种子:[{len(good_combos):,}]")
    return good_combos


def generate_next_dimension_combos_apriori(N, good_prev_combos, computed_set, strict_mode=False):
    """
    高维组合裂变引擎(Apriori 原理): 上一维优秀种子按共同前缀分家族, 家族内两两繁衍成 N 维新组合。
    strict_mode 开启时, 要求新组合的全部 N-1 维子集都优秀才放行(剪枝提纯)。
    返回: (待计算组合列表, 理论生成总数, 命中缓存被跳过数)
    """
    uncomputed_combos = []
    global_skipped = 0
    total_combos = 0

    if strict_mode and not isinstance(good_prev_combos, set):
        good_prev_combos = set(good_prev_combos)

    prefix_map = defaultdict(list)
    prefix_len = N - 2
    for combo in good_prev_combos:
        prefix_map[combo[:prefix_len]].append(combo)

    for prefix, combos in prefix_map.items():
        n_combos = len(combos)
        if n_combos < 2:  # 家族内不足两个, 无法繁衍
            continue
        for i in range(n_combos):
            for j in range(i + 1, n_combos):
                tail1, tail2 = combos[i][-1], combos[j][-1]
                new_combo = prefix + (tail1, tail2) if tail1 < tail2 else prefix + (tail2, tail1)

                if strict_mode:
                    is_elite = True
                    for idx in range(N):  # 手动剥离比 itertools 更快
                        if new_combo[:idx] + new_combo[idx + 1:] not in good_prev_combos:
                            is_elite = False
                            break
                    if not is_elite:
                        continue

                total_combos += 1
                if computed_set is not None and new_combo in computed_set:
                    global_skipped += 1
                else:
                    uncomputed_combos.append(new_combo)

    return uncomputed_combos, total_combos, global_skipped


# ====================================================================
# 主流程: 初始化 Base Pool → 维度爬坡评估
# ====================================================================
if __name__ == '__main__':
    GLOBAL_MAX_WORKERS = 30
    RESULT_CSV = 'fund_data/all_funds_result.csv'
    CORR_CSV = 'fund_data/fund_correlations_300.csv'
    DOWNSIDE_CSV = 'fund_data/fof_evaluation_results_2d_pool3917.csv'

    FIXED_MIN_DAY = 250   # Base Pool 初筛的最短运行天数
    MAX_DIMENSION = 10    # 最高评估维度

    _warmup_numba()

    # 步骤 1: 生成唯一的 Base Pool
    log("[主流程/初始化] 扫描汇总文件, 生成唯一基础池(Base Pool)...")
    if not os.path.exists(RESULT_CSV):
        log(f"[主流程/初始化] 未找到汇总文件 | 文件:[{RESULT_CSV}] (流程终止)", level='ERR')
        exit(0)

    df_results = pd.read_csv(RESULT_CSV)
    df_filtered = filter_fund_pool(df_results, min_annual_return=0.5, min_day=FIXED_MIN_DAY)
    if df_filtered.empty:
        log("[主流程/初始化] 符合基础要求的基金数量为 0, 流程退出。", level='WARN')
        exit(0)

    matched_downside = glob.glob(DOWNSIDE_CSV)
    ACTUAL_DOWNSIDE_CSV = (matched_downside[0] if matched_downside
                           else 'fund_data/fof_evaluation_results_2d_downside_300.csv')

    global_corr_matrix, global_downside_corr_matrix = precompute_correlations(
        result_csv=RESULT_CSV, corr_csv=CORR_CSV, downside_csv=ACTUAL_DOWNSIDE_CSV,
        max_workers=GLOBAL_MAX_WORKERS, df_filtered=df_filtered)

    if global_corr_matrix.empty:
        base_pool_files = df_filtered['adj_nav_file'].dropna().tolist()
    else:
        log("[主流程/去相关] 启动相关性优胜劣汰过滤...")
        base_pool_files = _greedy_correlation_filter(
            df_filtered, global_corr_matrix, global_downside_corr_matrix, 0.95, 0.75)

    master_df, base_pool_codes = _build_master_matrix(base_pool_files, GLOBAL_MAX_WORKERS)
    base_pool_codes_set = set(base_pool_codes)
    print(base_pool_codes)
    final_fund_count = len(base_pool_codes)

    print("\n" + "=" * 50)
    log(f"[主流程/Base Pool] 基础基金池已确定 | 最终保留:[{final_fund_count}] 只")
    print("=" * 50 + "\n")

    if final_fund_count < 2:
        log("[主流程/Base Pool] 基金数量不足以生成 2 维组合, 流程退出。", level='WARN')
        exit(0)

    # 步骤 2: 维度爬坡循环 (N = 2 起逐级向上)
    N = 2
    while N <= MAX_DIMENSION:
        print(f"\n{'#' * 70}")
        log(f"[爬坡引擎] 🚀 开始生成与评估 | 维度:【{N}D】 FOF 组合")
        print(f"{'#' * 70}")

        output_parquet = (f'fund_data/fof_evaluation_results_{N}d_'
                          f'pool{final_fund_count}_min_day_{FIXED_MIN_DAY}.parquet')
        if os.path.exists(output_parquet):
            log(f"[爬坡/跳过] 已存在结果文件, 直接进入下一维度 | 文件:[{os.path.basename(output_parquet)}]")
            N += 1
            continue

        # 2.1 获取上一维优秀种子(N=2 无需种子)
        good_prev_combos = set()
        if N > 2:
            good_prev_combos = get_previous_good_combos(
                N - 1, base_pool_codes_set, target_dim=N, target_max_combos=50000000)
            log(f"[爬坡/种子] 自【{N - 1}D】提取优秀种子:[{len(good_prev_combos):,}] 个")
            if not good_prev_combos:
                log(f"[爬坡/终止] 无【{N - 1}D】优秀组合, 失去升维裂变能力, 算法自然终止。", level='WARN')
                break

        # 2.2 加载当前维度缓存(断点续算)
        computed_set, cache_file = load_or_init_computed_set(N)

        # 2.3 生成待评估组合(边裂变、边去重、边剪枝)
        log(f"[爬坡/裂变] 生成【{N}D】待测序列池(极速剪枝模式)...")
        if N == 2:
            uncomputed_combos = []
            global_skipped = 0
            total_combos = 0
            for c in itertools.combinations(base_pool_codes, 2):  # 组合默认已升序
                total_combos += 1
                if c not in computed_set:
                    uncomputed_combos.append(c)
                else:
                    global_skipped += 1
        else:
            uncomputed_combos, total_combos, global_skipped = generate_next_dimension_combos_apriori(
                N=N, good_prev_combos=good_prev_combos, computed_set=computed_set, strict_mode=True)

        # 【Bug 修复】: 先判空再打印占比, 避免 total_combos=0 时的 ZeroDivisionError
        if total_combos == 0:
            log(f"[爬坡/裂变] 生成的待评估组合数量为 0 | 维度:【{N}D】 (流程结束)", level='WARN')
            break

        skip_ratio = global_skipped / total_combos * 100
        log(f"[爬坡/裂变] 理论组合:[{total_combos:,}] | 命中缓存跳过:[{global_skipped:,}] "
            f"| 待评估:[{len(uncomputed_combos):,}] | 跳过占比:[{skip_ratio:.2f}%]")

        if not uncomputed_combos:
            log(f"[爬坡/跳过] 【{N}D】理论组合已全部命中缓存, 直接进入下一维度。")
            N += 1
            continue

        ensure_dir(output_parquet)
        writer = None
        global_computed = 0

        # 2.4 并行批处理评估并流式落盘
        with ProcessPoolExecutor(max_workers=GLOBAL_MAX_WORKERS,
                                 initializer=_init_worker,
                                 initargs=(master_df,)) as executor:
            with tqdm(total=total_combos, desc=f"计算 {N} 维", unit="组") as pbar:
                if global_skipped > 0:
                    pbar.update(global_skipped)  # 缓存跳过部分预先推进进度条

                job_list = uncomputed_combos
                for i in range(0, len(job_list), BATCH_SIZE):
                    batch = job_list[i:i + BATCH_SIZE]
                    chunks = [batch[j:j + CHUNK_SIZE] for j in range(0, len(batch), CHUNK_SIZE)]
                    futures = [executor.submit(_worker_process_chunk, chunk) for chunk in chunks]

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

                    del futures, batch_results, batch, chunks

        # 2.5 收尾: 关闭 writer、复盘、回写缓存
        has_output = writer is not None
        if has_output:
            writer.close()

        _print_final_report(final_fund_count, N, total_combos,
                            global_skipped, global_computed, output_parquet, has_output)

        _update_cache_from_new_file(output_parquet, cache_file, computed_set)

        N += 1  # 向更高维度发起挑战