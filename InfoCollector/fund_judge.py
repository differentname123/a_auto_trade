import os
import itertools
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from numba import njit

# ================= [架构重构] 子进程全局变量容器 =================
WORKER_MASTER_DF = None


def _init_worker(master_df):
    """
    此函数会在每个新的工作进程启动时执行一次！
    它将主进程传递过来的巨大 DataFrame 锚定在子进程的本地内存中。
    """
    global WORKER_MASTER_DF
    WORKER_MASTER_DF = master_df


def get_current_time():
    """辅助函数：获取当前格式化时间用于日志打印"""
    return datetime.now().strftime("%H:%M:%S")


# ================= [A级优化] Numba 高速核心循环引擎 =================
@njit(fastmath=True, cache=True)
def _fast_simulate_path(fund_rets_arr, w_init, current_reb_days, offset, n_days, n_funds):
    """
    将核心路径演化逻辑抽离并交由 Numba 编译为机器码，
    速度提升百倍，彻底剥离 Pandas 在循环中的消耗。
    """
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


def evaluate_fof_portfolio_fast(merged_nav, rebalance_days=30, max_history_days=5 * 252,
                                max_mdd_limit=0.3, hurdle_rate=0.2):
    """
    第一性原理 FOF 绝对收益评估引擎 (极速切片版)
    """
    n_funds = merged_nav.shape[1]
    merged_nav_raw_index = merged_nav.index

    # 严禁压缩时间轴，基准日历为骨架。找到有效起止区间
    valid_mask = merged_nav.notna().all(axis=1)
    if not valid_mask.any():
        return {"error": "No overlapping data", "Total_Score": 0.0}

    valid_start = valid_mask.idxmax()
    valid_end = valid_mask[::-1].idxmax()

    merged_nav = merged_nav.loc[valid_start:valid_end]

    if len(merged_nav) > max_history_days:
        merged_nav = merged_nav.iloc[-max_history_days:]

    if len(merged_nav) < 252:
        return {"error": "Common data length < 252 days", "Total_Score": 0.0}

    missing_ratios = merged_nav.isna().sum() / len(merged_nav)
    metrics_max_missing_ratio = float(missing_ratios.max())

    valid_start = merged_nav.index[0]
    valid_end = merged_nav.index[-1]

    raw_max_date = merged_nav_raw_index[-1]
    if (raw_max_date - valid_end).days > 35:
        return {"error": f"Zombie fund detected. Cutoff at {valid_end.strftime('%Y-%m-%d')}", "Total_Score": 0.0}

    merged_nav_ffilled = merged_nav.ffill()
    fund_rets = merged_nav_ffilled.pct_change().fillna(0.0)

    zero_rets_df = (fund_rets.abs() < 1e-8).astype(int)
    max_continuous_zeros = 0
    for col in zero_rets_df.columns:
        consecutive = zero_rets_df[col].groupby((zero_rets_df[col] == 0).cumsum()).sum().max()
        max_continuous_zeros = max(max_continuous_zeros, consecutive)

    n_days = len(fund_rets)

    # ================= [防崩溃修复 1] 强制转换为 C-连续内存 =================
    # 防止 Numba 因 Pandas 切片底层内存不连续而触发崩溃或严重掉速
    fund_rets_arr = np.ascontiguousarray(fund_rets.values, dtype=np.float64)

    # ================= 2 & 3. 核心评估闭包引擎 =================
    def _evaluate_single_path(current_reb_days, offset=0, w_init=None):
        if w_init is None:
            w_init_arr = np.ones(n_funds, dtype=np.float64) / n_funds
        else:
            w_init_arr = np.array(w_init, dtype=np.float64)

        # 调用 Numba JIT 编译的纯机器码函数运行回测
        synth_ret_arr = _fast_simulate_path(fund_rets_arr, w_init_arr, current_reb_days, offset, n_days, n_funds)

        synth_ret = pd.Series(synth_ret_arr, index=fund_rets.index)
        synth_eq = (1 + synth_ret).cumprod()

        metrics = {
            'Start_Date': merged_nav.index[0].strftime('%Y-%m-%d'),
            'End_Date': merged_nav.index[-1].strftime('%Y-%m-%d'),
            'Total_Days': n_days,
            'n_funds': n_funds
        }

        # --- 核心指标计算 ---
        cagr = float(synth_eq.iloc[-1] ** (252 / n_days) - 1)
        metrics['CAGR'] = cagr

        cum_max = synth_eq.cummax().clip(lower=1.0)
        drawdowns = (synth_eq - cum_max) / cum_max
        metrics['Max_Drawdown'] = float(drawdowns.min())

        is_drawdown = drawdowns < 0
        recovery_groups = (~is_drawdown).cumsum()
        metrics['Max_Recovery_Days'] = int(is_drawdown.groupby(recovery_groups).sum().max())

        log_eq = np.log(synth_eq.clip(lower=1e-9).values)

        window = min(252, n_days)
        rolling_r2 = []
        x_arr = np.arange(window)

        for st in range(0, n_days - window + 1, 21):
            y_sub = log_eq[st:st + window]
            r_mat = np.corrcoef(x_arr, y_sub)
            if r_mat.shape == (2, 2) and not np.isnan(r_mat[0, 1]):
                rolling_r2.append(r_mat[0, 1] ** 2)
            else:
                rolling_r2.append(0.0)
        metrics['Worst_Rolling_1Y_R2'] = float(min(rolling_r2)) if rolling_r2 else 0.0

        ar1 = synth_ret.autocorr(lag=1)
        metrics['AR1_Coefficient'] = float(ar1) if pd.notna(ar1) else 1.0

        vol_annual = synth_ret.std() * np.sqrt(252)
        metrics['Annualized_Volatility'] = float(vol_annual)
        metrics['Sharpe_Ratio'] = float(cagr / vol_annual) if vol_annual > 1e-6 else 0.0
        metrics['Calmar_Ratio'] = float(cagr / abs(metrics['Max_Drawdown'])) if abs(
            metrics['Max_Drawdown']) > 1e-6 else 0.0
        metrics['Daily_Win_Rate'] = float((synth_ret > 0).sum() / n_days) if n_days > 0 else 0.0

        n_worst = max(5, int(len(synth_ret) * 0.05))
        worst_dates = synth_ret.nsmallest(n_worst).index
        worst_fund_rets = fund_rets.loc[worst_dates]

        if len(worst_fund_rets) > 3:
            corr_matrix_local = worst_fund_rets.corr().values
            triu_idx = np.triu_indices_from(corr_matrix_local, k=1)
            if len(triu_idx[0]) > 0:
                with np.errstate(invalid='ignore'):
                    max_corr = np.nanmax(corr_matrix_local[triu_idx])
                metrics['Downside_Correlation'] = float(max_corr) if not np.isnan(max_corr) else 0.5
            else:
                metrics['Downside_Correlation'] = 0.5
        else:
            metrics['Downside_Correlation'] = 0.5

        # --- 4. 底线判决 (Iron VETO) ---
        vetoes = {
            "VETO_Hurdle_Rate": metrics['CAGR'] < hurdle_rate,
            "VETO_Drawdown_Crash": abs(metrics['Max_Drawdown']) > max_mdd_limit,
            "VETO_Fake_Smooth": (metrics['AR1_Coefficient'] > 0.35) and (vol_annual < 0.03),
            "VETO_Endless_Bleeding": metrics['Max_Recovery_Days'] > 180,
            "VETO_Data_Distortion": (metrics_max_missing_ratio > 0.05) or (max_continuous_zeros > 10)
        }
        metrics.update(vetoes)

        if any(vetoes.values()):
            return metrics, 0.0

        # --- 第一性原理综合打分 ---
        excess_cagr = max(0.0, metrics['CAGR'] - hurdle_rate)
        adj_mdd = max(abs(metrics['Max_Drawdown']), 0.01)

        base_score = (excess_cagr / adj_mdd) * metrics['Worst_Rolling_1Y_R2']
        time_multiplier = min(1.0, np.sqrt(n_days / 756))

        if metrics['Downside_Correlation'] > 0.8:
            p_corr = 0.1
        elif metrics['Downside_Correlation'] > 0.5:
            p_corr = np.clip(1.0 - (metrics['Downside_Correlation'] - 0.5) * 1.0, 0.5, 1.0)
        else:
            p_corr = 1.0

        path_score = base_score * time_multiplier * p_corr
        return metrics, max(0.0, float(path_score))

    # ================= 5. 多维扰动验证闭环 =================
    final_metrics, score_base = _evaluate_single_path(rebalance_days, offset=0)

    if score_base == 0.0:
        final_metrics['Total_Score'] = 0.0
        final_metrics['VETO_Perturbation_Death'] = False
        return final_metrics

    metrics_w = None
    score_weight = float('inf')

    half_offset = rebalance_days // 2

    for seed_val in [1024, 2048, 4096]:
        np.random.seed(seed_val)
        shift_vector = np.random.uniform(-0.05, 0.05, n_funds)
        w_perturbed = np.ones(n_funds) / n_funds + shift_vector
        w_perturbed = np.clip(w_perturbed, 0.01, 1.0)
        w_perturbed = w_perturbed / np.sum(w_perturbed)

        m_w, s_w = _evaluate_single_path(rebalance_days, offset=half_offset, w_init=w_perturbed)
        if s_w < score_weight:
            score_weight = s_w
            metrics_w = m_w

    if score_weight == float('inf'):
        score_weight = 0.0

    final_metrics['Total_Score'] = min(score_base, score_weight)

    final_metrics['VETO_Perturbation_Death'] = final_metrics['Total_Score'] == 0.0
    if final_metrics['VETO_Perturbation_Death'] and metrics_w is not None:
        if score_weight == 0.0:
            for k, v in metrics_w.items():
                if k.startswith("VETO_") and v is True:
                    final_metrics[k + "_in_Perturb"] = True

    return final_metrics


def _worker_process_combo(combo_files):
    """独立的工作进程函数：通过全局变量从 Master DataFrame 中安全切片"""
    global WORKER_MASTER_DF
    file_names = [os.path.basename(f) for f in combo_files]
    combo_name_str = " + ".join(file_names)

    try:
        # [极致飞跃]: 仅进行列切片，微秒级操作！完美消灭 pd.concat
        merged_nav = WORKER_MASTER_DF[list(combo_files)]

        # 将切片直接传入极速版评估引擎
        result = evaluate_fof_portfolio_fast(merged_nav)

        # 挂载结果
        result['组合文件名'] = combo_name_str
        return result

    except Exception as e:
        return {
            '组合文件名': combo_name_str,
            'error': f"处理异常: {str(e)}",
            'Total_Score': 0.0
        }


# ================= [终极性能优化 2] 批处理函数，消灭 IPC 通信开销 =================
def _worker_process_chunk(combo_chunk):
    """批处理工作进程：接收一个包含数百个组合的 List，彻底抹平 IPC 进程间通信开销。"""
    results = []
    for combo_files in combo_chunk:
        res = _worker_process_combo(combo_files)
        results.append(res)
    return results


# ================= 多进程 I/O 加载模块 =================
def _load_single_nav(f):
    """独立的单进程函数：加载并清洗单只基金的净值数据"""
    try:
        if os.path.exists(f):
            temp_df = pd.read_csv(f)
            temp_df['净值日期'] = pd.to_datetime(temp_df['净值日期'])
            temp_df = temp_df.set_index('净值日期').sort_index()
            temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
            return temp_df['复权净值'].rename(f)
    except Exception:
        pass
    return None


def precompute_correlations(result_csv='fund_data/all_funds_result.csv',
                            corr_csv='fund_data/fund_correlations.csv',
                            downside_csv='fund_data/fof_evaluation_results_2d_pool3917.csv',
                            max_workers=10,
                            downside_corr_csv='fund_data/fund_downside_correlations.csv'):
    """预计算双相关矩阵"""
    print(f"\n[{get_current_time()}] ---------------- 开始准备全市场相关性双矩阵 ----------------")

    downside_corr_matrix = pd.DataFrame()

    if os.path.exists(downside_corr_csv):
        print(f"[{get_current_time()}] 已发现现有下行相关性文件 {downside_corr_csv}，直接加载缓存...")
        downside_corr_matrix = pd.read_csv(downside_corr_csv, index_col=0)
    elif os.path.exists(downside_csv):
        print(f"[{get_current_time()}] 正在从 2 维历史组合文件 {downside_csv} 提取 Downside_Correlation 矩阵...")
        try:
            df_2d = pd.read_csv(downside_csv)
            if '组合文件名' in df_2d.columns and 'Downside_Correlation' in df_2d.columns:
                records = []
                for _, row in df_2d.iterrows():
                    combo = row['组合文件名']
                    if pd.isna(combo): continue
                    parts = [p.strip() for p in str(combo).split('+')]
                    if len(parts) == 2:
                        records.append({'f1': f'fund_data/nav/{parts[0]}', 'f2': f'fund_data/nav/{parts[1]}',
                                        'dc': row['Downside_Correlation']})
                        records.append({'f1': f'fund_data/nav/{parts[1]}', 'f2': f'fund_data/nav/{parts[0]}',
                                        'dc': row['Downside_Correlation']})
                if records:
                    df_records = pd.DataFrame(records)
                    downside_corr_matrix = df_records.groupby(['f1', 'f2'])['dc'].mean().unstack()
                    print(f"[{get_current_time()}] 成功构建下行相关性矩阵，涵盖 {len(downside_corr_matrix)} 只标的。")
                    output_dir = os.path.dirname(downside_corr_csv)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    downside_corr_matrix.to_csv(downside_corr_csv)
        except Exception as e:
            print(f"[{get_current_time()}] 提取 Downside_Correlation 矩阵失败: {e}")
    else:
        print(f"[{get_current_time()}] 未发现 2 维组合文件 {downside_csv}，跳过下行相关性提取。")

    if os.path.exists(corr_csv):
        print(f"[{get_current_time()}] 已发现现有全天候相关性文件 {corr_csv}，直接加载缓存...")
        corr_matrix = pd.read_csv(corr_csv, index_col=0)
        return corr_matrix, downside_corr_matrix

    if not os.path.exists(result_csv):
        print(f"[{get_current_time()}] 错误: 未找到汇总文件 {result_csv}")
        return pd.DataFrame(), downside_corr_matrix

    df_results = pd.read_csv(result_csv)
    files = df_results['adj_nav_file'].dropna().tolist()

    print(f"[{get_current_time()}] 正在并行读取 {len(files)} 只基金...")
    nav_series_list = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_single_nav, f) for f in files]
        for future in tqdm(as_completed(futures), total=len(files), desc="并行加载数据"):
            res = future.result()
            if res is not None:
                nav_series_list.append(res)

    if not nav_series_list:
        return pd.DataFrame(), downside_corr_matrix

    merged_nav = pd.concat(nav_series_list, axis=1, join='outer').ffill()
    returns = merged_nav.pct_change()
    corr_matrix = returns.corr()

    output_dir = os.path.dirname(corr_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    corr_matrix.to_csv(corr_csv)
    return corr_matrix, downside_corr_matrix


def filter_fund_pool(df_results, active_cache='temp/active_fund_codes.csv', min_annual_return=0.15, min_day=1000):
    """独立的基金池漏斗过滤函数"""
    if df_results is None or df_results.empty:
        return pd.DataFrame()

    original_count = len(df_results)
    df_filtered = df_results.copy()

    if os.path.exists(active_cache):
        try:
            df_active = pd.read_csv(active_cache, dtype=str)
            if '基金代码' in df_active.columns:
                active_codes_set = set(df_active['基金代码'].str.strip().str.zfill(6).tolist())
                if '基金代码' in df_filtered.columns:
                    target_codes = df_filtered['基金代码'].astype(str).str.strip().str.zfill(6)
                elif 'fund_code' in df_filtered.columns:
                    target_codes = df_filtered['fund_code'].astype(str).str.strip().str.zfill(6)
                else:
                    target_codes = df_filtered['adj_nav_file'].apply(
                        lambda x: re.search(r'(\d{6})', str(x)).group(1) if pd.notna(x) and re.search(r'(\d{6})',
                                                                                                      str(x)) else "000000"
                    )
                df_filtered = df_filtered[target_codes.isin(active_codes_set)].copy()
                active_filtered_count = len(df_filtered)
            else:
                active_filtered_count = original_count
        except Exception:
            active_filtered_count = original_count
    else:
        active_filtered_count = original_count

    if df_filtered.empty:
        return df_filtered

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


def run_batch_evaluation(result_csv='fund_data/all_funds_result.csv', combo_size=4, max_workers=10,
                         corr_threshold=0.9, corr_matrix=None,
                         downside_corr_matrix=None, downside_corr_threshold=0.0, min_day=1000):
    # ================= [防崩溃修复 2] 预热 Numba 编译器 =================
    print(f"[{get_current_time()}] 正在预热 Numba 核心引擎，防止多进程缓存踩踏...")
    _fast_simulate_path(np.zeros((100, 2), dtype=np.float64),
                        np.array([0.5, 0.5], dtype=np.float64),
                        30, 0, 100, 2)
    # ===============================================================

    original_count = 0
    initial_filtered_count = 0
    final_fund_count = 0

    print(f"[{get_current_time()}] 步骤 1: 扫描汇总文件进行基础绩效条件筛选...")

    if os.path.exists(result_csv):
        df_results = pd.read_csv(result_csv)
        original_count = len(df_results)
        df_filtered = filter_fund_pool(df_results, min_annual_return=0.05, min_day=min_day)
        initial_filtered_count = len(df_filtered)

    if initial_filtered_count == 0:
        print(f"[{get_current_time()}] 符合要求的基金数量为0，退出流程。")
        return

    if corr_matrix is None:
        corr_csv = 'fund_data/fund_correlations.csv'
        downside_csv = 'fund_data/fof_evaluation_results_2d_pool3917.csv'
        corr_matrix, ds_matrix = precompute_correlations(result_csv, corr_csv, downside_csv, max_workers=max_workers)
        if downside_corr_matrix is None:
            downside_corr_matrix = ds_matrix

    if corr_matrix.empty:
        target_files = df_filtered['adj_nav_file'].dropna().tolist()
    else:
        print(f"[{get_current_time()}] 步骤 3: 启动优胜劣汰过滤...")
        selected_files = []
        for f in df_filtered['adj_nav_file'].dropna():
            if f not in corr_matrix.columns: continue

            is_highly_correlated = False
            for sel_f in selected_files:
                if sel_f in corr_matrix.columns:
                    corr_val = corr_matrix.loc[f, sel_f]
                    if pd.notna(corr_val) and corr_val > corr_threshold:
                        is_highly_correlated = True
                        break

                if downside_corr_matrix is not None and not downside_corr_matrix.empty:
                    if f in downside_corr_matrix.index and sel_f in downside_corr_matrix.columns:
                        dc_val = downside_corr_matrix.loc[f, sel_f]
                        if pd.notna(dc_val) and dc_val > downside_corr_threshold:
                            is_highly_correlated = True
                            break

            if not is_highly_correlated:
                selected_files.append(f)

        target_files = selected_files

    final_fund_count = len(target_files)
    print("\n" + "=" * 50)
    print(f"[{get_current_time()}] 🎯 基金池双矩阵排雷过滤后保留: {final_fund_count} 只")
    print("=" * 50 + "\n")

    if final_fund_count < combo_size:
        print(f"[{get_current_time()}] 过滤后基金不足以生成组合。")
        return

    output_csv = f'fund_data/fof_evaluation_results_{combo_size}d_pool{final_fund_count}_min_day_{min_day}.csv'

    if os.path.exists(output_csv):
        print(f"[{get_current_time()}] ⏭️ 目标文件 [{output_csv}] 已存在，直接跳过计算。")
        return

    print(f"[{get_current_time()}] 🚀 正在构建全局 Master Matrix...")
    all_nav_series = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_single_nav, f) for f in target_files]
        for future in tqdm(as_completed(futures), total=len(target_files), desc="构建Master列数据"):
            res = future.result()
            if res is not None:
                all_nav_series.append(res)

    master_df = pd.concat(all_nav_series, axis=1, join='outer')
    del all_nav_series

    # ================= [防崩溃修复 3] 剔除幽灵标的，强制对齐 =================
    target_files = master_df.columns.tolist()
    final_fund_count = len(target_files)
    # =======================================================================

    print(f"[{get_current_time()}] ✅ 全局 Master Matrix 构建完毕，行数: {len(master_df)}，列数: {final_fund_count}")

    total_combos = math.comb(final_fund_count, combo_size)
    combos = itertools.combinations(target_files, combo_size)

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    batch_size = 50000
    chunk_size = 500  # [开启批处理] 每个进程一次拿走 500 个组合
    is_first_write = True

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(master_df,)) as executor:
        with tqdm(total=total_combos, desc=f"评估 {combo_size} 维组合", unit="组") as pbar:
            while True:
                batch_combos = list(itertools.islice(combos, batch_size))
                if not batch_combos:
                    break

                # [修改为 Chunk 提交流程]
                chunks = [batch_combos[i:i + chunk_size] for i in range(0, len(batch_combos), chunk_size)]
                futures = [executor.submit(_worker_process_chunk, chunk) for chunk in chunks]

                batch_results = []
                for future in as_completed(futures):
                    chunk_res = future.result()
                    batch_results.extend(chunk_res)
                    pbar.update(len(chunk_res))

                if batch_results:
                    results_df = pd.DataFrame(batch_results)

                    all_possible_columns = [
                        '组合文件名', 'Start_Date', 'End_Date', 'Total_Days', 'n_funds',
                        'CAGR', 'Max_Drawdown', 'Max_Recovery_Days', 'Worst_Rolling_1Y_R2',
                        'AR1_Coefficient', 'Annualized_Volatility', 'Sharpe_Ratio',
                        'Calmar_Ratio', 'Daily_Win_Rate', 'Downside_Correlation',
                        'VETO_Hurdle_Rate', 'VETO_Drawdown_Crash', 'VETO_Fake_Smooth',
                        'VETO_Endless_Bleeding', 'VETO_Data_Distortion',
                        'VETO_Perturbation_Death',
                        'VETO_Hurdle_Rate_in_Perturb', 'VETO_Drawdown_Crash_in_Perturb',
                        'VETO_Fake_Smooth_in_Perturb', 'VETO_Endless_Bleeding_in_Perturb',
                        'VETO_Data_Distortion_in_Perturb',
                        'error', 'Total_Score'
                    ]

                    results_df = results_df.reindex(columns=all_possible_columns)

                    mode = 'w' if is_first_write else 'a'
                    header = is_first_write
                    results_df.to_csv(output_csv, mode=mode, header=header, index=False, encoding='utf-8-sig')
                    is_first_write = False

                del futures
                del batch_results
                del batch_combos
                del chunks

    print(f"[{get_current_time()}] 🎉 全部计算完成！结果已成功保存至: {output_csv}")


if __name__ == '__main__':
    GLOBAL_MAX_WORKERS = 25

    result_csv_path = 'fund_data/all_funds_result.csv'
    corr_csv_path = 'fund_data/fund_correlations.csv'
    downside_csv_path = 'fund_data/fof_evaluation_results_2d_pool3917.csv'

    print(f"[{get_current_time()}] 【全局初始化】正在准备全市场相关性双重矩阵...")
    global_corr_matrix, global_downside_corr_matrix = precompute_correlations(
        result_csv=result_csv_path,
        corr_csv=corr_csv_path,
        downside_csv=downside_csv_path,
        max_workers=GLOBAL_MAX_WORKERS
    )

    for i in range(17):
        min_day = 2000 - i * 100
        combo_size = 3
        print(f"\n{'#' * 70}\n[{get_current_time()}] 【启动引擎】正在评估 {combo_size} 维 FOF 组合\n{'#' * 70}")

        run_batch_evaluation(
            result_csv=result_csv_path,
            combo_size=combo_size,
            max_workers=GLOBAL_MAX_WORKERS,
            corr_matrix=global_corr_matrix,
            downside_corr_matrix=global_downside_corr_matrix,
            min_day=min_day
        )