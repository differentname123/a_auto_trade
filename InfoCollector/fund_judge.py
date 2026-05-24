import os
import itertools
import math  # 新增导入 math 用于计算组合总数
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime  # 新增导入用于格式化日志时间
import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm  # 新增导入 tqdm 进度条库


def get_current_time():
    """辅助函数：获取当前格式化时间用于日志打印"""
    return datetime.now().strftime("%H:%M:%S")


def evaluate_fof_portfolio(df_list, date_col='净值日期', nav_col='复权净值',
                           rebalance_days=30, max_history_days=5 * 252,
                           max_mdd_limit=0.2, hurdle_rate=0.2):
    """
    第一性原理 FOF 绝对收益评估引擎 (V4.2 终极数学严谨生产版)
    """
    if not df_list or len(df_list) < 2:
        return {"error": "Need at least 2 funds", "Total_Score": 0.0}

    # ================= 1. 数据清洗与对齐 =================
    nav_series_list = []

    for i, df in enumerate(df_list):
        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col])
        temp_df = temp_df.set_index(date_col).sort_index()
        temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
        nav_series = temp_df[nav_col].rename(f'fund_{i}')
        nav_series_list.append(nav_series)

    merged_nav = pd.concat(nav_series_list, axis=1, join='outer')
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

    # 记录缺失率用于 VETO
    missing_ratios = merged_nav.isna().sum() / len(merged_nav)
    metrics_max_missing_ratio = float(missing_ratios.max())

    valid_start = merged_nav.index[0]
    valid_end = merged_nav.index[-1]

    raw_max_date = merged_nav_raw_index[-1]
    if (raw_max_date - valid_end).days > 35:
        return {"error": f"Zombie fund detected. Cutoff at {valid_end.strftime('%Y-%m-%d')}", "Total_Score": 0.0}

    # 先用 ffill 填平缺失期，再算收益率，彻底杜绝跨期真实收益被蒸发的灾难
    merged_nav_ffilled = merged_nav.ffill()
    fund_rets = merged_nav_ffilled.pct_change().fillna(0.0)

    # 抓捕最长连续0收益天数
    zero_rets_df = (fund_rets.abs() < 1e-8).astype(int)
    max_continuous_zeros = 0
    for col in zero_rets_df.columns:
        consecutive = zero_rets_df[col].groupby((zero_rets_df[col] == 0).cumsum()).sum().max()
        max_continuous_zeros = max(max_continuous_zeros, consecutive)

    n_days = len(fund_rets)
    n_funds = len(df_list)
    fund_rets_arr = fund_rets.values

    # ================= 2 & 3. 核心评估闭包引擎 =================
    def _evaluate_single_path(current_reb_days, offset=0, w_init=None):
        synth_ret_arr = np.zeros(n_days)

        if w_init is None:
            w = np.ones(n_funds) / n_funds
        else:
            w = np.array(w_init)

        for t in range(n_days):
            daily_port_ret = np.sum(w * fund_rets_arr[t])
            synth_ret_arr[t] = daily_port_ret

            denom = 1 + daily_port_ret if (1 + daily_port_ret) > 1e-9 else 1e-9
            w = w * (1 + fund_rets_arr[t]) / denom

            if t >= current_reb_days and (t - offset) % current_reb_days == 0:
                if w_init is None:
                    w = np.ones(n_funds) / n_funds
                else:
                    w = np.array(w_init)

        synth_ret = pd.Series(synth_ret_arr, index=fund_rets.index)
        synth_eq = (1 + synth_ret).cumprod()

        metrics = {
            'Start_Date': merged_nav.index[0].strftime('%Y-%m-%d'),
            'End_Date': merged_nav.index[-1].strftime('%Y-%m-%d'),
            'Total_Days': n_days,
            'n_funds': n_funds
        }

        # --- 核心指标计算 ---
        # [致命Bug修复 1]: 直接基于隐含本金 1.0 的期末累计净值计算年化，防止抹除首日收益
        cagr = float(synth_eq.iloc[-1] ** (252 / n_days) - 1)
        metrics['CAGR'] = cagr

        # [致命Bug修复 2]: 强行锁定历史最高水位的底线为初始本金1.0，防止开局暴跌被豁免
        cum_max = synth_eq.cummax().clip(lower=1.0)
        drawdowns = (synth_eq - cum_max) / cum_max
        metrics['Max_Drawdown'] = float(drawdowns.min())

        is_drawdown = drawdowns < 0
        recovery_groups = (~is_drawdown).cumsum()
        metrics['Max_Recovery_Days'] = int(is_drawdown.groupby(recovery_groups).sum().max())

        log_eq = np.log(synth_eq.clip(lower=1e-9).values)

        window = min(252, n_days)
        rolling_r2 = []
        for st in range(0, n_days - window + 1, 21):
            y_sub = log_eq[st:st + window]
            _, _, r_sub, _, _ = stats.linregress(np.arange(window), y_sub)
            rolling_r2.append(r_sub ** 2 if pd.notna(r_sub) else 0.0)
        metrics['Worst_Rolling_1Y_R2'] = float(min(rolling_r2))

        ar1 = synth_ret.autocorr(lag=1)
        metrics['AR1_Coefficient'] = float(ar1) if pd.notna(ar1) else 1.0

        # 新增高价值字段：波动率、夏普、卡玛、日胜率
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


# ================= 新增：多进程 I/O 加载模块 =================

def _load_single_nav(f):
    """
    独立的单进程函数：加载并清洗单只基金的净值数据
    """
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


def precompute_correlations(result_csv='fund_data/all_funds_result.csv', corr_csv='fund_data/fund_correlations.csv',
                            max_workers=10):
    """
    加载所有基金记录的净值数据，进行时间对齐后计算相关性系数，并落盘存储。
    """
    print(f"\n[{get_current_time()}] ---------------- 开始计算全市场基金相关性矩阵 ----------------")
    if not os.path.exists(result_csv):
        print(f"[{get_current_time()}] 错误: 未找到汇总文件 {result_csv}")
        return pd.DataFrame()

    df_results = pd.read_csv(result_csv)
    files = df_results['adj_nav_file'].dropna().tolist()

    print(f"[{get_current_time()}] 正在使用 {max_workers} 个并行进程读取 {len(files)} 只基金的净值数据...")
    nav_series_list = []

    # 使用多进程加速文件读取
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_single_nav, f) for f in files]
        for future in tqdm(as_completed(futures), total=len(files), desc="并行加载净值数据", unit="只"):
            res = future.result()
            if res is not None:
                nav_series_list.append(res)

    if not nav_series_list:
        print(f"[{get_current_time()}] 警告: 未能成功加载任何有效净值数据！")
        return pd.DataFrame()

    print(f"[{get_current_time()}] 净值读取完毕。成功加载 {len(nav_series_list)} 只基金。")
    print(f"[{get_current_time()}] 正在拼接内存结构并按日期自动对齐 (可能占用较高内存)...")

    # 对齐所有时间轴，并向下填充处理缺失日期来计算真实日频相关性
    merged_nav = pd.concat(nav_series_list, axis=1, join='outer')
    merged_nav_ffilled = merged_nav.ffill()

    print(f"[{get_current_time()}] 正在计算每日收益率并生成 Pearson 相关性矩阵...")
    returns = merged_nav_ffilled.pct_change()

    # 计算皮尔逊相关系数
    corr_matrix = returns.corr()

    output_dir = os.path.dirname(corr_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"[{get_current_time()}] 矩阵运算完成！正在落盘保存至: {corr_csv}")
    corr_matrix.to_csv(corr_csv)
    print(f"[{get_current_time()}] ---------------- 相关性矩阵预处理流程顺利结束 ----------------\n")
    return corr_matrix


# ================= 多进程并行调度模块 =================

def _worker_process_combo(combo_files):
    """
    独立的工作进程函数：接收文件路径元组，读取CSV并调用评估引擎，返回带文件名的结果
    """
    file_names = [os.path.basename(f) for f in combo_files]
    combo_name_str = " + ".join(file_names)

    try:
        # 读取CSV文件为DataFrame列表
        df_list = [pd.read_csv(f) for f in combo_files]

        # 调用核心引擎
        result = evaluate_fof_portfolio(df_list)

        # 将文件名添加到结果字典中
        result['组合文件名'] = combo_name_str
        return result

    except Exception as e:
        # 即使发生读取或计算异常，依然捕获并返回包含错误信息的结果字典
        return {
            '组合文件名': combo_name_str,
            'error': f"处理异常: {str(e)}",
            'Total_Score': 0.0
        }


def filter_fund_pool(df_results, active_cache='temp/active_fund_codes.csv', min_annual_return=0.15):
    """
    独立的基金池漏斗过滤函数：执行可申购白名单过滤与财务指标硬性过滤。

    参数:
        df_results: DataFrame, 包含所有基金的基础表现指标
        active_cache: str, 最新可申购基金名单的 CSV 路径
        min_annual_return: float, 允许入池的最低年化收益阈值

    返回:
        DataFrame: 经过严格过滤并按年化收益降序排列的精选基金池
    """
    if df_results is None or df_results.empty:
        print("❌ 输入的 DataFrame 为空。")
        return pd.DataFrame()

    original_count = len(df_results)
    df_filtered = df_results.copy()

    # ================= 1. 实盘可申购状态（白名单）校验 =================
    if os.path.exists(active_cache):
        try:
            df_active = pd.read_csv(active_cache, dtype=str)
            if '基金代码' in df_active.columns:
                active_codes_set = set(df_active['基金代码'].str.strip().str.zfill(6).tolist())

                # 动态自适应：提取被过滤表的代码列
                if '基金代码' in df_filtered.columns:
                    target_codes = df_filtered['基金代码'].astype(str).str.strip().str.zfill(6)
                elif 'fund_code' in df_filtered.columns:
                    target_codes = df_filtered['fund_code'].astype(str).str.strip().str.zfill(6)
                else:
                    # 终极兜底：从文件名 "fund_data/000001_adj_nav.csv" 提取连续6位数字
                    target_codes = df_filtered['adj_nav_file'].apply(
                        lambda x: re.search(r'(\d{6})', str(x)).group(1) if pd.notna(x) and re.search(r'(\d{6})',
                                                                                                      str(x)) else "000000"
                    )

                # 执行强制交集过滤
                df_filtered = df_filtered[target_codes.isin(active_codes_set)].copy()
                active_filtered_count = len(df_filtered)
            else:
                print(f"⚠️ [状态校验跳过] {active_cache} 中未找到 '基金代码' 列。")
                active_filtered_count = original_count
        except Exception as e:
            print(f"⚠️ [状态校验异常] 读取白名单失败，跳过 ({str(e)})")
            active_filtered_count = original_count
    else:
        print(f"ℹ️ [状态校验跳过] 未发现可申购白名单文件: {active_cache}")
        active_filtered_count = original_count

    if df_filtered.empty:
        print("❌ 状态校验后无可用基金。")
        return df_filtered

    # ================= 2. 财务表现与数据质量底线校验 =================
    condition = (
            (df_filtered['total_active_days'] > 300) &
            (df_filtered['annualized_return'] > min_annual_return) &
            (df_filtered['missing_ratio'] < 0.02) &
            (df_filtered['max_zeros'] < 5) &
            (df_filtered['max_drawdown'] > -0.5)
    )

    # 按照年化收益降序排列（确保后续相关性剔除时，始终保留收益更高的“头雁”基金）
    df_final = df_filtered[condition].sort_values(by='annualized_return', ascending=False)
    final_count = len(df_final)

    # ================= 3. 打印漏斗统计日志 =================
    print("\n" + "=" * 55)
    print("🎯 基金池初筛漏斗统计:")
    print(f"  1. 初始输入总数    : {original_count} 只")
    print(f"  2. 可申购状态通过  : {active_filtered_count} 只")
    print(f"  3. 财务与质量达标  : {final_count} 只 (作为高优候选池)")
    print("=" * 55 + "\n")

    return df_final


def run_batch_evaluation(result_csv='fund_data/all_funds_result.csv', combo_size=4, max_workers=10, corr_threshold=0.9,
                         corr_matrix=None):
    """
    核心调度函数：读取汇总数据过滤、根据相关性去重、生成组合、分配多进程并发任务、落盘CSV
    新增外部传入 corr_matrix，避免内部重复计算或重复读取文件
    """
    original_count = 0
    initial_filtered_count = 0
    final_fund_count = 0

    print(f"[{get_current_time()}] 步骤 1: 扫描汇总文件进行基础绩效条件筛选...")

    # 1. 读取 all_funds_result 数据并初步过滤
    if os.path.exists(result_csv):
        df_results = pd.read_csv(result_csv)
        original_count = len(df_results)

        df_filtered = filter_fund_pool(df_results, min_annual_return=0.15)
        initial_filtered_count = len(df_filtered)
    # 防御性判断
    if initial_filtered_count == 0:
        print(f"[{get_current_time()}] 初步条件过滤后，符合要求的基金数量为0，退出流程。")
        return

    # 2. 检查并加载相关性矩阵 (优先使用外部传入)
    if corr_matrix is None:
        corr_csv = 'fund_data/fund_correlations.csv'
        print(f"[{get_current_time()}] 步骤 2: 检查底层相关性矩阵是否存在...")
        if not os.path.exists(corr_csv):
            corr_matrix = precompute_correlations(result_csv, corr_csv, max_workers=max_workers)
        else:
            print(f"[{get_current_time()}] 已发现现有相关性文件 {corr_csv}，直接加载缓存...")
            corr_matrix = pd.read_csv(corr_csv, index_col=0)
    else:
        print(f"[{get_current_time()}] 步骤 2: 成功接收并使用外部传入的相关性矩阵...")

    if corr_matrix.empty:
        print(f"[{get_current_time()}] 警告: 无法加载相关性矩阵，将跳过相关性过滤。")
        target_files = df_filtered['adj_nav_file'].dropna().tolist()
    else:
        # 3. 执行高相关性“优胜劣汰”过滤
        print(f"[{get_current_time()}] 步骤 3: 启动优胜劣汰相关性过滤引擎 (剔除阈值: {corr_threshold})...")
        selected_files = []
        for f in df_filtered['adj_nav_file'].dropna():
            if f not in corr_matrix.columns:
                continue

            is_highly_correlated = False
            # 与已经保留下来的优秀基金（排在前面的）对比
            for sel_f in selected_files:
                if sel_f in corr_matrix.columns:
                    corr_val = corr_matrix.loc[f, sel_f]
                    # 如果相关性过高，直接标记舍弃该基金
                    if pd.notna(corr_val) and corr_val > corr_threshold:
                        is_highly_correlated = True
                        break

            # 只有与已选“种子”基金都不高度重合的情况下，才收入组合池
            if not is_highly_correlated:
                selected_files.append(f)

        target_files = selected_files

    # ================= 数据漏斗结果打印 =================
    final_fund_count = len(target_files)
    print("\n" + "=" * 50)
    print(f"[{get_current_time()}] 🎯 基金池过滤漏斗统计:")
    print(f"  1. 原始总数量: {original_count} 只")
    print(f"  2. 初步条件过滤后: {initial_filtered_count} 只 (基于时间/收益/回撤/缺失等指标)")
    print(f"  3. 相关性(<{corr_threshold})去重后: {final_fund_count} 只 (保留高优标的)")
    print("=" * 50 + "\n")
    # =====================================================

    # 4. 动态确定输出文件及组合可行性判断
    if final_fund_count < combo_size:
        print(
            f"[{get_current_time()}] 放弃评估: 过滤后基金数量不足 ({final_fund_count} 个)，无法生成大小为 {combo_size} 的组合。")
        return

    output_csv = f'fund_data/fof_evaluation_results_{combo_size}d_pool{final_fund_count}.csv'

    # 新增跳过判断：如果目标组合评估文件已存在，则直接跳过本次评估
    if os.path.exists(output_csv):
        print(f"[{get_current_time()}] ⏭️ 目标结果文件 [{output_csv}] 已存在，直接跳过当前 {combo_size} 维度的计算。")
        return

    # 获取两两(或多维)组合 (惰性迭代器 + 总数计算)
    total_combos = math.comb(final_fund_count, combo_size)
    combos = itertools.combinations(target_files, combo_size)

    print(f"[{get_current_time()}] 步骤 4: 构建并行任务树并下发核心计算...")
    print(f"[{get_current_time()}] 任务构建: 将产生 {total_combos} 种组合，落盘文件: {output_csv}")
    print(f"[{get_current_time()}] 开启 {max_workers} 个并行进程计算...")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 5. 分块流式处理并落盘（彻底解决 137 OOM 内存杀手问题）
    batch_size = 50000  # 内存中最多保留 5 万个任务/结果
    is_first_write = True

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=total_combos, desc=f"评估 {combo_size} 维组合进度", unit="组") as pbar:
            while True:
                # 每次只从无穷尽的组合中切出 batch_size 大小的一块丢进内存
                batch_combos = list(itertools.islice(combos, batch_size))
                if not batch_combos:
                    break

                # 提交这一批次的任务
                futures = [executor.submit(_worker_process_combo, combo) for combo in batch_combos]

                batch_results = []
                for future in as_completed(futures):
                    batch_results.append(future.result())
                    pbar.update(1)

                # 当前批次跑完，直接追加写入磁盘，清空内存
                if batch_results:
                    results_df = pd.DataFrame(batch_results)

                    # 🚀 定义所有可能出现的列名大满贯 (Schema) 强制对齐
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

                    # 🚀 强制对齐列：缺少的列会自动填充 NaN，确保每一批次的结构绝对一致
                    results_df = results_df.reindex(columns=all_possible_columns)

                    mode = 'w' if is_first_write else 'a'
                    header = is_first_write
                    results_df.to_csv(output_csv, mode=mode, header=header, index=False, encoding='utf-8-sig')
                    is_first_write = False

                # 强行释放本轮循环产生的千万级别对象占用
                del futures
                del batch_results
                del batch_combos

    print(f"[{get_current_time()}] 🎉 全部计算完成！结果已成功保存至: {output_csv}")


# 如果您要在脚本中直接运行，可以调用以下启动代码：
if __name__ == '__main__':
    # 统一指定并发核心数，此处全局修改为 10 加快速度
    GLOBAL_MAX_WORKERS = 20

    result_csv_path = 'fund_data/all_funds_result.csv'
    corr_csv_path = 'fund_data/fund_correlations.csv'

    # 提前在全局初始化并读取/计算 corr_matrix
    print(f"[{get_current_time()}] 【全局初始化】正在准备全市场相关性矩阵...")
    if not os.path.exists(corr_csv_path):
        global_corr_matrix = precompute_correlations(result_csv_path, corr_csv_path, max_workers=GLOBAL_MAX_WORKERS)
    else:
        print(f"[{get_current_time()}] 已发现现有相关性文件 {corr_csv_path}，直接加载缓存...")
        global_corr_matrix = pd.read_csv(corr_csv_path, index_col=0)

    for i in range(4):
        combo_size = 2 + i  # 从2维组合开始，逐步增加维度
        print(f"\n{'#' * 70}\n[{get_current_time()}] 【启动引擎】正在评估 {combo_size} 维 FOF 组合\n{'#' * 70}")

        # 这里已经不用传 output_csv 参数了，它会在方法内基于最终保留数量动态拼出
        # 且将全局的 corr_matrix 作为参数传入，满足条件1要求
        run_batch_evaluation(
            result_csv=result_csv_path,
            combo_size=combo_size,
            max_workers=GLOBAL_MAX_WORKERS,
            corr_matrix=global_corr_matrix
        )