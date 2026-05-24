import glob
import akshare as ak
import pandas as pd
import time
import os
import concurrent.futures
from tqdm import tqdm


def create_folders():
    """创建用于保存数据的文件夹"""
    dirs = ['temp', 'fund_data/nav', 'fund_data/holdings']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def get_active_fund_codes():
    """
    获取当前存续的活跃基金，并过滤掉无需查询重仓股的基金类型（如货币、理财、债券），
    同时过滤掉当前处于【暂停申购】或【封闭期】的不可购买基金。
    """
    # 缓存检查：若有现成列表且包含名称，则直接读取
    cache_file = 'temp/active_fund_codes.csv'
    if os.path.exists(cache_file):
        df_cache = pd.read_csv(cache_file, dtype=str)
        if '基金简称' in df_cache.columns:
            print(f"✅ 检测到本地缓存文件 {cache_file}，直接加载已有基金列表...")
            return dict(zip(df_cache['基金代码'], df_cache['基金简称']))
        else:
            print("🔄 检测到旧版缓存文件缺失基金名称列，将重新获取并更新缓存...")

    print("1. 正在获取全市场基金基础信息...")
    df_all = ak.fund_name_em()

    print("2. 正在获取最新基金动态，剔除【已清盘】及【当前无法申购】的基金...")
    try:
        # 获取最新的每日开放式基金净值列表，里面包含非常关键的“申购状态”
        df_active = ak.fund_open_fund_daily_em()

        # 【新增逻辑】：剔除暂停申购、封闭期等无法购买的基金
        # 我们只保留允许买入的状态：“开放申购” 和 “限制大额申购”（大额限制通常是10万以上，不影响散户购买）
        buyable_status = ['开放申购', '限大额', '场内交易']

        df_buyable = df_active[df_active['申购状态'].isin(buyable_status)]

        active_codes = set(df_buyable['基金代码'].astype(str).str.zfill(6).tolist())
        print(f"   今日共有 {len(df_active)} 只基金更新净值，其中处于可正常申购状态的剩余 {len(df_buyable)} 只。")
    except Exception as e:
        print(f"获取活跃列表失败，将使用全量列表。错误: {e}")
        active_codes = set(df_all['基金代码'].astype(str).str.zfill(6).tolist())

    # 第一步过滤：只保留可申购的基金列表
    df_filtered = df_all[df_all['基金代码'].isin(active_codes)].copy()
    print(f"   df_all长度：{len(df_all)} 剔除不可购买基金后，剩余 {len(df_filtered)} 只候选基金。")

    # 第二步过滤：剔除货币、债券、理财等没有重仓股票的基金
    print("3. 正在剔除货币型、债券型等无重仓股票的基金...")
    exclude_keywords = '货币|理财|纯债|债券|保本'
    df_filtered = df_filtered[~df_filtered['基金类型'].str.contains(exclude_keywords, na=False)]

    # 提取基金代码和名称的字典映射
    fund_dict = dict(zip(df_filtered['基金代码'], df_filtered['基金简称']))
    print(f"✅ 最终过滤完毕！当前可申购且可能含有股票持仓的基金共计 {len(fund_dict)} 只。")

    # 将包含名称的列表存入 temp，供断点续传使用
    df_save = pd.DataFrame({
        '基金代码': list(fund_dict.keys()),
        '基金简称': list(fund_dict.values())
    })
    df_save.to_csv(cache_file, index=False, encoding='utf-8-sig')

    return fund_dict


def fetch_and_save_fund_data(fund_codes, year="2024", test_mode=False):
    """
    遍历基金列表，获取历史净值和重仓股并保存
    :param fund_codes: 基金代码和名称的字典 {code: name}
    :param year: 获取重仓股的年份（例如 "2024"）
    :param test_mode: 是否为测试模式，True 则只跑前 5 只基金
    """
    # 【修改】适应传入的字典类型
    all_codes_list = list(fund_codes.keys())

    if test_mode:
        codes_to_process = all_codes_list[:5]
        print(f"\n⚠️ 当前为【测试模式】，仅获取前 {len(codes_to_process)} 只基金的数据。")
    else:
        codes_to_process = all_codes_list
        print(f"\n🚀 开始全量获取，本次计划处理 {len(codes_to_process)} 只基金...")

    for code in tqdm(codes_to_process, desc="下载进度"):
        requested_this_loop = False  # 记录本轮是否发起了真实网络请求
        fund_name = fund_codes[code]  # 【新增】获取当前基金名称

        # --- 1. 获取历史净值 ---
        nav_file = f"fund_data/nav/{code}_nav.csv"
        # 仅当文件不存在时才下载（断点续传）
        if not os.path.exists(nav_file):
            requested_this_loop = True
            try:
                nav_df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
                if not nav_df.empty:
                    nav_df.to_csv(nav_file, index=False, encoding='utf-8-sig')
                else:
                    open(nav_file, 'w').close()  # 空数据也建空文件占位，防下次重复请求
            except Exception as e:
                open(nav_file, 'w').close()  # 异常时也建空文件占位，防下次重复请求
                # 【修改】加上基金名称的打印
                tqdm.write(f"⚠️ 跳过基金 {code} ({fund_name}) 的净值 (无数据/解析异常)")

        # --- 2. 获取重仓股票 ---
        holdings_file = f"fund_data/holdings/{code}_holdings_{year}.csv"
        if not os.path.exists(holdings_file):
            requested_this_loop = True
            try:
                holdings_df = ak.fund_portfolio_hold_em(symbol=code, date=year)
                if not holdings_df.empty:
                    holdings_df.to_csv(holdings_file, index=False, encoding='utf-8-sig')
                else:
                    open(holdings_file, 'w').close()  # 空数据建空文件占位
            except Exception as e:
                open(holdings_file, 'w').close()  # 异常建空文件占位
                # 【修改】加上基金名称的打印
                tqdm.write(f"⚠️ 跳过基金 {code} ({fund_name}) 的重仓股 (无持仓/解析异常)")

        # # 休眠防封 IP (1秒)
        # # 核心优化：只有真发起网络请求才睡1秒。若是已有文件直接跳过，1秒钟能直接跨过上千个进度！
        # if requested_this_loop:
        #     time.sleep(1)


def calculate_adj_nav(df):
    """
    根据包含“单位净值”和“日增长率”的数据计算完整的复权净值。
    """
    res_df = df.copy()

    if res_df.empty:
        return res_df

    # 1. 容错处理：确保“日增长率”是浮点数，且将缺失值（NaN）或异常字符替换为 0
    res_df['日增长率'] = pd.to_numeric(res_df['日增长率'], errors='coerce').fillna(0)

    # 2. 核心数学逻辑：计算累乘的复权因子
    # 接口的数据如果是 1.5 代表 1.5%，所以要除以 100
    res_df['复权因子'] = (1 + res_df['日增长率'] / 100).cumprod()

    # 3. 对齐起点：用第一天的真实单位净值乘以复权因子，得出每天的复权净值
    # 注意：为了严谨，确保取到的第一天是非空的净值
    first_day_nav = res_df['单位净值'].dropna().iloc[0]
    res_df['复权净值'] = res_df['复权因子'] * first_day_nav

    return res_df


def calculate_fund_stats(df, fund_name="Unknown", date_col='净值日期', nav_col='单位净值'):
    """
    将死刑判断改为数据统计，只记录基本情况放入 result 中。
    """
    result = {
        "fund_name": fund_name,
        "adj_nav_file": None,
        "total_active_days": 0,
        "real_start": None,
        "real_end": None,
        "missing_ratio": 0.0,
        "max_zeros": 0,
        "max_drawdown": 0.0,
        "annualized_return": 0.0,
        "top_10_holdings": ""
    }

    if df is None or df.empty or nav_col not in df.columns or date_col not in df.columns:
        return result

    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    temp_df = temp_df.sort_values(by=date_col)
    temp_df = temp_df.drop_duplicates(subset=[date_col], keep='last')
    temp_df = temp_df.set_index(date_col)

    nav_series = temp_df[nav_col]

    # 剥离前后的全空数据，找到该基金真实的生命周期
    valid_series = nav_series.dropna()
    if valid_series.empty:
        return result

    real_start = valid_series.index[0]
    real_end = valid_series.index[-1]
    active_series = nav_series.loc[real_start:real_end]
    total_active_days = len(active_series)

    result["total_active_days"] = total_active_days
    result["real_start"] = real_start.strftime('%Y-%m-%d')
    result["real_end"] = real_end.strftime('%Y-%m-%d')

    if total_active_days > 0:
        result["missing_ratio"] = float(active_series.isna().sum() / total_active_days)

    # 计算连续 0 收益（死水天数）
    if '日增长率' in temp_df.columns:
        rets = pd.to_numeric(temp_df.loc[real_start:real_end, '日增长率'], errors='coerce').fillna(0)
    else:
        nav_ffilled = active_series.ffill()
        rets = nav_ffilled.pct_change().fillna(0.0)

    is_zero = (rets.abs() < 1e-8).astype(int)
    max_zeros = is_zero.groupby((is_zero == 0).cumsum()).sum().max()
    result["max_zeros"] = int(max_zeros) if pd.notna(max_zeros) else 0

    return result


def process_fund_pipeline(file_path, save_qualified=True, date_col='净值日期', nav_col='单位净值'):
    """
    基金数据处理流水线：提取统计指标 -> 提取前十大持仓 -> 无差别计算并保存复权净值
    """
    fund_name = os.path.basename(file_path).split('_')[0]

    # 1. 读取数据
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        df = pd.DataFrame()

    # 2. 基础指标计算 (不再做死刑淘汰)
    result = calculate_fund_stats(df, fund_name=fund_name, date_col=date_col, nav_col=nav_col)

    if not df.empty and date_col in df.columns and nav_col in df.columns:
        # 3. 排序并计算复权净值
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).reset_index(drop=True)
        qualified_df = calculate_adj_nav(df)

        # 4. 基于复权净值计算衍生指标 (最大回撤、年化收益)
        if '复权净值' in qualified_df.columns:
            adj_series = qualified_df.set_index(date_col)['复权净值'].dropna()
            if not adj_series.empty and result["total_active_days"] > 0:
                start_val = adj_series.iloc[0]
                end_val = adj_series.iloc[-1]
                if start_val > 0:
                    years = result["total_active_days"] / 252.0
                    if years > 0:
                        result["annualized_return"] = round(float((end_val / start_val) ** (1 / years) - 1), 6)

                roll_max = adj_series.cummax()
                if not roll_max.empty and roll_max.max() > 0:
                    drawdown = (adj_series - roll_max) / roll_max
                    result["max_drawdown"] = round(float(drawdown.min()), 6)

        # 5. 无差别保存复权结果
        if save_qualified:
            saved_path = file_path.replace('.csv', '_adj.csv')
            qualified_df.to_csv(saved_path, index=False, encoding='utf-8-sig')
            result["adj_nav_file"] = saved_path

    # 6. 获取前十大持仓股票
    holdings_pattern = f"fund_data/holdings/{fund_name}_holdings_*.csv"
    holdings_files = glob.glob(holdings_pattern)
    if holdings_files:
        holdings_files.sort(reverse=True)  # 优先读取最新年份的文件
        try:
            hdf = pd.read_csv(holdings_files[0])
            if '股票名称' in hdf.columns:
                top_10 = hdf.head(10)['股票名称'].dropna().astype(str).tolist()
                result["top_10_holdings"] = ",".join(top_10)
        except Exception:
            pass

    return result


def judge_fund_df():
    target_dir = 'fund_data/nav'

    print(f"🚀 开始扫描目录: {target_dir}")
    if not os.path.exists(target_dir):
        print("❌ 错误: 目录不存在，请检查路径。")
        return

    # 获取所有 .csv 文件
    search_pattern = os.path.join(target_dir, "*.csv")
    all_csv_files = glob.glob(search_pattern)

    if not all_csv_files:
        print("⚠️ 未在该目录下找到任何 CSV 文件。")
        return

    print(f"📂 共发现 {len(all_csv_files)} 个 CSV 文件，开始计算统计指标并生成复权净值...\n" + "-" * 50)

    # 过滤出尚未处理过的原文件
    files_to_process = [f for f in all_csv_files if not os.path.basename(f).endswith("_adj.csv")]
    skipped_count = len(all_csv_files) - len(files_to_process)

    stats = {"processed": 0, "skipped": skipped_count, "error": 0}
    all_results = []

    if files_to_process:
        # 使用进程池执行并行处理，并行度设置为 20
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            # 提交任务
            futures = [executor.submit(process_fund_pipeline, file_path, True) for file_path in files_to_process]

            # 使用 tqdm 进度条结合 concurrent.futures.as_completed 获取结果
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理进度"):
                try:
                    result = future.result()
                    all_results.append(result)

                    if result.get("adj_nav_file"):
                        stats["processed"] += 1
                    else:
                        stats["error"] += 1
                except Exception as e:
                    stats["error"] += 1

    # 统一将 result 汇总为 CSV 文件
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_csv = "fund_data/all_funds_result.csv"
        results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ 所有基金结果汇总完毕，已保存至: {output_csv}")

    # 打印最终总结报表
    print("-" * 50)
    print("📊 批量处理完成！总结报告:")
    print(f"   总文件数 : {len(all_csv_files)}")
    print(f"   成功处理 : {stats['processed']}只基金 (生成了 _adj.csv 并记录统计特征)")
    print(f"   跳过处理 : {stats['skipped']}个已存在的复权文件")
    print(f"   空数/报错 : {stats['error']}个文件")
    print("-" * 50)


if __name__ == "__main__":

    df_file = r'fund_data/fof_evaluation_results_3d_pool263.csv'
    df = pd.read_csv(df_file, engine='python')
    # 只保留Total_Score大于0的行
    df_filtered = df[df['Total_Score'] > 0].copy()
    # 按照Total_Score列降序排序
    df_filtered = df_filtered.sort_values(by='Total_Score', ascending=False)



    judge_fund_df()
    #
    # # 1. 创建存放数据的目录
    # create_folders()
    #
    # # 2. 获取经过清洗过滤的活跃基金代码
    # active_codes = get_active_fund_codes()
    #
    # # 3. 开始下载数据
    # # 【重要说明】
    # # 先保持 test_mode=True 运行一次，确认能跑通。
    # # 确认没问题后，把下面这行的 test_mode 改为 False，就会去下载过滤后的大约 1万多只 基金了。
    # fetch_and_save_fund_data(active_codes, year="2026", test_mode=False)
    #
    # print("\n🎉 全部任务执行完毕！请查看 fund_data 文件夹。")