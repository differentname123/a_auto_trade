import glob

import akshare as ak
import pandas as pd
import time
import os
from tqdm import tqdm
import pandas as pd


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
        fund_name = fund_codes[code] # 【新增】获取当前基金名称

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
                open(nav_file, 'w').close()      # 异常时也建空文件占位，防下次重复请求
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
                    open(holdings_file, 'w').close() # 空数据建空文件占位
            except Exception as e:
                open(holdings_file, 'w').close()     # 异常建空文件占位
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


def strict_veto_single_fund(df, fund_name="Unknown", date_col='净值日期', nav_col='单位净值',
                            baseline_date=None):
    """
    FOF 绝对死刑过滤器 (100% 死亡判定)
    此时传入的还是未复权的原始 df，主要检查数据完整性与死水状态。
    """
    result = {
        "fund_name": fund_name,
        "is_dead": False,
        "death_reason": ""
    }

    # ================= 1. 数据结构直接报废 =================
    if df is None or df.empty or nav_col not in df.columns or date_col not in df.columns:
        result["is_dead"] = True
        result["death_reason"] = "Invalid DataFrame structure or missing columns"
        return result

    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    temp_df = temp_df.set_index(date_col).sort_index()
    temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
    nav_series = temp_df[nav_col]

    # 剥离前后的全空数据，找到该基金真实的生命周期
    valid_series = nav_series.dropna()
    if valid_series.empty:
        result["is_dead"] = True
        result["death_reason"] = "All NAV data is NaN"
        return result

    real_start = valid_series.index[0]
    real_end = valid_series.index[-1]
    active_series = nav_series.loc[real_start:real_end]
    total_active_days = len(active_series)

    # ================= 2. 长度绝对死刑 =================
    if total_active_days < 252:
        result["is_dead"] = True
        result["death_reason"] = f"Absolute short history: {total_active_days} days < 252 days"
        return result

    # ================= 3. 僵尸基金绝对死刑 =================
    if baseline_date is None:
        baseline_date = pd.Timestamp.today()
    else:
        baseline_date = pd.to_datetime(baseline_date)

    if (baseline_date - real_end).days > 35:
        result["is_dead"] = True
        result["death_reason"] = f"Zombie fund: Last valid date {real_end.strftime('%Y-%m-%d')}"
        return result

    # ================= 4. 缺失率绝对死刑 =================
    missing_ratio = active_series.isna().sum() / total_active_days
    if missing_ratio > 0.05:
        result["is_dead"] = True
        result["death_reason"] = f"Fatal missing data: {missing_ratio:.2%} > 5%"
        return result

    # ================= 5. 流动性枯竭/伪造平滑绝对死刑 =================
    # 核心优化：如果你传入的数据带有“日增长率”，直接用它来判定死水，避开分红造成的净值跳动误判
    if '日增长率' in temp_df.columns:
        rets = pd.to_numeric(temp_df.loc[real_start:real_end, '日增长率'], errors='coerce').fillna(0)
    else:
        nav_ffilled = active_series.ffill()
        rets = nav_ffilled.pct_change().fillna(0.0)

    # 抓捕连续0收益（误差容忍度 1e-8）
    is_zero = (rets.abs() < 1e-8).astype(int)
    max_zeros = is_zero.groupby((is_zero == 0).cumsum()).sum().max()

    if max_zeros > 10:
        result["is_dead"] = True
        result["death_reason"] = f"Stagnant pricing: {max_zeros} consecutive 0-return days"
        return result

    return result


def process_fund_pipeline(file_path, save_qualified=True, date_col='净值日期', nav_col='单位净值'):
    """
    基金数据处理流水线：读取文件 -> 初筛死刑判定 -> (若存活)计算复权净值 -> (可选)保存为新文件
    """
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File not found: {file_path}", "df": None}

    # 提取文件名作为基金标识
    fund_name = os.path.basename(file_path).split('.')[0]

    # 1. 读取数据
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to read CSV: {e}", "df": None}

    # 2. 执行死刑判定 (初筛)
    veto_report = strict_veto_single_fund(df, fund_name=fund_name, date_col=date_col, nav_col=nav_col)

    # 如果被判死刑，直接打回，不计算复权，也不保存
    if veto_report["is_dead"]:
        return {
            "status": "rejected",
            "message": veto_report["death_reason"],
            "df": None
        }

    # 3. 存活标的：按照日期排好序后，计算复权净值
    # 为保证 calculate_adj_nav 中 .iloc[0] 拿到的是最早期的数据，先正向排序
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)

    qualified_df = calculate_adj_nav(df)

    # 4. 可选：保存通过且加工完的数据
    saved_path = None
    if save_qualified:
        # 直接替换 .csv，加上后缀标识这是一个已清洗且包含复权净值的文件
        saved_path = file_path.replace('.csv', '_adj.csv')
        qualified_df.to_csv(saved_path, index=False)

    return {
        "status": "passed",
        "message": "Qualified and adjusted NAV calculated.",
        "df": qualified_df,
        "saved_path": saved_path
    }


def judge_fund_df():
    # 使用 raw string (r"") 防止 Windows 路径下的转义字符报错
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

    print(f"📂 共发现 {len(all_csv_files)} 个 CSV 文件，开始清洗过滤...\n" + "-" * 50)

    # 统计数据
    stats = {"passed": 0, "rejected": 0, "skipped": 0, "error": 0}

    for file_path in all_csv_files:
        filename = os.path.basename(file_path)

        # 排除已经处理过的 _adj.csv 文件
        if filename.endswith("_adj.csv"):
            stats["skipped"] += 1
            continue

        fund_name = filename.split('.')[0]

        # 执行流水线
        result = process_fund_pipeline(file_path, save_qualified=True)

        # 终端输出进度和结果
        if result["status"] == "passed":
            print(f"[ ✅ 准入 ] {fund_name.ljust(15)} -> {result['message']}")
            stats["passed"] += 1
        elif result["status"] == "rejected":
            print(f"[ ❌ 淘汰 ] {fund_name.ljust(15)} -> 原因: {result['message']}")
            stats["rejected"] += 1
        else:
            print(f"[ ⚠️ 报错 ] {fund_name.ljust(15)} -> 详情: {result['message']}")
            stats["error"] += 1

    # 打印最终总结报表
    print("-" * 50)
    print("📊 批量处理完成！总结报告:")
    print(f"   总文件数 : {len(all_csv_files)}")
    print(f"   成功生成 : {stats['passed']}只基金进入候选池 (生成了 _adj.csv)")
    print(f"   淘汰报废 : {stats['rejected']}只基金")
    print(f"   跳过处理 : {stats['skipped']}个已存在的复权文件")
    print(f"   处理报错 : {stats['error']}个文件")
    print("-" * 50)

if __name__ == "__main__":
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