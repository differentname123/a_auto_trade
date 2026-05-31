import glob
import re
from datetime import datetime

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


import pandas as pd
import re


def _get_active_bond_mask(name_series):
    """
    【内部提取函数】获取主动债的剔除掩码 (基于名称的语义识别)
    核心逻辑：有嫌疑词 且 无护身符 = 纯粹的主动债券
    """
    # 1. 嫌疑词：这些词通常代表基金经理在主动挑债券或做波段
    suspect_words = [
        '纯债', '短债', '信用债', '收益债', '回报',
        '双季', '季季', '月月', '添利', '增利', '稳利'
    ]

    # 2. 护身符：如果名字里带这些词，说明它是严格跟踪指数的被动债，不杀
    shield_words = [
        '指数', '中债', '彭博', '上清所', '国开', '政金', '农发'
    ]

    # 将列表转换为正则的或逻辑 (例如：'纯债|短债|信用债')
    suspect_pattern = '|'.join(suspect_words)
    shield_pattern = '|'.join(shield_words)

    # 向量化正则匹配：识别嫌疑与护身符 (na=False 应对缺失值报错)
    is_suspect = name_series.str.contains(suspect_pattern, regex=True, na=False)
    has_shield = name_series.str.contains(shield_pattern, regex=True, na=False)

    # 核心狙击逻辑：有嫌疑 且 没被护盾保护
    kill_mask = is_suspect & (~has_shield)

    return kill_mask


def filter_fund_universe(df, type_col='基金类型', name_col='基金简称', remove_c_class=True):
    """
    清洗过滤基金池流水线：
    1. 按大类剔除指定类型的基金（混合、假纯债、股票等）
    2. 按名称语义精准狙击“主动型债券基金”
    3. 剔除同一只基金的冗余份额（保留A份额，剔除C/E等份额）
    """
    if df is None or df.empty:
        print("⚠️ 传入的数据为空，跳过清洗。")
        return df

    res_df = df.copy()
    total_before = len(res_df)
    clean_stats = {}

    # ==========================================
    # 模块一：基于【基金类型】的常规大类剔除
    # ==========================================
    type_filter_rules = {
        "1. 混合型 (仓位飘忽污染相关性)": [
            '混合型-灵活', '混合型-偏股', '混合型-偏债', '混合型-平衡', '混合型-绝对收益',
            'QDII-混合偏股', 'QDII-混合债', 'QDII-混合灵活', 'QDII-混合平衡'
        ],
        "2. 假纯债 (含股票仓位易暴雷)": [
            '债券型-混合二级', '债券型-混合一级'
        ],
        "3. 纯主动股票型 (回测噪音太大)": [
            '股票型', 'QDII-普通股票'
        ],
        "4. 货币类 (拉低收益导致资金站岗)": [
            '货币型-普通货币', '货币型-浮动净值'
        ],
        "5. FOF类 (避免资产配置套娃)": [
            'FOF-稳健型', 'FOF-进取型', 'FOF-均衡型', 'QDII-FOF'
        ]
    }

    if type_col in res_df.columns:
        res_df[type_col] = res_df[type_col].fillna('').str.strip()
        for reason, types in type_filter_rules.items():
            mask = res_df[type_col].isin(types)
            hit_count = mask.sum()
            if hit_count > 0:
                clean_stats[reason] = hit_count
            res_df = res_df[~mask]

    # ==========================================
    # 模块二：基于【基金名称】精准狙击主动债券
    # ==========================================
    if name_col in res_df.columns:
        active_bond_mask = _get_active_bond_mask(res_df[name_col])
        hit_count = active_bond_mask.sum()

        if hit_count > 0:
            clean_stats["7. 主动型债券 (防范信用/久期暴露)"] = hit_count

        # 剔除命中目标
        res_df = res_df[~active_bond_mask]

    # ==========================================
    # 模块三：降维打击，剔除 A/C 冗余份额
    # ==========================================
    if remove_c_class and name_col in res_df.columns:
        before_ac_clean = len(res_df)

        # 提取基础名字并排序去重 (确保 A 排在 C/E 前面)
        res_df['base_name'] = res_df[name_col].str.replace(r'[ABCDEHIY]$', '', regex=True)
        res_df = res_df.sort_values(by=name_col)
        res_df = res_df.drop_duplicates(subset=['base_name'], keep='first')
        res_df = res_df.drop(columns=['base_name'])

        ac_clean_count = before_ac_clean - len(res_df)
        if ac_clean_count > 0:
            clean_stats["8. 冗余份额去重 (剔除C/E等)"] = ac_clean_count

    # ==========================================
    # 模块四：生成高颜值归因报告
    # ==========================================
    total_after = len(res_df)
    total_cleaned = total_before - total_after

    print("\n" + "=" * 55)
    print("📊 基金池底仓清洗过滤报告")
    print("=" * 55)
    print(f"🔹 清洗前总数 : {total_before:,} 只")
    print(f"✅ 最终保留数 : {total_after:,} 只")
    print(f"❌ 总计剔除数 : {total_cleaned:,} 只")

    if total_cleaned > 0:
        print("-" * 55)
        print("[详细剔除归因及占比]")
        # 按剔除数量从大到小排列输出
        sorted_stats = sorted(clean_stats.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_stats:
            ratio = count / total_cleaned
            print(f"  {reason:<26} : {count:>5,} 只 ({ratio:>6.2%})")
    print("=" * 55 + "\n")

    return res_df.reset_index(drop=True)


def get_active_fund_codes():
    """
    获取当前存续的活跃基金，并过滤掉无需查询重仓股的基金类型（如货币、理财、债券），
    同时过滤掉当前处于【暂停申购】或【封闭期】的不可购买基金。
    """
    # 缓存检查：若有现成列表且包含名称，则直接读取
    cache_file = 'temp/active_fund_codes_new.csv'
    if os.path.exists(cache_file):
        df_cache = pd.read_csv(cache_file, dtype=str)
        filter_df = filter_fund_universe(df_cache)
        filter_df.to_csv(cache_file, index=False, encoding='utf-8-sig')  # 更新缓存文件，剔除不合规基金
        # 获取filter_df所有的 基金简称列表 并且去重
        code_name_list = filter_df['基金简称'].dropna().unique().tolist()
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

    # # 第二步过滤：剔除货币、债券、理财等没有重仓股票的基金
    # print("3. 正在剔除货币型、债券型等无重仓股票的基金...")
    # exclude_keywords = '货币|理财|纯债|债券|保本'
    # df_filtered = df_filtered[~df_filtered['基金类型'].str.contains(exclude_keywords, na=False)]

    # 提取基金代码和名称的字典映射
    fund_dict = dict(zip(df_filtered['基金代码'], df_filtered['基金简称']))
    print(f"✅ 最终过滤完毕！当前可申购且可能含有股票持仓的基金共计 {len(fund_dict)} 只。")

    # 将包含名称的列表存入 temp，供断点续传使用
    df_save = pd.DataFrame({
        '基金代码': list(fund_dict.keys()),
        '基金简称': list(fund_dict.values()),
        '基金类型': df_filtered['基金类型'].tolist()
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


def load_and_merge_parquet_by_dim(dimension, data_dir='fund_data', min_days=600, min_score=None, filter_code=None):
    """
    根据给定的维度，自动搜索目录下所有匹配的 Parquet 文件，
    利用 PyArrow 谓词下推（直接从硬盘过滤）极速加载并合并。

    :param dimension: 组合维度 (例如 2, 3, 4)
    :param data_dir: 数据所在的文件夹路径
    :param min_days: 过滤条件：Total_Days > min_days
    :param min_score: 过滤条件：Total_Score > min_score (可选)
    :return: 过滤并合并后按分数排序的 DataFrame
    """
    # 1. 动态生成正则匹配模式，搜索指定维度的所有文件
    search_pattern = os.path.join(data_dir, f'fof_evaluation_results_{dimension}d_*.parquet')
    file_list = glob.glob(search_pattern)

    if not file_list:
        print(f"⚠️ 未找到 {dimension} 维的任何 Parquet 文件 (搜索路径: {search_pattern})")
        return pd.DataFrame()

    print(f"🔍 找到 {len(file_list)} 个 {dimension} 维历史文件，正在启动底层 C++ 并发下推加载...")

    # 2. 组装 PyArrow 底层过滤机制（谓词下推）
    read_filters = [('Total_Days', '>', min_days)]
    if min_score is not None:
        read_filters.append(('Total_Score', '>', min_score))

    # 3. 极速读取与合并
    try:
        df = pd.read_parquet(file_list, engine='pyarrow', filters=read_filters)
        if filter_code is not None:
            df = df[df['组合文件名'].str.contains(filter_code, na=False)]
    except Exception as e:
        print(f"❌ 批量并发加载遇到异常 (可能是个别文件损坏或字段不一致): {e}")
        print("🔄 触发自动降级机制，转为逐个文件安全加载模式...")
        dfs = []
        for file in file_list:
            try:
                df_part = pd.read_parquet(file, engine='pyarrow', filters=read_filters)
                if not df_part.empty:
                    dfs.append(df_part)
            except Exception as ex:
                print(f"  -> ⚠️ 跳过异常文件 [{os.path.basename(file)}]: {ex}")

        if not dfs:
            print(f"⚠️ {dimension} 维文件全部读取失败或无符合条件的数据。")
            return pd.DataFrame()

        # 安全模式合并
        df = pd.concat(dfs, ignore_index=True)

    # 如果过滤后一条数据都没有了
    if df.empty:
        print(f"ℹ️ {dimension} 维数据加载完成，但经过硬盘级过滤后，没有保留下任何组合。")
        return df

    # 4. 排序逻辑保持不变
    df = df.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)

    print(f"✅ 成功合并 {dimension} 维数据! 经过硬盘级过滤，最终留存: {len(df):,} 个优质组合。")
    return df


import pandas as pd


def get_blacklisted_fund_codes(df):
    """
    根据给定的主动基金黑名单检索 DataFrame。
    返回【命中黑名单】的 6 位基金代码列表，并打印详细日志（包括未匹配的黑名单项）。
    """
    active_funds_to_exclude = [
        "华夏鼎航债券", "兴业裕丰债券", "鹏华丰惠债券", "西部利得汇盈债券",
        "广发景宁债券", "华安安业债券", "中信保诚景丰", "国金惠安利率债",
        "南方交元债券", "华安鼎丰债券", "华宝宝泓债券", "兴全恒裕债券",
        "德邦锐乾债券", "中信保诚稳达", "兴全稳泰债券", "德邦锐裕利率债债券",
        "天弘信益债券", "中信保诚稳丰", "鹏华丰禄债券", "西部利得祥逸债券",
        "永赢裕益债券", "南方旭元债券", "兴业福鑫债券", "永赢邦利债券",
        "前海开源鼎欣债券",
        "中信保诚稳泰债券",
        "永赢昌利债券",
        "诺德安鸿",
        "鹏华金利债券",
        "华夏鼎通债券",
        "华夏鼎隆债券",
        "鹏扬淳开债券",
        "易方达中短期美元债",  # 主动型QDII债券基金，人为挑选海外债券
        "中银汇享债券",
        "永赢伟益债券",
        "中信保诚稳益",
        "建信利率债债券",  # 虽买利率债，但属于主动调控久期的纯债基金，非指数
        "鹏华丰鑫债券",
        "宏利永利债券",
        "鹏华0-5年利率债券",  # 带有0-5年期限，但通常此类未标明“指数”的为主动纯债
        "中信保诚稳鸿",
        "华夏鼎康债券",
        "南方泽元债券",
        "永赢瑞益债券",
        "永赢惠益债券",
        "兴业裕恒债券",
        "鹏华丰腾债券"

    ]

    original_size = len(df)
    df_work = df.copy()

    # 1. 补齐 6 位代码处理 (兼容浮点数读取的情况)
    df_work['基金代码'] = df_work['基金代码'].astype(str).str.split('.').str[0].str.zfill(6)

    # 2. 核心追踪与匹配逻辑
    matched_names = set()  # 用来记录成功匹配到的黑名单名称
    mask_is_blacklisted = pd.Series(False, index=df_work.index)  # 初始标记为全 False

    for bad_name in active_funds_to_exclude:
        # 使用纯字符串包含判断
        current_match = df_work['基金简称'].astype(str).str.contains(bad_name, regex=False, na=False)

        # 如果这个黑名单名字在数据中至少出现了一次
        if current_match.any():
            matched_names.add(bad_name)
            # 记录命中的行 (这里是累加逻辑，命中任意一个就算)
            mask_is_blacklisted = mask_is_blacklisted | current_match

            # 3. 找出未匹配到的黑名单项
    unmatched_names = [name for name in active_funds_to_exclude if name not in matched_names]

    # 4. 获取命中黑名单的行，并提取 6 位代码 (注意这里没有使用 ~ 取反)
    df_blacklisted = df_work[mask_is_blacklisted]
    blacklisted_codes = df_blacklisted['基金代码'].unique().tolist()

    # 5. 打印详细日志
    print("====== 黑名单提取日志 ======")
    print(f"黑名单检索词数量 : {len(active_funds_to_exclude)}")
    print(f"输入数据总行数   : {original_size}")
    print(f"命中黑名单行数   : {mask_is_blacklisted.sum()}")
    print(f"返回黑名单Code数 : {len(blacklisted_codes)}")

    print("-" * 24)
    if unmatched_names:
        print(f"⚠️ 以下 {len(unmatched_names)} 个黑名单词在数据中未匹配到:")
        for name in unmatched_names:
            print(f"  - {name}")
    else:
        print("✅ 所有黑名单词均已在数据中成功匹配。")
    print("==========================")

    return blacklisted_codes


if __name__ == "__main__":
    # 1. 读取报告文件
    report_df = pd.read_csv('fund_data/base_pool_rejection_reasons.csv')
    active_code_df = pd.read_csv('temp/active_fund_codes_new.csv')
    # 获取 active_code_df 的 基金简称列表
    name_list = active_code_df['基金简称'].dropna().unique().tolist()
    # 只保留 基金简称 包含 增强 的行
    filter_active_code_df = active_code_df[active_code_df['基金简称'].str.contains('增强', na=False)]

    # 按照 基金代码 进行合并
    report_df = report_df.merge(active_code_df, left_on='基金代码', right_on='基金代码', how='right')
    # black_code_list = get_blacklisted_fund_codes(report_df)
    all_codes = report_df['文件'].str.extract(r'(\d{6})')[0].dropna().unique().tolist()
    valid_code_list = [71, 179, 369, 596, 614, 826, 1092, 2236, 2423, 2659, 3081, 3520, 4532, 4597, 4854, 5561, 6098, 6286, 6341, 6473, 6481, 6486, 6748, 6809, 6932, 6959, 6961, 7094, 7252, 7300, 7390, 7431, 7464, 7505, 7593, 7751, 7765, 7788, 7809, 7910, 8040, 8163, 8189, 8256, 8279, 8326, 8340, 8399, 8482, 8574, 8583, 8626, 8707, 8713, 8928, 8956, 9033, 9051, 9067, 9219, 9225, 9421, 9560, 9615, 9625, 9721, 9742, 10497, 118002, 160222, 160626, 161628, 165519, 165520, 165521, 167506, 202021, 270042, 320013, 501310, 519981, 539001]


    # 将valid_code_list 补充为6位字符串，并且前面补0
    valid_code_list = [str(code).zfill(6) for code in valid_code_list]

    invalid_code_list = list(set(all_codes) - set(valid_code_list))
    # invalid_code_list.extend(black_code_list)

    # 3. 加载并合并数据
    all_df_list = []
    for i in range(4):
        df_2d = load_and_merge_parquet_by_dim(dimension=i + 2, min_days=1250, min_score=0)
        df_2d['Total_Score'] = df_2d['Total_Score'] * 10000
        # 只保留CAGR大于0.1的组合
        df_2d = df_2d[df_2d['CAGR'] > 0.1].reset_index(drop=True)

        # df_2d['score'] = df_2d['CAGR']  * df_2d['Total_Score'] * 10
        # # df_2d 按照score降序排序
        # df_2d = df_2d.sort_values(by='score', ascending=False).reset_index(drop=True)

        # df_d = load_and_merge_parquet_by_dim(dimension=5, min_days=1250, min_score=1.1)
        # df_d['score'] = df_d['CAGR'] * df_d['Total_Score'] * 10
        # # df_d 按照score降序排序
        # df_d = df_d.sort_values(by='score', ascending=False).reset_index(drop=True)
        # filtered_df1 = df_d[
        #     df_d['组合文件名'].str.contains('006961') & df_d['组合文件名'].str.contains('008326') & df_d[
        #         '组合文件名'].str.contains('009033') & df_d['组合文件名'].str.contains('519212')]

        all_df_list.append(df_2d)

    all_df = pd.concat(all_df_list, ignore_index=True)
    # 4. 排序
    all_df = all_df.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)
    print(
        f"合并完成，原始数据总量: {len(all_df):,} 条记录。正在进行代码过滤...当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    if invalid_code_list:
        # 1. 将无效列表转换为集合 (Set)，查找速度提升至 O(1)
        invalid_set = set(invalid_code_list)


        def is_valid_row(filename):
            if pd.isna(filename):
                return False
            # 提取该文件名里的所有6位代码
            codes_in_file = re.findall(r'\d{6}', str(filename))
            # 判断是否有重叠
            return invalid_set.isdisjoint(codes_in_file)


        # 3. 使用 Python 原生的列表推导式生成掩码 (比 pandas 的 .apply 还要快)
        mask = [is_valid_row(fname) for fname in all_df['组合文件名']]

        # 4. 过滤数据
        all_df_filter = all_df[mask].reset_index(drop=True)
    else:
        all_df_filter = all_df.copy()

    print(
        f"过滤前数据量: {len(all_df)}, 过滤后数据量: {len(all_df_filter)} (共过滤掉 {len(all_df) - len(all_df_filter):,} 条记录)，耗时: {time.time() - start_time:.2f} 秒")
    all_df_filter['score'] = all_df_filter['CAGR'] * 10 * all_df_filter['Total_Score']
    # df_d 按照score降序排序
    all_df_filter = all_df_filter.sort_values(by='score', ascending=False).reset_index(drop=True)

    all_df_filter['score1'] = all_df_filter['Calmar_Ratio'] - all_df_filter['Calmar_Baseline']


    # ================= 以下为新增与修改的代码 =================

    # 定义基金名称简化函数
    def simplify_fund_name(name):
        if not isinstance(name, str):
            return name
        # 1. 去除括号及其内部内容 (支持中英文括号)
        name = re.sub(r'\(.*?\)|（.*?）', '', name)
        # 2. 去除常见的冗余词汇 (较长的词放前面优先匹配)
        redundant_words = [
            'ETF联接', '联接', 'ETF', 'LOF', 'QDII',
            '灵活配置', '多策略', '混合型', '混合',
            '指数增强', '增强型', '指数', '发起式',
            '人民币份额', '人民币', '证券投资基金'
        ]
        pattern = '|'.join(redundant_words)
        name = re.sub(pattern, '', name)
        # 3. 去除末尾的份额字母 (如 A, C, E)
        name = re.sub(r'[A-Z]$', '', name)
        return name


    # 1. 将 active_code_df 中的基金代码转为字符串，并补齐前导0为6位
    padded_active_codes = active_code_df['基金代码'].astype(str).str.zfill(6)

    # 2. 对原有的基金简称进行简化处理
    simplified_names = active_code_df['基金简称'].apply(simplify_fund_name)

    # 3. 构建 {基金代码: 简化后的基金简称} 的哈希字典映射
    code_to_name_dict = dict(zip(padded_active_codes, simplified_names))


    # 4. 定义解析与替换函数
    def map_code_combo_to_name_combo(combo_str):
        if pd.isna(combo_str):
            return combo_str
        # 按下划线拆分代码组合
        codes = str(combo_str).split('_')
        # 查字典进行替换，如果字典中不存在对应简称，则保留原代码
        names = [str(code_to_name_dict.get(code, code)) for code in codes]
        # 将简称重新组合拼接
        return '_'.join(names)


    # 5. 应用函数，生成新增的 "基金简称组合" 列
    all_df_filter['基金简称组合'] = all_df_filter['组合文件名'].apply(map_code_combo_to_name_combo)

    # 6. 将 "基金简称组合" 列挪动到第一列
    col_to_move = all_df_filter.pop('基金简称组合')
    all_df_filter.insert(0, '基金简称组合', col_to_move)
    # 将 组合文件名 Start_Date End_Date Total_Days 放到最后几列
    cols = list(all_df_filter.columns)
    key_list = ['组合文件名', 'Start_Date', 'End_Date', 'Total_Days']
    for key in key_list:
        if key in cols:
            cols.remove(key)
            cols.append(key)
    all_df_filter = all_df_filter[cols]

    df_d = load_and_merge_parquet_by_dim(dimension=5, min_days=600, min_score=0)
    df_d['score'] = df_d['CAGR'] * df_d['Total_Score'] * 10

    filtered_df1 = df_d[df_d['组合文件名'].str.contains('000390') & df_d['组合文件名'].str.contains('009033') & df_d[
        '组合文件名'].str.contains('160323') & df_d['组合文件名'].str.contains('539001')]

    # 打印出df_2d中组合文件名 包含006373 并且也包含006372
    # 获取df_2d中组合文件名不重复的列表
    unique_combinations = df_2d['组合文件名'].unique()
    # 将每个元素按照 '_' 分割，并且加入列表,并且去重
    unique_funds = set()
    for combo in unique_combinations:
        parts = combo.split('_')
        unique_funds.update(parts)

    # 过滤得到report_df中 adj_nav_file 列包含 unique_funds中任意一个元素的行
    report_df['keep'] = report_df['adj_nav_file'].apply(
        lambda x: any(fund in x for fund in unique_funds)
        if isinstance(x, str) else False
    )

    # 1. 后缀名修改为 .parquet
    df_file = r'fund_data/fof_evaluation_results_2d_pool1084_min_day_1260.parquet'

    # 2. 利用 PyArrow 的底层过滤机制（谓词下推）
    # 引擎会在硬盘读取阶段直接丢弃不符合条件的数据，绝不让垃圾数据弄脏内存
    read_filters = [
        # ('Total_Score', '>', 0.01),
        ('Total_Days', '>', 600)
    ]

    # 3. 读取并直接完成过滤
    df = pd.read_parquet(df_file, engine='pyarrow', filters=read_filters)

    # 4. 排序逻辑保持不变
    df = df.sort_values(by='Total_Score', ascending=False)

    judge_fund_df()
    #
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
    #
    # judge_fund_df()
