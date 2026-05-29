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

if __name__ == "__main__":
    # 1. 读取报告文件
    report_df = pd.read_csv('fund_data/filtering_reasons_report.csv')

    all_codes = report_df['adj_nav_file'].str.extract(r'(\d{6})')[0].dropna().unique().tolist()
    valid_code_list = ['000011', '000029', '000065', '000110', '000121', '000124', '000127', '000136', '000165', '000167', '000179', '000214', '000251', '000279', '000308', '000309', '000369', '000390', '000432', '000433', '000523', '000531', '000557', '000567', '000573', '000574', '000587', '000591', '000619', '000634', '000649', '000652', '000742', '000743', '000746', '000794', '000828', '000928', '000963', '000969', '000991', '001060', '001103', '001115', '001136', '001140', '001144', '001173', '001182', '001198', '001216', '001223', '001224', '001229', '001250', '001267', '001279', '001325', '001359', '001365', '001380', '001398', '001411', '001412', '001447', '001448', '001479', '001487', '001488', '001496', '001500', '001534', '001543', '001562', '001569', '001579', '001604', '001605', '001607', '001628', '001643', '001651', '001656', '001657', '001678', '001682', '001703', '001707', '001709', '001716', '001718', '001723', '001732', '001753', '001759', '001799', '001801', '001803', '001809', '001816', '001825', '001830', '001832', '001850', '001852', '001864', '001874', '001877', '001880', '001955', '002001', '002015', '002018', '002035', '002049', '002059', '002082', '002087', '002095', '002123', '002125', '002133', '002152', '002179', '002181', '002207', '002222', '002244', '002258', '002296', '002319', '002340', '002360', '002383', '002390', '002407', '002455', '002457', '002498', '002514', '002553', '002555', '002556', '002574', '002612', '002649', '002662', '002666', '002707', '002718', '002767', '002770', '002792', '002819', '002839', '002849', '002863', '002891', '002910', '002938', '002943', '002945', '002952', '003025', '003026', '003028', '003030', '003053', '003115', '003120', '003152', '003154', '003169', '003243', '003291', '003292', '003299', '003304', '003343', '003446', '003501', '003502', '003503', '003561', '003567', '003593', '003594', '003629', '003642', '003659', '003670', '003692', '003734', '003751', '003843', '003857', '003862', '003890', '003962', '003993', '004008', '004011', '004050', '004076', '004119', '004128', '004142', '004153', '004183', '004205', '004223', '004249', '004263', '004316', '004332', '004341', '004351', '004352', '004390', '004423', '004448', '004475', '004495', '004496', '004604', '004666', '004671', '004702', '004703', '004707', '004737', '004745', '004750', '004763', '004788', '004809', '004815', '004819', '004823', '004833', '004846', '004852', '004854', '004890', '004895', '004913', '004946', '004986', '004987', '005005', '005039', '005059', '005067', '005117', '005128', '005140', '005161', '005164', '005187', '005242', '005259', '005265', '005268', '005274', '005290', '005296', '005324', '005331', '005402', '005409', '005412', '005482', '005492', '005505', '005534', '005541', '005550', '005618', '005634', '005669', '005674', '005682', '005689', '005698', '005699', '005708', '005732', '005741', '005743', '005760', '005763', '005774', '005775', '005777', '005794', '005802', '005815', '005823', '005825', '005826', '005834', '005840', '005844', '005851', '005855', '005865', '005876', '005885', '005888', '005901', '005903', '005904', '005910', '005911', '005914', '005939', '005955', '005970', '005977', '005984', '006005', '006051', '006085', '006098', '006122', '006128', '006132', '006136', '006154', '006167', '006227', '006251', '006259', '006267', '006270', '006274', '006281', '006282', '006299', '006308', '006314', '006323', '006366', '006373', '006377', '006395', '006401', '006429', '006445', '006449', '006457', '006523', '006533', '006535', '006538', '006547', '006551', '006574', '006603', '006616', '006620', '006692', '006700', '006720', '006748', '006751', '006780', '006792', '006803', '006864', '006961', '006973', '006976', '006992', '007043', '007049', '007107', '007137', '007139', '007202', '007229', '007231', '007254', '007271', '007280', '007300', '007439', '007455', '007467', '007484', '007509', '007549', '007632', '007663', '007674', '007729', '007731', '007775', '007777', '007809', '007811', '007827', '007850', '007925', '007945', '007950', '008060', '008071', '008077', '008082', '008128', '008180', '008189', '008212', '008227', '008253', '008270', '008274', '008279', '008297', '008308', '008314', '008326', '008328', '008381', '008420', '008457', '008480', '008499', '008532', '008602', '008638', '008640', '008707', '008713', '008763', '008905', '008917', '008928', '008980', '008997', '009016', '009033', '009049', '009055', '009077', '009128', '009206', '009225', '009258', '009370', '009402', '009486', '009488', '009537', '009601', '009690', '009707', '009828', '009849', '009853', '009882', '009907', '009912', '009913', '009932', '009989', '009993', '010020', '010029', '010128', '010349', '010365', '010371', '010389', '010563', '010636', '011066', '011150', '011172', '011230', '011351', '011452', '011462', '011707', '020003', '020005', '020015', '020026', '040011', '070021', '080005', '090016', '100056', '100060', '110005', '110011', '110025', '118002', '121006', '151002', '160215', '160226', '160311', '160323', '160512', '160517', '160610', '160627', '160642', '161233', '161605', '161606', '161611', '162102', '162202', '162205', '162207', '163412', '163503', '164824', '165512', '166011', '166019', '166023', '166109', '166801', '168002', '169201', '180001', '180018', '200006', '200010', '206002', '210004', '210008', '213001', '213006', '217002', '217012', '217021', '240004', '240008', '257040', '260112', '260117', '270001', '270002', '270042', '270050', '288001', '290006', '290012', '310308', '320001', '320006', '320012', '320013', '320016', '320018', '350001', '350002', '350007', '360016', '376510', '377150', '398011', '398031', '398061', '410006', '420005', '457001', '460005', '510080', '519002', '519019', '519033', '519056', '519087', '519095', '519097', '519120', '519125', '519183', '519196', '519197', '519198', '519212', '519616', '519644', '519655', '519670', '519696', '519702', '519710', '519712', '519767', '519773', '519778', '519918', '519959', '519979', '519981', '519997', '530016', '539002', '540001', '540004', '540007', '570008', '580001', '580009', '590003', '590005', '590008', '610002', '630001', '630002', '630006', '671030', '673010', '673043', '673071', '673141', '690001', '690009', '710001', '710002', '720001', '740001', '750001', '762001', '770001', '960004', '960008']
    invalid_code_list = list(set(all_codes) - set(valid_code_list))

    # 3. 加载并合并数据
    all_df_list = []
    for i in range(6):
        df_2d = load_and_merge_parquet_by_dim(dimension=i + 2, min_days=1250, min_score=0.5)
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
    print(f"合并完成，原始数据总量: {len(all_df):,} 条记录。正在进行代码过滤...当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

    print(f"过滤前数据量: {len(all_df)}, 过滤后数据量: {len(all_df_filter)} (共过滤掉 {len(all_df) - len(all_df_filter):,} 条记录)，耗时: {time.time() - start_time:.2f} 秒")
    all_df_filter['score'] = all_df_filter['CAGR'] * 10 *all_df_filter['CAGR'] * 10 *all_df_filter['CAGR'] * 10 *all_df_filter['CAGR'] * 10 * all_df_filter['Total_Score']
    # df_d 按照score降序排序
    all_df_filter = all_df_filter.sort_values(by='score', ascending=False).reset_index(drop=True)

    all_df_filter['score1'] = all_df_filter['Calmar_Ratio'] - all_df_filter['Calmar_Baseline']

    df_d = load_and_merge_parquet_by_dim(dimension=5, min_days=600, min_score=0)
    df_d['score'] = df_d['CAGR'] *  df_d['Total_Score'] * 10

    filtered_df1 = df_d[df_d['组合文件名'].str.contains('000390') & df_d['组合文件名'].str.contains('009033')& df_d['组合文件名'].str.contains('160323')& df_d['组合文件名'].str.contains('539001')]

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