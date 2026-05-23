import akshare as ak
import pandas as pd
import time
import os
from tqdm import tqdm
import pandas as pd


def calculate_adj_nav(df):
    """
    根据包含“单位净值”和“日增长率”的数据计算完整的复权净值。

    参数:
    df (pd.DataFrame): 必须包含 '单位净值' 和 '日增长率' 列。
                       通常按照日期从旧到新正序排列。

    返回:
    pd.DataFrame: 包含新增 '复权因子' 和 '复权单位净值' 列的 DataFrame。
    """
    # 建立副本以避免 pandas 的 SettingWithCopyWarning 警告
    res_df = df.copy()

    if res_df.empty:
        return res_df

    # 1. 容错处理：确保“日增长率”是浮点数，且将缺失值（NaN）或异常字符替换为 0
    res_df['日增长率'] = pd.to_numeric(res_df['日增长率'], errors='coerce').fillna(0)

    # 2. 核心数学逻辑：计算累乘的复权因子
    # 接口的数据如果是 1.5 代表 1.5%，所以要除以 100
    res_df['复权因子'] = (1 + res_df['日增长率'] / 100).cumprod()

    # 3. 对齐起点：用第一天的真实单位净值乘以复权因子，得出每天的复权净值
    first_day_nav = res_df['单位净值'].iloc[0]
    res_df['复权单位净值'] = res_df['复权因子'] * first_day_nav

    return res_df

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


if __name__ == "__main__":
    # 1. 创建存放数据的目录
    create_folders()

    # 2. 获取经过清洗过滤的活跃基金代码
    active_codes = get_active_fund_codes()

    # 3. 开始下载数据
    # 【重要说明】
    # 先保持 test_mode=True 运行一次，确认能跑通。
    # 确认没问题后，把下面这行的 test_mode 改为 False，就会去下载过滤后的大约 1万多只 基金了。
    fetch_and_save_fund_data(active_codes, year="2026", test_mode=False)

    print("\n🎉 全部任务执行完毕！请查看 fund_data 文件夹。")