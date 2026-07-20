# =============================================================================
# 【功能摘要】
#   公募基金一体化数据管线：完成"全市场基金采集 → 复权净值与风险指标加工 →
#   FOF 多基金组合择优排名"三段式流水，为量化选基/资产配置提供可读的候选组合。
#
# 【输入数据】
#   1) akshare 在线接口：全市场基金基础表、每日申购状态表、单只基金净值走势、重仓股。
#   2) 本地磁盘：
#      - fund_data/nav/*.csv               单只基金历史净值(含单位净值/日增长率)
#      - fund_data/holdings/*_holdings_*.csv 单只基金分季重仓股
#      - fund_data/*.parquet               各维度(2d/3d/5d)FOF组合评估结果
#      - temp/active_fund_codes.csv         基金代码/简称/类型映射(采集段产物)
#      - fund_data/base_pool_rejection_reasons.csv 基础池剔除归因(含"文件"列)
#
# 【数据流转/交互】
#   · 采集段  akshare 拉全量 → 只留"可申购"状态 → filter_fund_universe 语义清洗
#             → 逐只落盘净值/重仓股(断点续传, 空数据建空文件占位)。
#   · 加工段  逐 CSV 截近 N 行 → calculate_adj_nav 累乘复权 →
#             calculate_fund_stats 出统计指标 → 多进程并行(ProcessPool, 20 并发)。
#   · 择优段  pyarrow 谓词下推读 Parquet → 白名单代码过滤 → 复合打分 →
#             用简称字典把组合代码翻译成"人类可读组合名" → 标记并筛出含海外基金的组合。
#
# 【输出数据】
#   · 磁盘副作用：fund_data/nav|holdings 下的原始与 *_adj.csv、汇总 all_funds_result.csv。
#   · 返回值    ：analyze_fof_combinations 返回按复合分排序的海外候选组合 DataFrame。
# =============================================================================

import os
import re
import glob
import time
import concurrent.futures
from datetime import datetime

import pandas as pd
import akshare as ak
from tqdm import tqdm


# -----------------------------------------------------------------------------
# 业务白名单：经人工回测验证、允许进入 FOF 组合的基金代码(纯数字, 后续统一补零到6位)。
# 属于业务数据资产, 原样保留。
# -----------------------------------------------------------------------------
VALID_FUND_CODES = [
    20, 61, 66, 73, 166, 173, 209, 263, 354, 409, 411, 457, 458, 462, 496, 513,
    522, 524, 586, 592, 598, 601, 609, 612, 619, 646, 688, 689, 697, 714, 717,
    800, 805, 823, 828, 845, 927, 935, 979, 1000, 1009, 1039, 1053, 1069, 1070,
    1072, 1075, 1076, 1103, 1105, 1128, 1144, 1152, 1154, 1156, 1158, 1162, 1173,
    1184, 1188, 1194, 1198, 1210, 1216, 1227, 1239, 1245, 1261, 1268, 1297, 1302,
    1313, 1365, 1370, 1387, 1396, 1398, 1402, 1404, 1411, 1416, 1437, 1471, 1475,
    1476, 1496, 1521, 1532, 1534, 1538, 1607, 1613, 1672, 1678, 1702, 1707, 1709,
    1712, 1731, 1741, 1744, 1749, 1753, 1759, 1808, 1816, 1856, 1869, 1877, 1892,
    1933, 1959, 1985, 2064, 2083, 2095, 2125, 2145, 2149, 2160, 2244, 2251, 2256,
    2272, 2281, 2289, 2292, 2345, 2367, 2376, 2407, 2420, 2446, 2450, 2482, 2563,
    2577, 2692, 2707, 2770, 2776, 2860, 2861, 2862, 2863, 2885, 2893, 2900, 2910,
    2939, 3025, 3131, 3145, 3304, 3567, 3586, 3626, 3659, 3670, 3745, 3853, 3857,
    3961, 4128, 4148, 4183, 4332, 4352, 4374, 4390, 4434, 4497, 4616, 4641, 4666,
    4671, 4677, 4784, 4833, 4890, 5001, 5009, 5028, 5037, 5041, 5090, 5136, 5161,
    5186, 5251, 5268, 5299, 5310, 5328, 5343, 5472, 5482, 5537, 5544, 5550, 5571,
    5628, 5660, 5668, 5682, 5700, 5711, 5726, 5729, 5774, 5775, 5777, 5819, 5825,
    5826, 5844, 5939, 5962, 5977, 5983, 6025, 6154, 6195, 6230, 6250, 6270, 6374,
    6429, 6430, 6502, 6522, 6551, 6615, 6736, 6769, 6813, 6863, 6864, 6868, 6887,
    6969, 6976, 7074, 7113, 7119, 7133, 7146, 7203, 7277, 7449, 7497, 7527, 7639,
    7674, 7775, 8009, 8020, 8065, 8072, 8082, 8085, 8185, 8186, 8251, 8264, 8381,
    8382, 8555, 8633, 8635, 8638, 8655, 8671, 8903, 8949, 8962, 8980, 8983, 8988,
    9008, 9023, 9025, 9048, 9055, 9062, 9188, 9234, 9318, 9394, 9402, 9486, 9488,
    9640, 9644, 9651, 9715, 9808, 9847, 9853, 9855, 9857, 9861, 9899, 9913, 9929,
    9988, 9989, 9993, 9994, 10020, 10088, 10114, 10147, 10166, 10180, 10202, 10303,
    10313, 10335, 10371, 10389, 10415, 10421, 10495, 10622, 10662, 10730, 10761,
    10792, 10807, 10808, 10826, 10925, 10936, 11011, 11030, 11035, 11056, 11122,
    11144, 11186, 11196, 11264, 11269, 11369, 11377, 11446, 11488, 11599, 11603,
    11637, 11800, 11815, 11828, 11884, 11888, 11956, 12093, 12098, 12102, 12147,
    12188, 12198, 12200, 12223, 12243, 12294, 12301, 12319, 12445, 12454, 12477,
    12491, 12493, 12500, 12530, 12545, 12696, 12844, 12846, 12850, 12920, 12925,
    13000, 13085, 13103, 13107, 13175, 13238, 13242, 13250, 13296, 13341, 13365,
    13383, 13389, 13469, 13495, 13674, 13693, 13755, 13842, 13855, 13886, 13910,
    13942, 13958, 14144, 14175, 14185, 14189, 14191, 14254, 14267, 14287, 14292,
    14299, 14319, 14352, 14401, 14416, 14478, 14488, 14526, 14541, 14545, 14558,
    14600, 14647, 14736, 14807, 14818, 14825, 14854, 15035, 15079, 15145, 15192,
    15229, 15368, 15381, 15527, 15699, 15703, 15724, 15749, 15789, 15842, 15904,
    15967, 15970, 16045, 16097, 16105, 16117, 16122, 16165, 16182, 16183, 16237,
    16243, 16250, 16305, 16340, 16388, 16485, 16568, 16605, 16623, 16664, 16703,
    16772, 16873, 17036, 17073, 17075, 17192, 17234, 17471, 17483, 17488, 17547,
    17549, 17551, 17602, 17639, 17667, 17730, 17737, 17744, 17746, 17751, 17794,
    17824, 17835, 17876, 17878, 17960, 17987, 18000, 18019, 18122, 18194, 18229,
    18244, 18287, 18358, 18375, 18418, 18430, 18442, 18504, 18547, 18554, 18611,
    18730, 18790, 18796, 18815, 18835, 18865, 18868, 18876, 18910, 18916, 18918,
    18926, 18956, 18983, 18993, 18999, 19006, 19119, 19155, 19219, 19226, 19281,
    19293, 19336, 19347, 19367, 19374, 19410, 19426, 19431, 19447, 19612, 19702,
    19759, 19765, 19767, 19820, 19829, 19888, 20010, 20018, 20064, 20416, 20424,
    20433, 20440, 20469, 20553, 20560, 20661, 20685, 20722, 20755, 20821, 20975,
    21033, 21145, 21278, 21382, 21431, 21510, 21593, 21623, 21626, 21642, 21647,
    21730, 21792, 21875, 21967, 22003, 22028, 22299, 22311, 22334, 22364, 22490,
    22704, 22717, 22754, 23044, 23135, 23298, 23397, 23407, 23451, 23461, 23518,
    23524, 23532, 23632, 23638, 23651, 23765, 23782, 23851, 23859, 23875, 23889,
    23907, 23954, 23989, 23990, 24020, 24059, 40001, 40021, 40025, 50004, 50014,
    90012, 100055, 100060, 110005, 110009, 110010, 110012, 110015, 160220, 160324,
    160605, 160638, 160642, 160722, 161133, 161217, 161606, 161610, 161728, 161910,
    162201, 162703, 163116, 163411, 163818, 164205, 164212, 166301, 167002, 168002,
    169101, 169105, 180031, 200001, 200012, 202023, 202027, 210004, 217020, 240011,
    257070, 270028, 290008, 320001, 320006, 320016, 340006, 350007, 377240, 410001,
    410006, 410009, 457001, 460007, 470009, 481001, 481010, 481015, 501015, 501026,
    501064, 501073, 501085, 501096, 501097, 501200, 501201, 501205, 501210, 501226,
    519025, 519026, 519029, 519089, 519095, 519158, 519172, 519183, 519195, 519644,
    519674, 519679, 519688, 519704, 519766, 519770, 519773, 519929, 519935, 530019,
    539002, 540010, 570001, 570008, 580006, 582003, 610006, 630006, 630010, 630011,
    630016, 660015, 673060, 673141, 740001, 952099, 959991,
]


# -----------------------------------------------------------------------------
# 目录初始化
# -----------------------------------------------------------------------------
def create_folders():
    """确保所有落盘目录存在, 缺失即创建。"""
    dirs = ['temp', 'fund_data/nav', 'fund_data/holdings']
    created = [d for d in dirs if not os.path.exists(d)]
    for d in created:
        os.makedirs(d)
    print(f"【目录初始化】完成 | 需创建: {created if created else '无(均已存在)'}")


# -----------------------------------------------------------------------------
# 基金池语义清洗：识别并剔除"主动型债券基金"(有嫌疑词且无被动指数护身符)。
# -----------------------------------------------------------------------------
def _get_active_bond_mask(name_series):
    """返回主动债剔除掩码。判据: 命中嫌疑词 且 未命中被动指数护身符 = 主动债。"""
    # 嫌疑词: 通常代表基金经理主动挑券或做波段
    suspect_words = ['纯债', '短债', '信用债', '收益债', '回报',
                     '双季', '季季', '月月', '添利', '增利', '稳利']
    # 护身符: 严格跟踪指数的被动债, 不予剔除
    shield_words = ['指数', '中债', '彭博', '上清所', '国开', '政金', '农发']

    is_suspect = name_series.str.contains('|'.join(suspect_words), regex=True, na=False)
    has_shield = name_series.str.contains('|'.join(shield_words), regex=True, na=False)
    return is_suspect & (~has_shield)


def filter_fund_universe(df, type_col='基金类型', name_col='基金简称', remove_c_class=True):
    """
    清洗基金底仓池: 大类剔除 → 主动债狙击 → A/C 冗余份额去重, 并输出归因报告。
    目的是留下"仓位稳定、可复权、代表性单一份额"的干净候选池。
    """
    if df is None or df.empty:
        print("【基金池清洗】跳过 | 原因: [传入数据为空]")
        return df

    res_df = df.copy()
    total_before = len(res_df)
    clean_stats = {}

    # 模块一: 按【基金类型】剔除会污染相关性/收益的大类
    type_filter_rules = {
        "1. 混合型 (仓位飘忽污染相关性)": [
            '混合型-灵活', '混合型-偏股', '混合型-偏债', '混合型-平衡', '混合型-绝对收益',
            'QDII-混合偏股', 'QDII-混合债', 'QDII-混合灵活', 'QDII-混合平衡'],
        "2. 假纯债 (含股票仓位易暴雷)": ['债券型-混合二级', '债券型-混合一级'],
        "3. 纯主动股票型 (回测噪音太大)": ['股票型', 'QDII-普通股票'],
        "4. 货币类 (拉低收益导致资金站岗)": ['货币型-普通货币', '货币型-浮动净值'],
        "5. FOF类 (避免资产配置套娃)": ['FOF-稳健型', 'FOF-进取型', 'FOF-均衡型', 'QDII-FOF'],
    }
    if type_col in res_df.columns:
        res_df[type_col] = res_df[type_col].fillna('').str.strip()
        for reason, types in type_filter_rules.items():
            mask = res_df[type_col].isin(types)
            if mask.sum() > 0:
                clean_stats[reason] = int(mask.sum())
            res_df = res_df[~mask]

    # 模块二: 按【基金名称】语义精准狙击主动债
    if name_col in res_df.columns:
        active_bond_mask = _get_active_bond_mask(res_df[name_col])
        if active_bond_mask.sum() > 0:
            clean_stats["6. 主动型债券 (防范信用/久期暴露)"] = int(active_bond_mask.sum())
        res_df = res_df[~active_bond_mask]

    # 模块三: 剔除同一只基金的冗余份额, 排序后仅保留 A 份额
    if remove_c_class and name_col in res_df.columns:
        before_ac = len(res_df)
        res_df['base_name'] = res_df[name_col].str.replace(r'[ABCDEHIY]$', '', regex=True)
        res_df = res_df.sort_values(by=name_col).drop_duplicates(subset=['base_name'], keep='first')
        res_df = res_df.drop(columns=['base_name'])
        ac_count = before_ac - len(res_df)
        if ac_count > 0:
            clean_stats["7. 冗余份额去重 (剔除C/E等)"] = ac_count

    # 模块四: 归因报告(聚合为一段, 一眼看清剔除结构)
    total_after = len(res_df)
    total_cleaned = total_before - total_after
    print("\n" + "=" * 55)
    print("【基金池清洗报告】")
    print(f"  清洗前: [{total_before:,}] 只 | 最终保留: [{total_after:,}] 只 | 剔除: [{total_cleaned:,}] 只")
    if total_cleaned > 0:
        print("-" * 55 + "\n  [剔除归因及占比]")
        for reason, count in sorted(clean_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"    {reason:<26} : [{count:>5,}] 只 ({count / total_cleaned:>6.2%})")
    print("=" * 55 + "\n")

    return res_df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 采集段-1: 获取"当前可申购"的活跃基金代码/名称映射, 并落盘供断点续传。
# -----------------------------------------------------------------------------
def get_active_fund_codes():
    """
    拉取全市场基金, 仅保留申购状态可买入的基金, 输出 {代码: 简称} 字典。
    Why: 已清盘/封闭期的基金无法建仓, 提前剔除避免后续无效抓取。
    """
    cache_file = 'temp/active_fund_codes.csv'

    print("【采集-基础信息】正在拉取全市场基金列表...")
    df_all = ak.fund_name_em()

    print("【采集-申购状态】正在剔除已清盘/当前无法申购的基金...")
    try:
        df_active = ak.fund_open_fund_daily_em()
        # 仅保留散户可买入的状态(大额限制一般>10万, 不影响普通申购)
        buyable_status = ['开放申购', '限大额', '场内交易']
        df_buyable = df_active[df_active['申购状态'].isin(buyable_status)]
        active_codes = set(df_buyable['基金代码'].astype(str).str.zfill(6).tolist())
        print(f"  今日更新净值: [{len(df_active)}] 只 | 可正常申购: [{len(df_buyable)}] 只")
    except Exception as e:
        print(f"【警告】拉取每日申购状态失败, 降级为使用全量列表 | 可能原因: [接口异常/字段变动] | 详情: {e}")
        active_codes = set(df_all['基金代码'].astype(str).str.zfill(6).tolist())

    df_filtered = df_all[df_all['基金代码'].isin(active_codes)].copy()
    print(f"  全量: [{len(df_all)}] 只 → 剔除不可购买后候选: [{len(df_filtered)}] 只")

    fund_dict = dict(zip(df_filtered['基金代码'], df_filtered['基金简称']))
    print(f"【采集-完成】可申购候选基金共计: [{len(fund_dict)}] 只")

    # 落盘映射(代码/简称/类型), 供后续断点续传与名称翻译使用
    pd.DataFrame({
        '基金代码': list(fund_dict.keys()),
        '基金简称': list(fund_dict.values()),
        '基金类型': df_filtered['基金类型'].tolist(),
    }).to_csv(cache_file, index=False, encoding='utf-8-sig')
    print(f"  已落盘代码映射 → [{cache_file}]")

    return fund_dict


# -----------------------------------------------------------------------------
# 采集段-2: 逐只下载历史净值与重仓股。断点续传, 空/异常数据建空文件占位防重复请求。
# -----------------------------------------------------------------------------
def fetch_and_save_fund_data(fund_codes, year="2024", test_mode=False):
    """遍历基金字典 {code: name}, 抓取净值走势与重仓股并落盘。"""
    all_codes_list = list(fund_codes.keys())
    codes_to_process = all_codes_list[:5] if test_mode else all_codes_list
    mode_tag = '测试模式(前5只)' if test_mode else '全量'
    print(f"\n【采集-下载】启动 | 模式: [{mode_tag}] | 计划处理: [{len(codes_to_process)}] 只")

    for code in tqdm(codes_to_process, desc="下载进度"):
        fund_name = fund_codes[code]

        # 净值走势: 仅当本地无文件时才请求
        nav_file = f"fund_data/nav/{code}_nav.csv"
        if not os.path.exists(nav_file):
            try:
                nav_df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
                if not nav_df.empty:
                    nav_df.to_csv(nav_file, index=False, encoding='utf-8-sig')
                else:
                    open(nav_file, 'w').close()  # 空数据占位, 防下次重复请求
            except Exception:
                open(nav_file, 'w').close()
                tqdm.write(f"【跳过-净值】抓取失败 | 基金: [{code} {fund_name}] | 可能原因: [无净值数据/接口解析异常]")

        # 重仓股: 同样断点续传
        holdings_file = f"fund_data/holdings/{code}_holdings_{year}.csv"
        if not os.path.exists(holdings_file):
            try:
                holdings_df = ak.fund_portfolio_hold_em(symbol=code, date=year)
                if not holdings_df.empty:
                    holdings_df.to_csv(holdings_file, index=False, encoding='utf-8-sig')
                else:
                    open(holdings_file, 'w').close()
            except Exception:
                open(holdings_file, 'w').close()
                tqdm.write(f"【跳过-重仓股】抓取失败 | 基金: [{code} {fund_name}] | 可能原因: [无持仓披露/接口解析异常]")


# -----------------------------------------------------------------------------
# 加工段: 复权净值计算
# -----------------------------------------------------------------------------
def calculate_adj_nav(df):
    """
    由"单位净值 + 日增长率"累乘还原复权净值。
    注意: 接口日增长率以百分数计(1.5 表示 1.5%), 故除以 100。
    ⚠ 保留原口径: 复权因子从首行即计入当日涨幅(cumprod), 首日复权净值=真实净值×(1+r0/100)。
       若需严格对齐首日, 应令首行因子为 1 —— 此为财务口径变更, 待授权后再改。
    """
    res_df = df.copy()
    if res_df.empty:
        return res_df

    res_df['日增长率'] = pd.to_numeric(res_df['日增长率'], errors='coerce').fillna(0)
    res_df['复权因子'] = (1 + res_df['日增长率'] / 100).cumprod()
    first_day_nav = res_df['单位净值'].dropna().iloc[0]
    res_df['复权净值'] = res_df['复权因子'] * first_day_nav
    return res_df


def calculate_fund_stats(df, fund_name="Unknown", date_col='净值日期', nav_col='单位净值'):
    """
    统计单只基金的生命周期与数据质量指标(不做淘汰判断, 仅体检记录)。
    产出: 活跃天数、真实起止日、缺失率、最长连续零收益(死水)天数。
    """
    result = {
        "fund_name": fund_name, "adj_nav_file": None, "total_active_days": 0,
        "real_start": None, "real_end": None, "missing_ratio": 0.0, "max_zeros": 0,
        "max_drawdown": 0.0, "annualized_return": 0.0, "top_10_holdings": "",
    }
    if df is None or df.empty or nav_col not in df.columns or date_col not in df.columns:
        return result

    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    temp_df = (temp_df.sort_values(by=date_col)
                      .drop_duplicates(subset=[date_col], keep='last')
                      .set_index(date_col))
    nav_series = temp_df[nav_col]

    # 剥离首尾全空段, 定位真实生命周期
    valid_series = nav_series.dropna()
    if valid_series.empty:
        return result

    real_start, real_end = valid_series.index[0], valid_series.index[-1]
    active_series = nav_series.loc[real_start:real_end]
    total_active_days = len(active_series)

    result["total_active_days"] = total_active_days
    result["real_start"] = real_start.strftime('%Y-%m-%d')
    result["real_end"] = real_end.strftime('%Y-%m-%d')
    if total_active_days > 0:
        result["missing_ratio"] = float(active_series.isna().sum() / total_active_days)

    # 最长连续零收益(死水)天数: 优先用日增长率, 无则由净值推算
    if '日增长率' in temp_df.columns:
        rets = pd.to_numeric(temp_df.loc[real_start:real_end, '日增长率'], errors='coerce').fillna(0)
    else:
        rets = active_series.ffill().pct_change().fillna(0.0)
    is_zero = (rets.abs() < 1e-8).astype(int)
    max_zeros = is_zero.groupby((is_zero == 0).cumsum()).sum().max()
    result["max_zeros"] = int(max_zeros) if pd.notna(max_zeros) else 0

    return result


def process_fund_pipeline(file_path, save_qualified=True, date_col='净值日期',
                          nav_col='单位净值', head_count=300):
    """
    单文件加工流水: 截近 N 行 → 统计指标 → 复权净值 → 年化/回撤 → 前十大持仓。
    设计为进程池的工作单元, 故所有异常在内部吞掉并以空结果兜底, 保证批处理不中断。
    """
    fund_name = os.path.basename(file_path).split('_')[0]

    try:
        df = pd.read_csv(file_path)
    except Exception:
        df = pd.DataFrame()

    # 仅保留最近 head_count 行(倒序取头再正序还原)
    if not df.empty and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = (df.sort_values(by=date_col, ascending=False)
                .head(head_count).sort_values(by=date_col).reset_index(drop=True))

    result = calculate_fund_stats(df, fund_name=fund_name, date_col=date_col, nav_col=nav_col)

    if not df.empty and date_col in df.columns and nav_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).reset_index(drop=True)
        qualified_df = calculate_adj_nav(df)

        # 基于复权净值算年化收益与最大回撤
        if '复权净值' in qualified_df.columns:
            adj_series = qualified_df.set_index(date_col)['复权净值'].dropna()
            if not adj_series.empty and result["total_active_days"] > 0:
                start_val, end_val = adj_series.iloc[0], adj_series.iloc[-1]
                years = result["total_active_days"] / 252.0
                if start_val > 0 and years > 0:
                    result["annualized_return"] = round(float((end_val / start_val) ** (1 / years) - 1), 6)
                roll_max = adj_series.cummax()
                if not roll_max.empty and roll_max.max() > 0:
                    drawdown = (adj_series - roll_max) / roll_max
                    result["max_drawdown"] = round(float(drawdown.min()), 6)

        if save_qualified:
            saved_path = file_path.replace('.csv', '_adj.csv')
            qualified_df.to_csv(saved_path, index=False, encoding='utf-8-sig')
            result["adj_nav_file"] = saved_path

    # 前十大持仓: 取最新年份的持仓文件
    holdings_files = sorted(glob.glob(f"fund_data/holdings/{fund_name}_holdings_*.csv"), reverse=True)
    if holdings_files:
        try:
            hdf = pd.read_csv(holdings_files[0])
            if '股票名称' in hdf.columns:
                result["top_10_holdings"] = ",".join(
                    hdf.head(10)['股票名称'].dropna().astype(str).tolist())
        except Exception:
            pass

    return result


def judge_fund_df(head_count=300, max_workers=20):
    """
    批量加工 fund_data/nav 下所有原始净值 CSV(跳过已生成的 *_adj.csv), 多进程并行,
    结果汇总落盘 all_funds_result.csv, 并输出统一体检报告。
    """
    target_dir = 'fund_data/nav'
    print(f"【批量加工】扫描目录: [{target_dir}]")
    if not os.path.exists(target_dir):
        print("【批量加工】终止 | 原因: [目录不存在, 请先执行采集段]")
        return

    all_csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not all_csv_files:
        print("【批量加工】终止 | 原因: [目录下无任何 CSV 文件]")
        return

    files_to_process = [f for f in all_csv_files if not os.path.basename(f).endswith("_adj.csv")]
    skipped_count = len(all_csv_files) - len(files_to_process)
    print(f"  发现文件: [{len(all_csv_files)}] 个 | 待处理: [{len(files_to_process)}] 个 | 跳过(已复权): [{skipped_count}] 个")

    stats = {"processed": 0, "skipped": skipped_count, "error": 0}
    all_results = []

    if files_to_process:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_fund_pipeline, fp, True, '净值日期', '单位净值', head_count)
                       for fp in files_to_process]
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="处理进度"):
                try:
                    result = future.result()
                    all_results.append(result)
                    stats["processed" if result.get("adj_nav_file") else "error"] += 1
                except Exception:
                    stats["error"] += 1

    if all_results:
        output_csv = "fund_data/all_funds_result.csv"
        pd.DataFrame(all_results).to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"【批量加工】汇总已落盘 → [{output_csv}]")

    print("-" * 50)
    print("【批量加工报告】")
    print(f"  总文件: [{len(all_csv_files)}] | 成功: [{stats['processed']}] | "
          f"跳过: [{stats['skipped']}] | 空数/报错: [{stats['error']}]")
    print("-" * 50)


# -----------------------------------------------------------------------------
# 择优段-辅助: 按维度并发加载 Parquet(pyarrow 谓词下推硬盘级过滤), 合并排序。
# -----------------------------------------------------------------------------
def load_and_merge_parquet_by_dim(dimension, data_dir='fund_data', min_days=600,
                                  min_score=None, filter_code=None):
    """
    加载并合并某维度(如 2/3/5)的所有 FOF 组合评估 Parquet, 在硬盘读取阶段即过滤,
    最终按 Total_Score 降序返回。异常时自动降级为逐文件安全加载。
    """
    search_pattern = os.path.join(data_dir, f'fof_evaluation_results_{dimension}d_*.parquet')
    file_list = glob.glob(search_pattern)
    if not file_list:
        print(f"【Parquet加载】跳过 | 维度: [{dimension}d] | 原因: [未找到匹配文件] | 路径: {search_pattern}")
        return pd.DataFrame()

    print(f"【Parquet加载】维度: [{dimension}d] | 命中文件: [{len(file_list)}] 个 | 启动谓词下推加载...")

    read_filters = [('Total_Days', '>', min_days)]
    if min_score is not None:
        read_filters.append(('Total_Score', '>', min_score))

    try:
        df = pd.read_parquet(file_list, engine='pyarrow', filters=read_filters)
        if filter_code is not None:
            df = df[df['组合文件名'].str.contains(filter_code, na=False)]
    except Exception as e:
        print(f"【Parquet加载】批量并发失败, 触发逐文件降级 | 可能原因: [个别文件损坏/字段不一致] | 详情: {e}")
        dfs = []
        for file in file_list:
            try:
                df_part = pd.read_parquet(file, engine='pyarrow', filters=read_filters)
                if not df_part.empty:
                    dfs.append(df_part)
            except Exception as ex:
                print(f"  【跳过】损坏文件: [{os.path.basename(file)}] | 详情: {ex}")
        if not dfs:
            print(f"【Parquet加载】维度: [{dimension}d] 全部读取失败或无符合条件数据。")
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)

    if df.empty:
        print(f"【Parquet加载】维度: [{dimension}d] 加载完成, 但硬盘级过滤后无留存组合。")
        return df

    df = df.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)
    print(f"【Parquet加载】维度: [{dimension}d] 合并完成 | 过滤后留存: [{len(df):,}] 个组合")
    return df


def get_blacklisted_fund_codes(df):
    """
    按人工维护的"主动债黑名单"检索命中行, 返回其 6 位代码列表, 并报告未匹配到的黑名单词。
    Why: 这些主动债 QDII/久期暴露基金不适合进入被动化 FOF 组合。
    """
    active_funds_to_exclude = [
        "华夏鼎航债券", "兴业裕丰债券", "鹏华丰惠债券", "西部利得汇盈债券",
        "广发景宁债券", "华安安业债券", "中信保诚景丰", "国金惠安利率债",
        "南方交元债券", "华安鼎丰债券", "华宝宝泓债券", "兴全恒裕债券",
        "德邦锐乾债券", "中信保诚稳达", "兴全稳泰债券", "德邦锐裕利率债债券",
        "天弘信益债券", "中信保诚稳丰", "鹏华丰禄债券", "西部利得祥逸债券",
        "永赢裕益债券", "南方旭元债券", "兴业福鑫债券", "永赢邦利债券",
        "前海开源鼎欣债券", "中信保诚稳泰债券", "永赢昌利债券", "诺德安鸿",
        "鹏华金利债券", "华夏鼎通债券", "华夏鼎隆债券", "鹏扬淳开债券",
        "易方达中短期美元债", "中银汇享债券", "永赢伟益债券", "中信保诚稳益",
        "建信利率债债券", "鹏华丰鑫债券", "宏利永利债券", "鹏华0-5年利率债券",
        "中信保诚稳鸿", "华夏鼎康债券", "南方泽元债券", "永赢瑞益债券",
        "永赢惠益债券", "兴业裕恒债券", "鹏华丰腾债券",
    ]

    original_size = len(df)
    df_work = df.copy()
    # 兼容浮点读入, 统一补齐 6 位代码
    df_work['基金代码'] = df_work['基金代码'].astype(str).str.split('.').str[0].str.zfill(6)

    matched_names = set()
    mask_is_blacklisted = pd.Series(False, index=df_work.index)
    for bad_name in active_funds_to_exclude:
        current_match = df_work['基金简称'].astype(str).str.contains(bad_name, regex=False, na=False)
        if current_match.any():
            matched_names.add(bad_name)
            mask_is_blacklisted = mask_is_blacklisted | current_match

    unmatched_names = [n for n in active_funds_to_exclude if n not in matched_names]
    blacklisted_codes = df_work[mask_is_blacklisted]['基金代码'].unique().tolist()

    print("【黑名单提取报告】")
    print(f"  检索词: [{len(active_funds_to_exclude)}] 个 | 输入行: [{original_size}] | "
          f"命中行: [{int(mask_is_blacklisted.sum())}] | 返回代码: [{len(blacklisted_codes)}] 个")
    if unmatched_names:
        print(f"  【提示】未匹配到的黑名单词 [{len(unmatched_names)}] 个(可能已清盘或改名):")
        for name in unmatched_names:
            print(f"    - {name}")
    else:
        print("  ✅ 所有黑名单词均已成功匹配。")

    return blacklisted_codes


# -----------------------------------------------------------------------------
# 择优段-主流程: 加载多维 FOF 组合 → 白名单过滤 → 复合打分 → 生成可读组合名 → 筛出含海外基金组合。
# -----------------------------------------------------------------------------
def _simplify_fund_name(name):
    """把冗长的官方基金全称压缩为便于人眼识别的核心简称。"""
    if not isinstance(name, str):
        return name
    name = re.sub(r'\(.*?\)|（.*?）', '', name)  # 去括号及内容
    redundant_words = ['ETF联接', '联接', 'ETF', 'LOF', 'QDII', '灵活配置', '多策略',
                       '混合型', '混合', '指数增强', '增强型', '指数', '发起式',
                       '人民币份额', '人民币', '证券投资基金']
    name = re.sub('|'.join(redundant_words), '', name)
    name = re.sub(r'[A-Z]$', '', name)  # 去末尾份额字母
    return name


def analyze_fof_combinations(report_path='fund_data/base_pool_rejection_reasons.csv',
                             active_code_path='temp/active_fund_codes.csv',
                             dims=(2, 3), min_days=250, min_cagr=1.0):
    """
    输出按复合分排序、且包含海外配置的 FOF 组合候选表。
    数据流: base_pool 提供全体代码域 → active_code 提供名称/海外标签 →
            Parquet 提供组合评估指标 → 白名单交集过滤 → 打分排序 → 翻译成可读组合名。
    """
    print("\n" + "=" * 55)
    print("【FOF择优】启动组合分析流程")
    print("=" * 55)

    report_df = pd.read_csv(report_path)
    active_code_df = pd.read_csv(active_code_path)

    # 海外基金代码集合(QDII/海外), 用于最终标记
    overseas_mask = active_code_df['基金类型'].str.contains('QDII|海外', na=False)
    overseas_set = set(active_code_df[overseas_mask]['基金代码'].astype(str).str.zfill(6).tolist())

    # 以 active_code 为右表对齐, 从 report 的"文件"列抽取全体 6 位代码域
    report_df = report_df.merge(active_code_df, on='基金代码', how='right')
    all_codes = report_df['文件'].str.extract(r'(\d{6})')[0].dropna().unique().tolist()

    valid_codes = set(str(c).zfill(6) for c in VALID_FUND_CODES)
    invalid_set = set(all_codes) - valid_codes
    print(f"【FOF择优】代码域: [{len(all_codes)}] | 白名单: [{len(valid_codes)}] | 判为无效: [{len(invalid_set)}]")

    # 加载各维度组合, Total_Score 放大后合并, 仅留 CAGR>min_cagr 的组合
    all_df_list = []
    for dim in dims:
        df_dim = load_and_merge_parquet_by_dim(dimension=dim, min_days=min_days, min_score=0)
        if df_dim.empty:
            continue
        df_dim['Total_Score'] = df_dim['Total_Score'] * 10000
        df_dim = df_dim[df_dim['CAGR'] > min_cagr].reset_index(drop=True)
        all_df_list.append(df_dim)

    if not all_df_list:
        print("【FOF择优】终止 | 原因: [各维度均无符合 CAGR 条件的组合]")
        return pd.DataFrame()

    all_df = pd.concat(all_df_list, ignore_index=True)
    all_df = all_df.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)
    print(f"【FOF择优】合并完成 | 原始组合: [{len(all_df):,}] 条 | 时间: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # 剔除任何包含无效代码的组合(内联高速过滤: 预编译正则 + set 无交集判定)
    start_time = time.time()
    if invalid_set:
        pattern = re.compile(r'\d{6}')
        mask = [invalid_set.isdisjoint(pattern.findall(fn)) if type(fn) is str else False
                for fn in all_df['组合文件名'].tolist()]
        all_df_filter = all_df[mask].reset_index(drop=True)
    else:
        all_df_filter = all_df.copy()
    print(f"【FOF择优】白名单过滤 | 过滤前: [{len(all_df)}] → 过滤后: [{len(all_df_filter)}] "
          f"(剔除 [{len(all_df) - len(all_df_filter):,}]) | 耗时: <{time.time() - start_time:.2f}s>")

    # 复合打分: 强化 CAGR 权重((CAGR*10)^3 * Total_Score) 后排序, 附加辅助评分列
    all_df_filter['score'] = (all_df_filter['CAGR'] * 10) ** 3 * all_df_filter['Total_Score']
    all_df_filter = all_df_filter.sort_values(by='score', ascending=False).reset_index(drop=True)
    all_df_filter['score1'] = all_df_filter['Calmar_Ratio'] - all_df_filter['Calmar_Baseline']

    # 把组合代码翻译成"人类可读组合名"(简称拼接), 并提到首列
    padded_active_codes = active_code_df['基金代码'].astype(str).str.zfill(6)
    code_to_name = dict(zip(padded_active_codes, active_code_df['基金简称'].apply(_simplify_fund_name)))
    all_df_filter['基金简称组合'] = [
        '_'.join([str(code_to_name.get(c, c)) for c in fn.split('_')]) if type(fn) is str else fn
        for fn in all_df_filter['组合文件名'].tolist()
    ]
    all_df_filter.insert(0, '基金简称组合', all_df_filter.pop('基金简称组合'))

    # 定位类字段挪到末尾, 便于阅读
    tail_cols = ['组合文件名', 'Start_Date', 'End_Date', 'Total_Days']
    cols = [c for c in all_df_filter.columns if c not in tail_cols] + \
           [c for c in tail_cols if c in all_df_filter.columns]
    all_df_filter = all_df_filter[cols]

    # 派生指标: 组合基金数量、下行相关性惩罚下的多套评分
    all_df_filter['组合数量'] = all_df_filter['组合文件名'].apply(
        lambda x: len(str(x).split('_')) if type(x) is str else 0)
    dc = all_df_filter['Downside_Correlation']
    all_df_filter['score_corr_down'] = all_df_filter['Total_Score'] / (dc + 1)
    all_df_filter['score_corr_down1'] = all_df_filter['score'] / (dc + 1)
    all_df_filter['score_corr_down2'] = all_df_filter['Total_Score'] / (dc + 2)
    all_df_filter['score_corr_down3'] = all_df_filter['score'] / (dc + 2)

    # 标记并仅保留含海外基金的组合(海外配置视角)
    all_df_filter['包含海外基金'] = all_df_filter['组合文件名'].apply(
        lambda x: any(c in overseas_set for c in str(x).split('_')))
    result_df = all_df_filter[all_df_filter['包含海外基金']].reset_index(drop=True)

    print(f"【FOF择优】完成 | 候选组合: [{len(all_df_filter)}] → 含海外组合: [{len(result_df)}]")
    print("=" * 55 + "\n")
    return result_df


# -----------------------------------------------------------------------------
# 主入口: 择优分析 + 批量净值加工。采集段默认关闭, 需要时解注释即可跑全链路。
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- 采集段(可选): 首次运行请开启, 建议先 test_mode=True 跑通再切全量 ----
    create_folders()
    active_codes = get_active_fund_codes()
    fetch_and_save_fund_data(active_codes, year="2026", test_mode=True)

    # ---- 择优段: 输出含海外配置的 FOF 候选组合 ----
    fof_candidates = analyze_fof_combinations(dims=(2, 3), min_days=250, min_cagr=1.0)

    # ---- 加工段: 批量生成复权净值与统计汇总 ----
    judge_fund_df(head_count=300, max_workers=20)

    print("\n🎉 全部任务执行完毕, 请查看 fund_data 目录。")