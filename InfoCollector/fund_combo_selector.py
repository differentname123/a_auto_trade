# =====================================================================================
# 【功能摘要】
#   公募基金一体化量化管线（单文件版）。三段式流水贯通:
#     ① 采集段  : 全市场基金采集 → 只留可申购 → 逐只落盘净值/重仓股(断点续传)。
#     ② 加工段  : 截近 N 日 → 复权净值累乘 → 年化/回撤/数据质量体检 → 多进程并行落盘。
#     ③ 择优段  : 财务初筛 → 相关性去重 → Master 矩阵 → 逐维(2→N)Apriori 裂变评估 →
#                 分层 VETO 否决 + 打分 → 复合排序输出可读候选组合。
#
# 【输入数据】
#   1. akshare 在线接口             : 基金基础表/申购状态/净值走势/重仓股。
#   2. fund_data/nav/*.csv          : 单只基金历史净值(单位净值 + 日增长率)。
#   3. fund_data/all_funds_result.csv: 全市场基金汇总表(加工段产物)。
#   4. fund_data/*.parquet          : 各维度既往评估结果(断点缓存与升维种子)。
#
# 【输出数据】
#   1. fund_data/all_funds_result.csv                            : 加工段汇总。
#   2. fund_data/fof_evaluation_results_{N}d_*.parquet           : 每维评估明细。
#   3. fund_data/computed_combos_{N}d_global_cache.parquet       : 全局已算缓存。
#   4. fund_data/base_pool_rejection_reasons.csv                 : 全链路筛选审计。
#   5. base_pool_codes.json                                      : 最终 Base Pool 代码。
#   6. analyze_fof_combinations 返回按复合分排序的候选组合 DataFrame。
# =====================================================================================
import os
import re
import json
import glob
import math
import time
import shutil
import pickle
import itertools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import akshare as ak
from tqdm import tqdm
from numba import njit


# ====================================================================
# 【全局统一关键配置区】(所有影响回测与模型结果的参数统一在此维护)
# ====================================================================

# ----------------- 1. 核心回测时间与运行范围配置 -----------------
CFG_MAX_DIMENSION = 3                # 回测的最高维度(例如3代表最高计算3只基金组合)
CFG_RECENT_DAYS_LIMIT = 50           # 截取最近N天数据(决定了回测的时间窗口长度)
CFG_HOLDINGS_YEAR = "2026"           # 重仓股穿透对应的披露年份

TRADING_DAYS_PER_YEAR = 252          # 每年平均交易日天数(基准常数)
DEFAULT_REBALANCE_DAYS = 30          # FOF组合的默认调仓周期(天)
DEFAULT_MAX_HISTORY = 5 * TRADING_DAYS_PER_YEAR  # 算法最多追溯的历史天数(防老数据过度影响)
ROLLING_WINDOW = TRADING_DAYS_PER_YEAR # 滚动计算R2等指标的固定窗口长度
ROLLING_STEP = 21                    # 滚动计算时窗口的滑动步进天数(约1个月)

# ----------------- 2. 单只基金初筛与准入条件 (Base Pool 准入) -----------------
FIXED_MIN_DAY = 40                   # 基金必须存活的最短有效交易天数
CFG_BASE_POOL_MIN_CAGR = 0.5         # 单只基金初筛门槛: 年化收益率必须大于此绝对值(0.5即50%)
CFG_FUND_MIN_MDD = -0.7              # 单只基金初筛门槛: 历史最大回撤底线(低于此值剔除)

# ----------------- 3. 相关性过滤配置 (去同质化) -----------------
CFG_CORR_THRESHOLD = 0.95            # 全天候相关性剔除阈值(两只基金相关性高于此则视为高度冗余)
CFG_DOWNSIDE_CORR_THRESHOLD = 0.75   # 下行相关性剔除阈值(暴跌时高度同步则不能有效分散风险)
CFG_DOWNSIDE_TAIL_RATIO = 0.05       # 计算下行相关性时，取跌幅最大的百分之多少的数据进行计算
CFG_DOWNSIDE_MIN_DAYS = 5            # 计算下行相关性时，最少需要的数据天数

# ----------------- 4. 组合评估 VETO 一票否决机制 (核心避坑逻辑) -----------------
VETO_MAX_MDD = 0.5                   # 组合一票否决: 组合复合最大回撤不能超过绝对值(0.5=50%)
VETO_HURDLE_RATE = 0.05              # 组合一票否决: 组合复合年化必须跑赢此基准(0.05=5%)
VETO_AR1 = 0.55                      # 组合一票否决: 收益率一阶自相关系数上限(防虚假平滑)
VETO_VOL = 0.03                      # 组合一票否决: 年化波动率下限(低于此值视为固收类/造假)
VETO_RECOVERY_DAYS = 350             # 组合一票否决: 最长未创新高天数(回撤修复期太长)
VETO_MISSING_RATIO = 0.05            # 数据一票否决: 数据缺失率上限(0.05=5%)
VETO_CONTINUOUS_ZEROS = 10           # 数据一票否决: 最大连续零收益天数(防停牌/死水基金)
VETO_CALMAR_MULTIPLIER = 1.01        # 卡玛过滤: 组合卡玛比率必须大于(单只平均卡玛 * 1.01倍)
VETO_WORST_1Y = -0.15                # 组合一票否决: 任意滚动1年的最差收益不能低于(-15%)
VETO_ZOMBIE_DAYS = 35                # 僵尸基金判定: 距离当前断更天数超过此值则直接判死

# ----------------- 5. 组合综合打分与最终择优输出参数 -----------------
TAIL_RISK_SENSITIVITY = 2.5          # 尾部风险敏感度(对数得分惩罚系数,越大对极端下跌的惩罚越重)
TIME_SATURATION_DAYS = DEFAULT_MAX_HISTORY # 时间分饱和天数(回测天数接近此值则不继续奖励加分)
SCORE_DC_HIGH = 0.8                  # 下行相关性高位惩罚线(大于0.8扣分极重)
SCORE_DC_MID = 0.5                   # 下行相关性中位惩罚线(0.5~0.8按线性平滑扣分)
CFG_FINAL_FOF_MIN_CAGR = 1.0         # 最终结果输出门槛: 复合组合年化筛选底线

# ----------------- 6. 数据质检与黑名单清洗界限 -----------------
JUMP_HARD_LIMIT = 0.315              # 脏数据物理上限: 单日涨跌幅超31.5%无脑击杀(覆盖各板涨停并加缓冲)
JUMP_SOFT_LIMIT = 0.06               # 固收异常上限: 疑似固收类单日涨跌超6%启动复查
FIXED_INCOME_VOL = 0.006             # 固收判定波动率: 历史波动率小于0.6%则视为固收

# 手工维护的主动债及问题基金黑名单(防久期与信用暴露，防止不适合做被动FOF的基金混入)
CFG_ACTIVE_FUNDS_BLACKLIST = [
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

# ----------------- 7. 算力保护与多进程系统级配置 -----------------
CFG_GLOBAL_MAX_WORKERS = 30          # 并行进程池的最大可用进程数(依机器核心数调整)
CFG_DIM_UPGRADE_TARGET_COMBOS = 50000000 # 升维阀门: 理论组合数膨胀超此值时动态截留精英种子(防内存撑爆)
PERTURBATION_SEEDS = [1024, 2048, 4096]  # 权重扰动测试种子集(检查组合稳健性防过拟合)
BATCH_SIZE = 200000                  # 每批次写入 Parquet 的缓冲池大小
CHUNK_SIZE = 2000                    # 每个进程 worker 一次处理的组合切片大小
# ====================================================================

# 子进程全局容器(由 initializer 锚定,避免主进程大 DataFrame 反复序列化)
WORKER_MASTER_DF = None

# 输出列定义
OUTPUT_COLUMNS = [
    '组合文件名', 'Start_Date', 'End_Date', 'Total_Days',
    'CAGR', 'Max_Drawdown', 'Max_Recovery_Days', 'Worst_Rolling_1Y_R2',
    'AR1_Coefficient', 'Annualized_Volatility', 'Sharpe_Ratio',
    'Calmar_Ratio', 'Daily_Win_Rate', 'Downside_Correlation',
    'Avg_CAGR', 'Avg_Max_Drawdown', 'Calmar_Baseline', 'Worst_Rolling_1Y_Return',
    'error', 'Total_Score'
]


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


def read_json(json_path):
    """读取 JSON 文件并返回内容, 文件缺失返回空字典。"""
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 JSON 文件 '{json_path}': {e}")


def save_json(json_path, data):
    """原子写入 JSON(先写临时文件再替换), 避免中断产生半截文件。"""
    dir_path = os.path.dirname(json_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    tmp_path = json_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)
    os.replace(tmp_path, json_path)


def create_folders():
    """确保所有落盘目录存在, 缺失即创建。"""
    dirs = ['temp', 'fund_data/nav', 'fund_data/holdings']
    created = [d for d in dirs if not os.path.exists(d)]
    for d in created:
        os.makedirs(d)
    print(f"【目录初始化】完成 | 需创建: {created if created else '无(均已存在)'}")


# ====================================================================
# 数据采集段: 全市场基金拉取 → 可申购过滤 → 逐只落盘净值/重仓股
# ====================================================================
def _get_active_bond_mask(name_series):
    """返回主动债剔除掩码。判据: 命中嫌疑词 且 未命中被动指数护身符 = 主动债。"""
    suspect_words = ['纯债', '短债', '信用债', '收益债', '回报',
                     '双季', '季季', '月月', '添利', '增利', '稳利']
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

    # 模块四: 归因报告
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


def get_active_fund_codes():
    """
    拉取全市场基金, 仅保留申购状态可买入的基金, 输出 {代码: 简称} 字典并落盘映射。
    Why: 已清盘/封闭期的基金无法建仓, 提前剔除避免后续无效抓取。
    """
    create_folders()
    cache_file = 'fund_data/active_fund_codes.csv'

    print("【采集-基础信息】正在拉取全市场基金列表...")
    df_all = ak.fund_name_em()

    print("【采集-申购状态】正在剔除已清盘/当前无法申购的基金...")
    try:
        df_active = ak.fund_open_fund_daily_em()
        buyable_status = ['开放申购', '限大额', '场内交易']  # 散户可买入状态
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

    pd.DataFrame({
        '基金代码': list(fund_dict.keys()),
        '基金简称': list(fund_dict.values()),
        '基金类型': df_filtered['基金类型'].tolist(),
    }).to_csv(cache_file, index=False, encoding='utf-8-sig')
    print(f"  已落盘代码映射 → [{cache_file}]")

    return fund_dict


def fetch_and_save_fund_data(fund_codes, year="2024", test_mode=False, fetch_holdings=False, max_workers=10):
    """遍历基金字典 {code: name}, 抓取净值走势与重仓股并落盘。"""
    all_codes_list = list(fund_codes.keys())
    codes_to_process = all_codes_list[:5] if test_mode else all_codes_list
    mode_tag = '测试模式(前5只)' if test_mode else '全量'
    print(f"\n【采集-下载】启动 | 模式: [{mode_tag}] | 计划处理: [{len(codes_to_process)}] 只")

    # 将原 for 循环内部逻辑提取为独立的单任务函数
    # 借助闭包特性，直接使用外部的 fund_codes, year, fetch_holdings 等变量
    def _process_single_fund(code):
        fund_name = fund_codes[code]

        # 净值走势: 仅当本地无文件，或文件大小为0（处理历史遗留的占位空文件）时才请求
        nav_file = f"fund_data/nav/{code}_nav.csv"
        if not os.path.exists(nav_file) or os.path.getsize(nav_file) == 0:
            try:
                nav_df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
                if not nav_df.empty:
                    nav_df.to_csv(nav_file, index=False, encoding='utf-8-sig')
                # 删除了 else 分支的占位文件生成逻辑
            except Exception:
                # 删除了异常时的占位文件生成逻辑
                tqdm.write(f"【跳过-净值】抓取失败 | 基金: [{code} {fund_name}] | 可能原因: [无净值数据/接口解析异常]")

        # 重仓股: 同样断点续传，并增加对空占位文件的重新拉取判定
        if fetch_holdings:
            holdings_file = f"fund_data/holdings/{code}_holdings_{year}.csv"
            if not os.path.exists(holdings_file) or os.path.getsize(holdings_file) == 0:
                try:
                    holdings_df = ak.fund_portfolio_hold_em(symbol=code, date=year)
                    if not holdings_df.empty:
                        holdings_df.to_csv(holdings_file, index=False, encoding='utf-8-sig')
                    # 删除了 else 分支的占位文件生成逻辑
                except Exception:
                    # 删除了异常时的占位文件生成逻辑
                    tqdm.write(
                        f"【跳过-重仓股】抓取失败 | 基金: [{code} {fund_name}] | 可能原因: [无持仓披露/接口解析异常]")

    # 启用线程池进行 20 并发拉取
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 将所有基金的抓取任务提交到线程池
        futures = [executor.submit(_process_single_fund, code) for code in codes_to_process]

        # 使用 tqdm 包装 as_completed，保持原有的进度条视觉效果
        for _ in tqdm(as_completed(futures), total=len(codes_to_process), desc="下载进度"):
            pass

# ====================================================================
# 净值加工段: 复权净值 → 统计体检 → 单文件流水 → 批量并行汇总
# ====================================================================
def calculate_adj_nav(df):
    """
    由"单位净值 + 日增长率"累乘还原复权净值。
    注意: 接口日增长率以百分数计(1.5 表示 1.5%), 故除以 100。
    ⚠ 保留原口径: 复权因子从首行即计入当日涨幅(cumprod), 首日复权净值=真实净值×(1+r0/100)。
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
    设计为进程池工作单元, 故所有异常在内部吞掉并以空结果兜底, 保证批处理不中断。
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


def judge_fund_df(head_count=CFG_RECENT_DAYS_LIMIT, max_workers=20, force_update=False):
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

    all_csv_set = set(all_csv_files)
    # 排除 _adj 文件自身; 非强制更新时跳过已有 _adj 的原始文件
    files_to_process = [
        f for f in all_csv_files
        if not os.path.basename(f).endswith("_adj.csv")
           and (force_update or (f[:-4] + "_adj.csv") not in all_csv_set)
    ]

    skipped_count = len(all_csv_files) - len(files_to_process)
    print(f"  发现文件: [{len(all_csv_files)}] 个 | 待处理: [{len(files_to_process)}] 个 "
          f"| 跳过(已复权): [{skipped_count}] 个")

    stats = {"processed": 0, "skipped": skipped_count, "error": 0}
    all_results = []

    if files_to_process:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_fund_pipeline, fp, True, '净值日期', '单位净值', head_count)
                       for fp in files_to_process]
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
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
    n_worst = max(CFG_DOWNSIDE_MIN_DAYS, int(len(synth_ret) * CFG_DOWNSIDE_TAIL_RATIO))
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
    if len(merged_nav) < FIXED_MIN_DAY:
        return {"error": f"Common data length < {FIXED_MIN_DAY} days", "Total_Score": 0.0}

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
                            downside_csv='fund_data/fof_evaluation_results_2d_pool3917.csv', max_workers=CFG_GLOBAL_MAX_WORKERS,
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


def filter_fund_pool(df_results, active_cache='fund_data/active_fund_codes.csv',
                     min_annual_return=CFG_BASE_POOL_MIN_CAGR,
                     min_day=FIXED_MIN_DAY,
                     missing_ratio_limit=VETO_MISSING_RATIO,
                     max_zeros_limit=VETO_CONTINUOUS_ZEROS,
                     min_mdd=CFG_FUND_MIN_MDD):
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
        if not (row['missing_ratio'] <= missing_ratio_limit):
            return f"数据缺失率过高, 当前{row['missing_ratio'] * 100:.2f}%, 目标<={missing_ratio_limit * 100:.2f}%。"
        if not (row['max_zeros'] < max_zeros_limit):
            return f"连续零收益异常, 当前最大连续{row['max_zeros']}天, 目标<{max_zeros_limit}天。"
        if not (row['max_drawdown'] > min_mdd):
            return f"最大回撤已击穿底线, 当前{row['max_drawdown'] * 100:.2f}%, 目标>{min_mdd * 100:.2f}%。"
        return ""

    df_filtered['剔除原因'] = df_filtered.apply(_get_reject_reason, axis=1)

    reject_counts = {
        "运行天数不足": df_filtered['剔除原因'].str.startswith("运行天数不足").sum(),
        "年化收益不达标": df_filtered['剔除原因'].str.startswith("年化收益不达标").sum(),
        "数据缺失率过高": df_filtered['剔除原因'].str.startswith("数据缺失率过高").sum(),
        "连续零收益异常": df_filtered['剔除原因'].str.startswith("连续零收益异常").sum(),
        "最大回撤已击穿底线": df_filtered['剔除原因'].str.startswith("最大回撤已击穿底线").sum()
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


def calculate_dynamic_k_by_binary_search(N, candidate_list,
                                         target_min=int(CFG_DIM_UPGRADE_TARGET_COMBOS * 0.9),
                                         target_max=int(CFG_DIM_UPGRADE_TARGET_COMBOS * 1.1),
                                         strict_mode=True):
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


def get_previous_good_combos(prev_dim, base_pool_codes_set, target_dim, target_max_combos=CFG_DIM_UPGRADE_TARGET_COMBOS):
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
            print(f"[升维/读取] 文件:[{os.path.basename(pf)}] | 读取行数:[{len(df):,}] | "
                  f"通过卡玛门槛:[{len(df) / total_read_rows * 100:.2f}%]")

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
# FOF 择优段: 多维加载 → 白名单过滤 → 复合打分 → 生成可读组合名
# ====================================================================
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
    original_size = len(df)
    df_work = df.copy()
    df_work['基金代码'] = df_work['基金代码'].astype(str).str.split('.').str[0].str.zfill(6)

    matched_names = set()
    mask_is_blacklisted = pd.Series(False, index=df_work.index)
    for bad_name in CFG_ACTIVE_FUNDS_BLACKLIST:
        current_match = df_work['基金简称'].astype(str).str.contains(bad_name, regex=False, na=False)
        if current_match.any():
            matched_names.add(bad_name)
            mask_is_blacklisted = mask_is_blacklisted | current_match

    unmatched_names = [n for n in CFG_ACTIVE_FUNDS_BLACKLIST if n not in matched_names]
    blacklisted_codes = df_work[mask_is_blacklisted]['基金代码'].unique().tolist()

    print("【黑名单提取报告】")
    print(f"  检索词: [{len(CFG_ACTIVE_FUNDS_BLACKLIST)}] 个 | 输入行: [{original_size}] | "
          f"命中行: [{int(mask_is_blacklisted.sum())}] | 返回代码: [{len(blacklisted_codes)}] 个")
    if unmatched_names:
        print(f"  【提示】未匹配到的黑名单词 [{len(unmatched_names)}] 个(可能已清盘或改名):")
        for name in unmatched_names:
            print(f"    - {name}")
    else:
        print("  ✅ 所有黑名单词均已成功匹配。")

    return blacklisted_codes


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
                             active_code_path='fund_data/active_fund_codes.csv',
                             dims=tuple(range(2, CFG_MAX_DIMENSION + 1)),
                             min_days=CFG_RECENT_DAYS_LIMIT,
                             min_cagr=CFG_FINAL_FOF_MIN_CAGR,
                             VALID_FUND_CODES=[]):
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
    all_df_filter['calmar_diff'] = all_df_filter['Calmar_Ratio'] - all_df_filter['Calmar_Baseline']

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
    return all_df_filter


# ====================================================================
# 回测主流程: 初始化 Base Pool → 维度爬坡评估
# ====================================================================
def run_backest_process():
    """进行回测: 生成唯一 Base Pool, 逐维(2→MAX)裂变评估并落盘结果。"""
    RESULT_CSV = 'fund_data/all_funds_result.csv'
    CORR_CSV = 'fund_data/fund_correlations_300.csv'
    DOWNSIDE_CSV = 'fund_data/fof_evaluation_results_2d_pool3917.csv'

    _warmup_numba()

    # 步骤 1: 生成唯一的 Base Pool
    log("[主流程/初始化] 扫描汇总文件, 生成唯一基础池(Base Pool)...")
    if not os.path.exists(RESULT_CSV):
        log(f"[主流程/初始化] 未找到汇总文件 | 文件:[{RESULT_CSV}] (流程终止)", level='ERR')
        exit(0)

    df_results = pd.read_csv(RESULT_CSV)
    df_filtered = filter_fund_pool(df_results, min_annual_return=CFG_BASE_POOL_MIN_CAGR, min_day=FIXED_MIN_DAY)
    if df_filtered.empty:
        log("[主流程/初始化] 符合基础要求的基金数量为 0, 流程退出。", level='WARN')
        exit(0)

    matched_downside = glob.glob(DOWNSIDE_CSV)
    ACTUAL_DOWNSIDE_CSV = (matched_downside[0] if matched_downside
                           else 'fund_data/fof_evaluation_results_2d_downside_300.csv')

    global_corr_matrix, global_downside_corr_matrix = precompute_correlations(
        result_csv=RESULT_CSV, corr_csv=CORR_CSV, downside_csv=ACTUAL_DOWNSIDE_CSV,
        max_workers=CFG_GLOBAL_MAX_WORKERS, df_filtered=df_filtered)

    if global_corr_matrix.empty:
        base_pool_files = df_filtered['adj_nav_file'].dropna().tolist()
    else:
        log("[主流程/去相关] 启动相关性优胜劣汰过滤...")
        base_pool_files = _greedy_correlation_filter(
            df_filtered, global_corr_matrix, global_downside_corr_matrix, CFG_CORR_THRESHOLD, CFG_DOWNSIDE_CORR_THRESHOLD)

    master_df, base_pool_codes = _build_master_matrix(base_pool_files, CFG_GLOBAL_MAX_WORKERS)
    base_pool_codes_set = set(base_pool_codes)
    print(base_pool_codes)
    save_json("base_pool_codes.json", base_pool_codes)

    final_fund_count = len(base_pool_codes)

    print("\n" + "=" * 50)
    log(f"[主流程/Base Pool] 基础基金池已确定 | 最终保留:[{final_fund_count}] 只")
    print("=" * 50 + "\n")

    if final_fund_count < 2:
        log("[主流程/Base Pool] 基金数量不足以生成 2 维组合, 流程退出。", level='WARN')
        exit(0)

    # 步骤 2: 维度爬坡循环 (N = 2 起逐级向上)
    N = 2
    while N <= CFG_MAX_DIMENSION:
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
                N - 1, base_pool_codes_set, target_dim=N, target_max_combos=CFG_DIM_UPGRADE_TARGET_COMBOS)
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

        # 先判空再打印占比, 避免 total_combos=0 时的 ZeroDivisionError
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
        with ProcessPoolExecutor(max_workers=CFG_GLOBAL_MAX_WORKERS,
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


# ====================================================================
# 主入口: 采集 → 加工 → 回测 → 择优统计
# ====================================================================
if __name__ == '__main__':
    # ---- 步骤 1: 清空历史数据, 重新采集原始净值 ----
    if os.path.exists('fund_data'):
        shutil.rmtree('fund_data')
    active_codes = get_active_fund_codes()
    fetch_and_save_fund_data(active_codes, year=CFG_HOLDINGS_YEAR, test_mode=False, max_workers=1)
    judge_fund_df(head_count=CFG_RECENT_DAYS_LIMIT, max_workers=CFG_GLOBAL_MAX_WORKERS, force_update=True)

    # ---- 步骤 2: 维度爬坡回测 ----
    run_backest_process()

    # ---- 步骤 3: FOF 择优统计 ----
    valid_fund_codes = read_json("base_pool_codes.json")
    fof_candidates = analyze_fof_combinations(VALID_FUND_CODES=valid_fund_codes)