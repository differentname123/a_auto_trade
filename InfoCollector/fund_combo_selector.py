# =====================================================================================
# 【功能摘要】
#   公募基金一体化量化筛选与FOF组合构建管线（单文件闭环版）。从全市场数据采集、本地清洗复权、
#   相关性筛除，到多维度的组合裂变测算与最终打分，提供端到端的基金组合优选方案。
#
# 【输入数据】
#   1. 在线数据源 (akshare): 全市场基金基础信息表、实时申购状态、历史单位净值走势、历史重仓股明细。
#   2. 本地缓存 (CSV/Parquet): 存量的基金净值文件 (fund_data/nav/*.csv)、以及历史多维组合测算的
#      断点结果文件 (fund_data/*.parquet)。
#
# 【数据流转/交互】
#   1. [采集段]: akshare 拉取全市场基金 → 结合白名单过滤不可申购标的 → 并发下载净值与重仓股落盘 CSV。
#   2. [加工段]: 读取净值 CSV → 截取近 N 日数据 → 计算复权净值累乘 → 提取年化/回撤等体检指标 → 输出全市场体检汇总表。
#   3. [清洗段]: 读取体检汇总表 → 剔除未达基准(天数/收益/回撤)基金 → 载入全局相关性矩阵剔除同质化标的 → 锁定高优 Base Pool。
#   4. [评估段]: 提取 Base Pool 净值生成纯整型 Master DataFrame → Numba 模拟资金曲线(支持降维/卡玛等一票否决) →
#                基于 Apriori 算法逐维(2D->ND)生成组合种子并多进程并行计算 → 结果追加至 Parquet。
#   5. [择优段]: 加载各维度 Parquet 结果 → 按主动债黑名单进行安全过滤 → 综合打分(兼顾下行相关性惩罚) → 转化为可读组合名输出。
#
# 【输出数据】
#   1. 过程账本: fund_data/all_funds_result.csv (体检汇总)、fund_data/base_pool_rejection_reasons.csv (淘汰审计日志)。
#   2. 缓存体系: fund_data/base_pool_codes.json (当前高优池)、fund_data/computed_combos_*_global_cache.parquet (防重复计算缓存)。
#   3. 终态资产: fund_data/fof_evaluation_results_{N}d_*.parquet (评估明细)，内存返回包含最终排序组合的 DataFrame。
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
# [全局配置区] 核心回测时间与运行范围配置
# ====================================================================
CFG_MAX_DIMENSION = 3
CFG_RECENT_DAYS_LIMIT = 60
CFG_HOLDINGS_YEAR = "2026"

TRADING_DAYS_PER_YEAR = 252
DEFAULT_REBALANCE_DAYS = 30
DEFAULT_MAX_HISTORY = 5 * TRADING_DAYS_PER_YEAR
ROLLING_WINDOW = TRADING_DAYS_PER_YEAR
ROLLING_STEP = 21

FIXED_MIN_DAY = 40
CFG_BASE_POOL_MIN_CAGR = 0.5
CFG_FUND_MIN_MDD = -0.7

CFG_CORR_THRESHOLD = 0.95
CFG_DOWNSIDE_CORR_THRESHOLD = 0.75
CFG_DOWNSIDE_TAIL_RATIO = 0.05
CFG_DOWNSIDE_MIN_DAYS = 5

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

TAIL_RISK_SENSITIVITY = 2.5
TIME_SATURATION_DAYS = DEFAULT_MAX_HISTORY
SCORE_DC_HIGH = 0.8
SCORE_DC_MID = 0.5
CFG_FINAL_FOF_MIN_CAGR = 1.0

JUMP_HARD_LIMIT = 0.315
JUMP_SOFT_LIMIT = 0.06
FIXED_INCOME_VOL = 0.006

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

CFG_GLOBAL_MAX_WORKERS = 30
CFG_DIM_UPGRADE_TARGET_COMBOS = 50000000
PERTURBATION_SEEDS = [1024, 2048, 4096]
BATCH_SIZE = 200000
CHUNK_SIZE = 2000

WORKER_MASTER_DF = None

OUTPUT_COLUMNS = [
    '组合文件名', 'Start_Date', 'End_Date', 'Total_Days',
    'CAGR', 'Max_Drawdown', 'Max_Recovery_Days', 'Worst_Rolling_1Y_R2',
    'AR1_Coefficient', 'Annualized_Volatility', 'Sharpe_Ratio',
    'Calmar_Ratio', 'Daily_Win_Rate', 'Downside_Correlation',
    'Avg_CAGR', 'Avg_Max_Drawdown', 'Calmar_Baseline', 'Worst_Rolling_1Y_Return',
    'error', 'Total_Score'
]


# ====================================================================
# [系统基建] 日志与文件工具
# ====================================================================
def log(msg, level="INFO"):
    """
    统一结构化日志入口。强制聚合语意，拒绝碎片化打印。
    What & Why: 通过时间戳与级别前缀，让控制台输出兼顾监控程序的易读性与排障的清晰度。
    """
    time_str = datetime.now().strftime('%H:%M:%S')
    prefix = "" if level == "INFO" else f"[{level}] "
    print(f"[{time_str}] {prefix}{msg}")


def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def extract_fund_code(filepath):
    match = re.search(r'(\d{6})', os.path.basename(filepath))
    return match.group(1) if match else os.path.basename(filepath).replace('.csv', '')


def normalize_combo_tuple(combo_str, sep='_'):
    parts = [int(str(x).strip()) for x in str(combo_str).split(sep)]
    return tuple(sorted(parts))


def read_json(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            log(f"[系统/读取JSON] 解析失败 | 文件: [{json_path}] | 异常原因: [文件格式损坏, {e}]", level="ERROR")
            return {}


def save_json(json_path, data):
    dir_path = os.path.dirname(json_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    tmp_path = json_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)
    os.replace(tmp_path, json_path)


def create_folders():
    dirs = ['temp', 'fund_data/nav', 'fund_data/holdings']
    created = []
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            created.append(d)
    log(f"[系统/初始化] 目录挂载检查 | 需新建目录: {created if created else '[均已存在]'} | 结果: [完成]")


def _init_worker(master_df):
    """
    进程池初始化钩子。
    What & Why: 将只读的全局大 DataFrame 锚定到各 worker 进程空间，避免在后续序列化传参中撑爆内存。
    """
    global WORKER_MASTER_DF
    WORKER_MASTER_DF = master_df


# ====================================================================
# [采集段] 外部数据获取与本地落盘
# ====================================================================
def _get_active_bond_mask(name_series):
    suspect_words = ['纯债', '短债', '信用债', '收益债', '回报', '双季', '季季', '月月', '添利', '增利', '稳利']
    shield_words = ['指数', '中债', '彭博', '上清所', '国开', '政金', '农发']
    is_suspect = name_series.str.contains('|'.join(suspect_words), regex=True, na=False)
    has_shield = name_series.str.contains('|'.join(shield_words), regex=True, na=False)
    return is_suspect & (~has_shield)


def filter_fund_universe(df, type_col='基金类型', name_col='基金简称', remove_c_class=True):
    """
    清洗大盘基金底库。
    What & Why: 剔除会污染量化回测的混合型、纯债以及冗余的C/E份额，仅留下高纯度的候选池以提高评估准确率。
    """
    if df is None or df.empty:
        log("[采集/底池清洗] 输入数据空 | 结果: [跳过清洗]", level="WARN")
        return df

    res_df = df.copy()
    total_before = len(res_df)
    clean_stats = {}

    type_filter_rules = {
        "1.混合型(仓位飘忽)": ['混合型-灵活', '混合型-偏股', '混合型-偏债', '混合型-平衡', '混合型-绝对收益',
                               'QDII-混合偏股', 'QDII-混合债', 'QDII-混合灵活', 'QDII-混合平衡'],
        "2.假纯债(含股票)": ['债券型-混合二级', '债券型-混合一级'],
        "3.纯股票(噪音大)": ['股票型', 'QDII-普通股票'],
        "4.货币类(拉低收益)": ['货币型-普通货币', '货币型-浮动净值'],
        "5.FOF类(防套娃)": ['FOF-稳健型', 'FOF-进取型', 'FOF-均衡型', 'QDII-FOF'],
    }

    if type_col in res_df.columns:
        res_df[type_col] = res_df[type_col].fillna('').str.strip()
        for reason, types in type_filter_rules.items():
            mask = res_df[type_col].isin(types)
            if mask.sum() > 0:
                clean_stats[reason] = int(mask.sum())
            res_df = res_df[~mask]

    if name_col in res_df.columns:
        active_bond_mask = _get_active_bond_mask(res_df[name_col])
        if active_bond_mask.sum() > 0:
            clean_stats["6.主动型债券(防暴雷)"] = int(active_bond_mask.sum())
        res_df = res_df[~active_bond_mask]

    if remove_c_class and name_col in res_df.columns:
        before_ac = len(res_df)
        res_df['base_name'] = res_df[name_col].str.replace(r'[ABCDEHIY]$', '', regex=True)
        res_df = res_df.sort_values(by=name_col).drop_duplicates(subset=['base_name'], keep='first')
        res_df = res_df.drop(columns=['base_name'])
        ac_count = before_ac - len(res_df)
        if ac_count > 0:
            clean_stats["7.冗余份额去重(剔除C/E)"] = ac_count

    total_after = len(res_df)
    total_cleaned = total_before - total_after

    report_lines = [
        f"[采集/底池清洗] 概览 | 初始池: [{total_before:,}]只 | 净余: [{total_after:,}]只 | 剔除: [{total_cleaned:,}]只"]
    for reason, count in sorted(clean_stats.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"    - {reason}: 剔除 [{count:,}] 只")
    log("\n".join(report_lines))

    return res_df.reset_index(drop=True)


def get_active_fund_codes():
    """
    全市场白名单抓取。
    What & Why: 确保进入管线的基金在现实中是“可申购的”，避免推荐一堆处于封闭期或已清盘的僵尸基金。
    """
    create_folders()
    cache_file = 'fund_data/active_fund_codes.csv'

    log("[采集/全市场名录] 正在拉取在线数据 | 接口: [ak.fund_name_em]")
    df_all = ak.fund_name_em()

    try:
        df_active = ak.fund_open_fund_daily_em()
        buyable_status = ['开放申购', '限大额', '场内交易']
        df_buyable = df_active[df_active['申购状态'].isin(buyable_status)]
        active_codes = set(df_buyable['基金代码'].astype(str).str.zfill(6).tolist())
        log(f"[采集/申购状态] 清洗不可交易基金 | 实时名单: [{len(df_active)}]只 | 符合申购态: [{len(df_buyable)}]只 | 结果: [成功]")
    except Exception as e:
        log(f"[采集/申购状态] 拉取实时状态失败, 业务降级为全量池 | 异常原因: [接口阻断或数据变动, {e}]", level="WARN")
        active_codes = set(df_all['基金代码'].astype(str).str.zfill(6).tolist())

    df_filtered = df_all[df_all['基金代码'].isin(active_codes)].copy()
    fund_dict = dict(zip(df_filtered['基金代码'], df_filtered['基金简称']))
    log(f"[采集/底池生成] 可申购基金代码映射完成 | 总数: [{len(fund_dict)}]只 | 落盘路径: [{cache_file}]")

    pd.DataFrame({
        '基金代码': list(fund_dict.keys()),
        '基金简称': list(fund_dict.values()),
        '基金类型': df_filtered['基金类型'].tolist(),
    }).to_csv(cache_file, index=False, encoding='utf-8-sig')

    return fund_dict


def fetch_and_save_fund_data(fund_codes, year="2024", test_mode=False, fetch_holdings=False, max_workers=10):
    """
    高并发底层数据下载。
    What & Why: 利用多线程池批量将云端数据转化为本地断点续传的 CSV 文件集，剥离网络耗时对计算链路的影响。
    """
    all_codes_list = list(fund_codes.keys())
    codes_to_process = all_codes_list[:5] if test_mode else all_codes_list
    mode_tag = '测试模式(前5只)' if test_mode else '全量'
    log(f"[采集/批量下载] 启动净值下载任务 | 运行模式: [{mode_tag}] | 队列总数: [{len(codes_to_process)}]只")

    def _process_single_fund(code):
        fund_name = fund_codes[code]
        nav_file = f"fund_data/nav/{code}_nav.csv"

        if not os.path.exists(nav_file) or os.path.getsize(nav_file) == 0:
            try:
                nav_df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
                if not nav_df.empty:
                    nav_df.to_csv(nav_file, index=False, encoding='utf-8-sig')
            except Exception as e:
                pass  # 采用静默失败，后续体检流程会自动拦截空文件

        if fetch_holdings:
            holdings_file = f"fund_data/holdings/{code}_holdings_{year}.csv"
            if not os.path.exists(holdings_file) or os.path.getsize(holdings_file) == 0:
                try:
                    holdings_df = ak.fund_portfolio_hold_em(symbol=code, date=year)
                    if not holdings_df.empty:
                        holdings_df.to_csv(holdings_file, index=False, encoding='utf-8-sig')
                except Exception:
                    pass

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_single_fund, code) for code in codes_to_process]
        for _ in tqdm(as_completed(futures), total=len(codes_to_process), desc="并行下载"):
            pass

    log(f"[采集/批量下载] 执行完毕 | 尝试获取完成")


# ====================================================================
# [加工段] 本地数据清洗、复权与基础体检
# ====================================================================
def calculate_adj_nav(df):
    res_df = df.copy()
    if res_df.empty:
        return res_df
    res_df['日增长率'] = pd.to_numeric(res_df['日增长率'], errors='coerce').fillna(0)
    res_df['复权因子'] = (1 + res_df['日增长率'] / 100).cumprod()
    first_day_nav = res_df['单位净值'].dropna().iloc[0]
    res_df['复权净值'] = res_df['复权因子'] * first_day_nav
    return res_df


def calculate_fund_stats(df, fund_name="Unknown", date_col='净值日期', nav_col='单位净值'):
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

    if '日增长率' in temp_df.columns:
        rets = pd.to_numeric(temp_df.loc[real_start:real_end, '日增长率'], errors='coerce').fillna(0)
    else:
        rets = active_series.ffill().pct_change().fillna(0.0)

    is_zero = (rets.abs() < 1e-8).astype(int)
    max_zeros = is_zero.groupby((is_zero == 0).cumsum()).sum().max()
    result["max_zeros"] = int(max_zeros) if pd.notna(max_zeros) else 0

    return result


def process_fund_pipeline(file_path, save_qualified=True, date_col='净值日期', nav_col='单位净值', head_count=300):
    fund_name = os.path.basename(file_path).split('_')[0]
    try:
        df = pd.read_csv(file_path)
    except Exception:
        df = pd.DataFrame()

    if not df.empty and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = (df.sort_values(by=date_col, ascending=False)
              .head(head_count).sort_values(by=date_col).reset_index(drop=True))

    result = calculate_fund_stats(df, fund_name=fund_name, date_col=date_col, nav_col=nav_col)

    if not df.empty and date_col in df.columns and nav_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).reset_index(drop=True)
        qualified_df = calculate_adj_nav(df)

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

    holdings_files = sorted(glob.glob(f"fund_data/holdings/{fund_name}_holdings_*.csv"), reverse=True)
    if holdings_files:
        try:
            hdf = pd.read_csv(holdings_files[0])
            if '股票名称' in hdf.columns:
                result["top_10_holdings"] = ",".join(hdf.head(10)['股票名称'].dropna().astype(str).tolist())
        except Exception:
            pass

    return result


def judge_fund_df(head_count=CFG_RECENT_DAYS_LIMIT, max_workers=20, force_update=False):
    """
    离线批量加工与体检。
    What & Why: 在多进程环境下将生肉(原始净值)转化为熟肉(复权数据与体检指标)，为高维排列组合提供干净可信的底料。
    """
    target_dir = 'fund_data/nav'
    log(f"[加工/扫描目录] 准备执行体检任务 | 路径: [{target_dir}]")
    if not os.path.exists(target_dir):
        log("[加工/扫描目录] 依赖路径缺失，业务终止 | 原因: [需优先执行采集段]", level="ERROR")
        return

    all_csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not all_csv_files:
        log("[加工/扫描目录] 目录下无文件，业务终止", level="ERROR")
        return

    all_csv_set = set(all_csv_files)
    files_to_process = [
        f for f in all_csv_files
        if not os.path.basename(f).endswith("_adj.csv")
           and (force_update or (f[:-4] + "_adj.csv") not in all_csv_set)
    ]

    skipped_count = len(all_csv_files) - len(files_to_process)
    log(f"[加工/任务分配] 并发体检任务解析完成 | 总计: [{len(all_csv_files)}]个 | 需处理: [{len(files_to_process)}]个 | 跳过已复权: [{skipped_count}]个")

    stats = {"processed": 0, "skipped": skipped_count, "error": 0}
    all_results = []

    if files_to_process:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_fund_pipeline, fp, True, '净值日期', '单位净值', head_count)
                       for fp in files_to_process]
            for future in tqdm(as_completed(futures), total=len(futures), desc="净值复权与体检"):
                try:
                    result = future.result()
                    all_results.append(result)
                    stats["processed" if result.get("adj_nav_file") else "error"] += 1
                except Exception:
                    stats["error"] += 1

    # ================= 仅修改以下落盘与汇总逻辑 =================
    output_csv = "fund_data/all_funds_result.csv"
    final_len = 0  # 用于记录最终落盘的总长度

    # 只要有新结果，或者历史文件本身存在，就执行落盘/读取逻辑
    if all_results or os.path.exists(output_csv):
        df_new = pd.DataFrame(all_results) if all_results else pd.DataFrame()

        # 当【非强制更新】且【历史文件存在】时，进行增量合并
        if not force_update and os.path.exists(output_csv):
            try:
                df_old = pd.read_csv(output_csv)
                if not df_new.empty:
                    # 拼接历史与新增数据
                    df_final = pd.concat([df_old, df_new], ignore_index=True)
                    # 容错：如果结果里有'基金代码'，则基于基金代码去重，保留最新的一条
                    if '基金代码' in df_final.columns:
                        df_final = df_final.drop_duplicates(subset=['基金代码'], keep='last')
                else:
                    df_final = df_old
            except Exception as e:
                log(f"[加工/合并] 读取历史结果失败，回退为覆写模式。原因: {e}", level="WARNING")
                df_final = df_new
        else:
            # 强制更新或首次运行时，直接使用当次全部计算结果
            df_final = df_new

        # 落盘并获取最终行数
        if not df_final.empty:
            df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')
            final_len = len(df_final)
            log(f"[加工/汇总落盘] 体检数据整合完毕 | 落盘地址: [{output_csv}] | 当前落盘记录数: [{final_len}]条")

    # 在最终报告中追加打印库容
    log(f"[加工/汇总报告] 流水线执行完毕 | 成功: [{stats['processed']}]个 | 失败(含空): [{stats['error']}]个 | 跳过: [{stats['skipped']}]个 | 最终有效总库容: [{final_len}]个")


# ====================================================================
# [缓存系统] Parquet 整型化全局防重记忆库
# ====================================================================
def _save_set_to_parquet(combo_set, dim, filepath):
    try:
        if not combo_set:
            return
        combo_list = list(combo_set)
        cols = [[combo[i] for combo in combo_list] for i in range(dim)]
        arrays = [pa.array(col, type=pa.int32()) for col in cols]
        names = [f'dim_{i}' for i in range(dim)]
        table = pa.Table.from_arrays(arrays, names=names)
        pq.write_table(table, filepath, compression='zstd')
        del combo_list, cols, arrays, table
    except Exception as e:
        log(f"[系统/持久化] 组合字典写入失败 | 目标文件: [{os.path.basename(filepath)}] | 异常原因: [可能磁盘满或无权限, {e}]",
            level="ERROR")


def load_or_init_computed_set(curr_dim):
    cache_file = f'fund_data/computed_combos_{curr_dim}d_global_cache.parquet'
    old_pkl_file = f'fund_data/computed_combos_{curr_dim}d_global_cache.pkl'
    parquet_files = glob.glob(f'fund_data/fof_evaluation_results_{curr_dim}d_*.parquet')

    computed_set = set()
    cache_time = 0

    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        try:
            table = pq.read_table(cache_file)
            cols = [table.column(f'dim_{i}').to_pylist() for i in range(curr_dim)]
            computed_set = set(zip(*cols))
            log(f"[缓存/命中] 加载【{curr_dim}D】主缓存成功 | 容量: [{len(computed_set):,}]个组合")
        except Exception as e:
            log(f"[缓存/异常] Parquet 缓存解析失败, 将重构 | 维度: [【{curr_dim}D】] | 异常原因: [数据损坏或版本不符, {e}]",
                level="WARN")
            cache_time = 0

    elif os.path.exists(old_pkl_file):
        try:
            with open(old_pkl_file, 'rb') as f:
                old_set = pickle.load(f)
            for combo in old_set:
                computed_set.add(tuple(int(x) for x in combo))
            _save_set_to_parquet(computed_set, curr_dim, cache_file)
            os.rename(old_pkl_file, old_pkl_file + ".bak")
            cache_time = os.path.getmtime(cache_file)
            log(f"[缓存/升级] 成功将旧版 PKL 升级为 Parquet | 维度: [【{curr_dim}D】] | 容量: [{len(computed_set):,}]个组合")
        except Exception as e:
            log(f"[缓存/异常] 历史 PKL 升级失败 | 维度: [【{curr_dim}D】] | 异常原因: [旧格式不兼容, 将废弃, {e}]",
                level="WARN")

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

            before_len = len(computed_set)
            computed_set.update(file_temp_set)
            added = len(computed_set) - before_len

            if added > 0:
                updated = True
                log(f"[缓存/增量] 回溯未录入结果文件 | 文件: [{os.path.basename(pf)}] | 新吸收: [{added:,}]个组合")
        except Exception as e:
            log(f"[缓存/异常] 读取结果文件失败跳过 | 文件: [{os.path.basename(pf)}] | 异常原因: [文件损坏或被占用, {e}]",
                level="ERROR")

    if updated or (not os.path.exists(cache_file) and computed_set):
        _save_set_to_parquet(computed_set, curr_dim, cache_file)
        log(f"[缓存/持久化] 全局记忆库已更新 | 维度: [【{curr_dim}D】] | 总记录数: [{len(computed_set):,}]")

    return computed_set, cache_file


def _update_cache_from_new_file(output_file, cache_file, computed_set):
    if not output_file or not os.path.exists(output_file):
        return
    cache_time = os.path.getmtime(cache_file) if os.path.exists(cache_file) else 0
    if os.path.getmtime(output_file) <= cache_time:
        return

    try:
        if os.path.getsize(output_file) <= 100:
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

        if added > 0:
            dim = len(next(iter(computed_set))) if computed_set else 0
            if dim > 0:
                _save_set_to_parquet(computed_set, dim, cache_file)
            log(f"[缓存/回写] 动态热更新记忆库 | 本次注入: [{added:,}]个 | 全局容量: [{len(computed_set):,}]个")
    except Exception as e:
        log(f"[缓存/异常] 动态更新缓存库失败 | 异常原因: [文件读写冲突, {e}]", level="ERROR")


# ====================================================================
# [核心引擎] Numba 蒙特卡洛与指标测算区
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


def evaluate_fof_portfolio_fast(merged_nav, rebalance_days=DEFAULT_REBALANCE_DAYS,
                                max_history_days=DEFAULT_MAX_HISTORY,
                                max_mdd_limit=VETO_MAX_MDD, hurdle_rate=VETO_HURDLE_RATE):
    """
    分层一票否决与得分评估引擎。
    What & Why: 根据极严苛的业务指标（最大回撤、连续跌幅、虚假平滑、偏离基准卡玛）逐层短路拦截不合格组合，节省昂贵的关联性与夏普算力。
    """
    n_funds = merged_nav.shape[1]
    raw_index = merged_nav.index
    valid_mask = merged_nav.notna().all(axis=1)

    if not valid_mask.any():
        return {"error": "无有效重叠时间段", "Total_Score": 0.0}

    merged_nav = merged_nav.loc[valid_mask.idxmax():valid_mask[::-1].idxmax()]
    if len(merged_nav) > max_history_days:
        merged_nav = merged_nav.iloc[-max_history_days:]
    if len(merged_nav) < FIXED_MIN_DAY:
        return {"error": f"有效重叠天数不足 {FIXED_MIN_DAY} 天", "Total_Score": 0.0}

    max_missing_ratio = float((merged_nav.isna().sum() / len(merged_nav)).max())
    if (raw_index[-1] - merged_nav.index[-1]).days > VETO_ZOMBIE_DAYS:
        return {"error": f"僵尸基金断更熔断", "Total_Score": 0.0}

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

        if (max_missing_ratio > VETO_MISSING_RATIO) or (max_continuous_zeros > VETO_CONTINUOUS_ZEROS):
            metrics["VETO_Data_Distortion"] = True
            return metrics, 0.0

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

        worst_1y = _compute_worst_rolling_1y(synth_eq.values, n_days)
        metrics.update({'Worst_Rolling_1Y_Return': worst_1y})
        if worst_1y < VETO_WORST_1Y:
            metrics["VETO_Worst_1Y_Crash"] = True
            return metrics, 0.0

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


def _worker_process_combo(combo_codes):
    global WORKER_MASTER_DF
    combo_name = "_".join([f"{c:06d}" for c in combo_codes])
    try:
        merged_nav = WORKER_MASTER_DF[list(combo_codes)]
        result = evaluate_fof_portfolio_fast(merged_nav)
        result['组合文件名'] = combo_name
        return result
    except Exception as e:
        return {'组合文件名': combo_name, 'error': f"处理异常: {str(e)}", 'Total_Score': 0.0}


def _worker_process_chunk(combo_chunk):
    return [_worker_process_combo(combo) for combo in combo_chunk]


# ====================================================================
# [矩阵计算] 全局数据挂载与关联性清洗
# ====================================================================
def _load_single_nav(filepath):
    try:
        if not os.path.exists(filepath):
            return None
        df = pd.read_csv(filepath)
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        df = df.set_index('净值日期').sort_index()
        df = df[~df.index.duplicated(keep='last')]

        nav_series = df['复权净值'].rename(filepath)
        recent_series = nav_series.iloc[-DEFAULT_MAX_HISTORY:] if len(nav_series) > DEFAULT_MAX_HISTORY else nav_series
        daily_returns = recent_series.ffill().pct_change().dropna()

        if not daily_returns.empty:
            max_jump = float(daily_returns.max())
            if max_jump > JUMP_HARD_LIMIT:
                return None
            if max_jump > JUMP_SOFT_LIMIT:
                normal_returns = daily_returns.sort_values().iloc[:-3]
                normal_vol = float(normal_returns.std())
                if normal_vol < FIXED_INCOME_VOL:
                    return None
        return nav_series
    except Exception:
        return None


def _parallel_load_navs(files, max_workers, desc):
    series_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_single_nav, f) for f in files]
        for future in tqdm(as_completed(futures), total=len(files), desc=desc):
            res = future.result()
            if res is not None:
                series_list.append(res)
    return series_list


def precompute_correlations(result_csv='fund_data/all_funds_result.csv', corr_csv='fund_data/fund_correlations.csv',
                            downside_csv='fund_data/fof_evaluation_results_2d_pool3917.csv',
                            max_workers=CFG_GLOBAL_MAX_WORKERS,
                            downside_corr_csv='fund_data/fund_downside_correlations_300.csv', df_filtered=None):
    log("[相关矩阵/初始化] 开始计算全市场皮尔逊相关性矩阵...")
    downside_corr_matrix = pd.DataFrame()

    if os.path.exists(corr_csv):
        log(f"[相关矩阵/命中] 发现历史相关性缓存 | 缓存路径: [{corr_csv}] | 结果: [直接读取使用]")
        return pd.read_csv(corr_csv, index_col=0), downside_corr_matrix

    if not os.path.exists(result_csv):
        return pd.DataFrame(), downside_corr_matrix

    df_results = pd.read_csv(result_csv)
    if df_filtered is not None and not df_filtered.empty:
        df_results = df_filtered

    files = df_results['adj_nav_file'].dropna().tolist()
    nav_series_list = _parallel_load_navs(files, max_workers, "加载相关性矩阵底仓")
    if not nav_series_list:
        return pd.DataFrame(), downside_corr_matrix

    merged_nav = pd.concat(nav_series_list, axis=1, join='outer').ffill()
    corr_matrix = merged_nav.pct_change().corr()
    ensure_dir(corr_csv)
    corr_matrix.to_csv(corr_csv)

    log(f"[相关矩阵/生成] 矩阵构建完成 | 覆盖基金: [{len(corr_matrix)}]只 | 落盘路径: [{corr_csv}]")
    return corr_matrix, downside_corr_matrix


def _extract_target_codes(df_filtered):
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
    单基入池财务初筛。
    What & Why: 通过刚性条件(存活期、收益下限、回撤底线等)淘汰劣质或高危标的，输出一份包含裁决原因的审计账本。
    """
    if df_results is None or df_results.empty:
        return pd.DataFrame()

    original_count = len(df_results)
    df_filtered = df_results.copy()
    active_filtered_count = original_count
    rejection_records = []

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
                        '基金代码': target_codes.loc[idx], '文件': row.get('adj_nav_file', ''),
                        '筛选阶段': '初筛-申购状态', '剔除原因': '该基金暂不可申购或已下架'
                    })

                df_filtered = df_filtered[is_active].copy()
                active_filtered_count = len(df_filtered)
                log(f"[单基初筛/活体检测] 申购状态比对 | 拦截不可买基金: [{original_count - active_filtered_count}]只")
        except Exception as e:
            log(f"[单基初筛/活体检测] 缓存读取失败跳过检测 | 异常原因: [{e}]", level="WARN")

    if df_filtered.empty:
        ensure_dir('fund_data/base_pool_rejection_reasons.csv')
        pd.DataFrame(rejection_records).to_csv('fund_data/base_pool_rejection_reasons.csv', index=False,
                                               encoding='utf-8-sig')
        return df_filtered

    def _get_reject_reason(row):
        if not (row['total_active_days'] > min_day): return "运行天数不足"
        if not (row['annualized_return'] > min_annual_return): return "年化收益不达标"
        if not (row['missing_ratio'] <= missing_ratio_limit): return "数据缺失率过高"
        if not (row['max_zeros'] < max_zeros_limit): return "连续零收益异常"
        if not (row['max_drawdown'] > min_mdd): return "最大回撤已击穿底线"
        return ""

    df_filtered['剔除原因'] = df_filtered.apply(_get_reject_reason, axis=1)

    rejected_metrics = df_filtered[df_filtered['剔除原因'] != ""]
    target_codes_filtered = _extract_target_codes(df_filtered)
    for idx, row in rejected_metrics.iterrows():
        rejection_records.append({
            '基金代码': target_codes_filtered.loc[idx], '文件': row.get('adj_nav_file', ''),
            '筛选阶段': '初筛-财务硬性指标', '剔除原因': row['剔除原因']
        })

    reason_file = 'fund_data/base_pool_rejection_reasons.csv'
    ensure_dir(reason_file)
    pd.DataFrame(rejection_records).to_csv(reason_file, index=False, encoding='utf-8-sig')

    df_final = df_filtered[df_filtered['剔除原因'] == ""].drop(columns=['剔除原因']).sort_values(by='annualized_return',
                                                                                                 ascending=False)

    report = [
        f"[单基初筛/最终决议] 漏斗筛查完毕",
        f"    - 初始池进入: [{original_count}]只",
        f"    - 通过申购态: [{active_filtered_count}]只",
        f"    - 通过财务面: [{len(df_final)}]只 (作为高优池)"
    ]
    log("\n".join(report))

    return df_final


def _greedy_correlation_filter(df_filtered, corr_matrix, downside_corr_matrix, corr_threshold, downside_corr_threshold):
    """
    贪婪降维去重算法。
    What & Why: 如果两只基金走势高度一致(皮尔逊极高)，保留收益更好的那只即可。借此缩减算力消耗，提升最终配置的分散度。
    """
    selected = []
    action_records = []
    has_downside = downside_corr_matrix is not None and not downside_corr_matrix.empty

    for f in df_filtered['adj_nav_file'].dropna():
        if f not in corr_matrix.columns:
            selected.append(f)
            action_records.append({
                '基金代码': extract_fund_code(f), '文件': f, '筛选阶段': '次筛-相关性',
                '剔除原因': '【入选】矩阵缺数据,从宽保留'
            })
            continue

        is_corr, reason = False, ""
        for sel_f in selected:
            if sel_f in corr_matrix.columns:
                corr_val = corr_matrix.loc[f, sel_f]
                if pd.notna(corr_val) and corr_val > corr_threshold:
                    is_corr, reason = True, f"全天候相关性达[{corr_val:.4f}], 与[{extract_fund_code(sel_f)}]撞车"
                    break
            if has_downside and f in downside_corr_matrix.index and sel_f in downside_corr_matrix.columns:
                dc_val = downside_corr_matrix.loc[f, sel_f]
                if pd.notna(dc_val) and dc_val > downside_corr_threshold:
                    is_corr, reason = True, f"下行相关性达[{dc_val:.4f}], 与[{extract_fund_code(sel_f)}]撞车"
                    break

        if not is_corr:
            selected.append(f)
            action_records.append(
                {'基金代码': extract_fund_code(f), '文件': f, '筛选阶段': '终选', '剔除原因': '【入选】通过相关性检查'})
        else:
            action_records.append(
                {'基金代码': extract_fund_code(f), '文件': f, '筛选阶段': '次筛-相关性', '剔除原因': reason})

    if action_records:
        reason_file = 'fund_data/base_pool_rejection_reasons.csv'
        df_action = pd.DataFrame(action_records)
        df_action['is_selected'] = df_action['筛选阶段'] == '终选'
        df_action = df_action.sort_values(by=['is_selected', '基金代码'], ascending=[False, True]).drop(
            columns=['is_selected'])
        df_action.to_csv(reason_file, mode='a', header=False, index=False, encoding='utf-8-sig')

    return selected


# ====================================================================
# [系统管道] Master 构建与环境预热
# ====================================================================
def _prepare_results_table(batch_results):
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
    log("[基建/引擎保护] Numba JIT 预热 | 动作: [空载试运行] | 结果: [防止子进程编译踩踏]")
    _fast_simulate_path(np.zeros((10, 2), dtype=np.float64), np.array([0.5, 0.5], dtype=np.float64), 30, 0, 10, 2)


def _build_master_matrix(target_files, max_workers):
    log("[系统/核心组装] 提取入选基金构建 Master Dataframe 矩阵...")
    nav_series_list = _parallel_load_navs(target_files, max_workers, "内存列装填")
    master_df = pd.concat(nav_series_list, axis=1, join='outer')
    del nav_series_list

    master_df.columns = [int(extract_fund_code(col)) for col in master_df.columns]
    sorted_codes = sorted(master_df.columns.tolist())
    master_df = master_df[sorted_codes]

    log(f"[系统/核心组装] Master 构建完毕 | 矩阵行数: [{len(master_df)}] | 基金列数: [{len(sorted_codes)}]")
    return master_df, sorted_codes


def _print_final_report(final_fund_count, combo_size, total_combos, skipped, computed, output_parquet, has_output):
    skip_ratio = (skipped / total_combos * 100) if total_combos > 0 else 0.0
    report = [
        f"[回测批处理/维度落幕] 盘口: [{final_fund_count}]只 | 当前维度: 【{combo_size}D】",
        f"    - 理论下发组合: [{total_combos:,}]组",
        f"    - 命中缓存跳过: [{skipped:,}]组 ({skip_ratio:.2f}%)",
        f"    - 最终算力消耗: [{computed:,}]组",
        f"    - 数据落地轨迹: [{output_parquet if has_output else '无新生数据'}]"
    ]
    log("\n".join(report))


# ====================================================================
# [升维发生器] Apriori 组合裂变与智能截流
# ====================================================================
def fast_estimate_combos(N, seeds_list, strict_mode=True):
    seeds_set = set(seeds_list)
    prefix_map = defaultdict(list)
    prefix_len = N - 2
    for combo in seeds_list:
        prefix_map[combo[:prefix_len]].append(combo)

    total_count = 0
    for prefix, combos in prefix_map.items():
        n_combos = len(combos)
        if n_combos < 2: continue
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
                    if not is_elite: continue
                total_count += 1
    return total_count


def calculate_dynamic_k_by_binary_search(N, candidate_list, target_min=int(CFG_DIM_UPGRADE_TARGET_COMBOS * 0.9),
                                         target_max=int(CFG_DIM_UPGRADE_TARGET_COMBOS * 1.1), strict_mode=True):
    """
    智能截留器(二分逼近)。
    What & Why: 面对组合数非线性爆炸，动态寻找最完美的种子录用数量K，使衍生出来的高维组合数正好落在我们设置的安全阀内。
    """
    if len(candidate_list) < 2:
        return len(candidate_list), 0

    high_total = len(candidate_list)
    target_mid = (target_min + target_max) // 2
    init_k = max(2, min(high_total // 2, 1000000))
    count_init = fast_estimate_combos(N, candidate_list[:init_k], strict_mode)

    best_diff = abs(count_init - target_mid)
    best_k, best_count = init_k, count_init

    if target_min <= count_init <= target_max:
        return init_k, count_init

    if count_init < target_mid:
        low, count_low = init_k, count_init
        high, count_high = high_total, None
    else:
        low = 2
        count_low = fast_estimate_combos(N, candidate_list[:low], strict_mode)
        high, count_high = init_k, count_init
        if count_low > target_max: return low, count_low
        if target_min <= count_low <= target_max: return low, count_low

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
                safe_multiplier = min(3.5, max(1.15, math.pow(shortfall_ratio, 1 / E) * 1.05))
            except Exception:
                safe_multiplier = min(2.0, max(1.15, math.pow(shortfall_ratio, 1 / max(1, N)) * 1.05))
        else:
            safe_multiplier = min(2.0, max(1.15, math.pow(shortfall_ratio, 1 / max(1, N)) * 1.05))

        next_k = min(high_total, max(low + 10, int(low * safe_multiplier)))
        count = fast_estimate_combos(N, candidate_list[:next_k], strict_mode)

        if abs(count - target_mid) < best_diff:
            best_diff, best_k, best_count = abs(count - target_mid), next_k, count
        if target_min <= count <= target_max: return next_k, count

        if count > target_max:
            high, count_high = next_k, count
            break
        prev_k, prev_count = low, count_low
        low, count_low = next_k, count

    if count_high is None: return best_k, best_count

    while low <= high:
        if high - low < 1000: break
        try:
            log_low_k, log_high_k = math.log(low), math.log(high)
            log_low_c, log_high_c = math.log(max(1, count_low)), math.log(max(1, count_high))
            ratio = (math.log(target_mid) - log_low_c) / (log_high_c - log_low_c)
            mid_guess = int(math.exp(log_low_k + ratio * (log_high_k - log_low_k)))
            mid = max(low + 1, min(high - 1, mid_guess))
        except Exception:
            mid = (low + high) // 2

        count = fast_estimate_combos(N, candidate_list[:mid], strict_mode)
        diff = abs(count - target_mid)
        if diff < best_diff:
            best_diff, best_k, best_count = diff, mid, count
        if target_min <= count <= target_max: return mid, count

        if count > target_max:
            high, count_high = mid, count
        else:
            low, count_low = mid, count

    return best_k, best_count


def get_previous_good_combos(prev_dim, base_pool_codes_set, target_dim,
                             target_max_combos=CFG_DIM_UPGRADE_TARGET_COMBOS):
    def fast_filter_and_normalize(combo_str):
        items = [int(x.strip()) for x in str(combo_str).split('_')]
        if not base_pool_codes_set.issuperset(items): return None
        return tuple(sorted(items))

    parquet_files = glob.glob(f'fund_data/fof_evaluation_results_{prev_dim}d_*.parquet')
    all_clean_records = []

    for pf in parquet_files:
        try:
            df = pq.read_table(pf, columns=['组合文件名', 'Calmar_Ratio', 'Calmar_Baseline', 'Total_Score',
                                            'CAGR']).to_pandas()
            df['Total_Score'] = df['Total_Score'].fillna(0) if 'Total_Score' in df.columns else 0
            df = df[df['Calmar_Ratio'] > df['Calmar_Baseline']]
            if df.empty: continue

            df['Combo_Tuple'] = df['组合文件名'].apply(fast_filter_and_normalize)
            df = df[df['Combo_Tuple'].notna()]
            if df.empty: continue

            df = df.sort_values(['Total_Score', 'CAGR'], ascending=[False, False]).drop_duplicates(
                subset=['Combo_Tuple'])
            all_clean_records.append(df[['Combo_Tuple', 'Calmar_Ratio', 'Total_Score', 'CAGR']])
        except Exception as e:
            log(f"[升维种子/加载] 异常跳过 | 文件: [{os.path.basename(pf)}] | 原因: [读取或解析出错, {e}]",
                level="WARN")

    if not all_clean_records:
        log("[升维种子/拦截] 尚未获取到任何优良基因，失去向高维繁衍能力", level="WARN")
        return set()

    combined_df = pd.concat(all_clean_records, ignore_index=True)
    pool_only_df = combined_df.sort_values(['Total_Score', 'CAGR'], ascending=[False, False]).drop_duplicates(
        subset=['Combo_Tuple'])

    score_gt_0_df = pool_only_df[pool_only_df['Total_Score'] > 0]
    score_lte_0_df = pool_only_df[pool_only_df['Total_Score'] <= 0]
    full_candidate_df = pd.concat([score_gt_0_df, score_lte_0_df], ignore_index=True)

    dynamic_k, expected_count = calculate_dynamic_k_by_binary_search(
        N=target_dim, candidate_list=full_candidate_df['Combo_Tuple'].tolist(),
        target_min=int(target_max_combos * 0.9), target_max=int(target_max_combos * 1.1), strict_mode=True)

    log(f"[升维种子/截流] 基于目标上限锁定量级 | 录取前排: [{dynamic_k:,}]个精英 | 预计衍生高维组合: [{expected_count:,}]组")
    return set(full_candidate_df.head(dynamic_k)['Combo_Tuple'])


def generate_next_dimension_combos_apriori(N, good_prev_combos, computed_set, strict_mode=False):
    uncomputed_combos = []
    global_skipped, total_combos = 0, 0
    if strict_mode and not isinstance(good_prev_combos, set): good_prev_combos = set(good_prev_combos)

    prefix_map = defaultdict(list)
    prefix_len = N - 2
    for combo in good_prev_combos: prefix_map[combo[:prefix_len]].append(combo)

    for prefix, combos in prefix_map.items():
        n_combos = len(combos)
        if n_combos < 2: continue
        for i in range(n_combos):
            for j in range(i + 1, n_combos):
                t1, t2 = combos[i][-1], combos[j][-1]
                new_combo = prefix + (t1, t2) if t1 < t2 else prefix + (t2, t1)

                if strict_mode:
                    is_elite = True
                    for idx in range(N):
                        if new_combo[:idx] + new_combo[idx + 1:] not in good_prev_combos:
                            is_elite = False;
                            break
                    if not is_elite: continue

                total_combos += 1
                if computed_set is not None and new_combo in computed_set:
                    global_skipped += 1
                else:
                    uncomputed_combos.append(new_combo)

    return uncomputed_combos, total_combos, global_skipped


# ====================================================================
# [终态输出] 最终 FOF 候选组合的翻译与报表
# ====================================================================
def load_and_merge_parquet_by_dim(dimension, data_dir='fund_data', min_days=600, min_score=None, filter_code=None):
    search_pattern = os.path.join(data_dir, f'fof_evaluation_results_{dimension}d_*.parquet')
    file_list = glob.glob(search_pattern)
    if not file_list:
        return pd.DataFrame()

    read_filters = [('Total_Days', '>', min_days)]
    if min_score is not None: read_filters.append(('Total_Score', '>', min_score))

    try:
        df = pd.read_parquet(file_list, engine='pyarrow', filters=read_filters)
        if filter_code: df = df[df['组合文件名'].str.contains(filter_code, na=False)]
    except Exception:
        dfs = []
        for file in file_list:
            try:
                df_part = pd.read_parquet(file, engine='pyarrow', filters=read_filters)
                if not df_part.empty: dfs.append(df_part)
            except Exception:
                pass
        if not dfs: return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)

    return df.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)


def _simplify_fund_name(name):
    if not isinstance(name, str): return name
    name = re.sub(r'\(.*?\)|（.*?）', '', name)
    redundant_words = ['ETF联接', '联接', 'ETF', 'LOF', 'QDII', '灵活配置', '多策略',
                       '混合型', '混合', '指数增强', '增强型', '指数', '发起式', '人民币份额', '人民币', '证券投资基金']
    name = re.sub('|'.join(redundant_words), '', name)
    return re.sub(r'[A-Z]$', '', name)


def analyze_fof_combinations(report_path='fund_data/base_pool_rejection_reasons.csv',
                             active_code_path='fund_data/active_fund_codes.csv',
                             dims=tuple(range(2, CFG_MAX_DIMENSION + 1)),
                             min_days=CFG_RECENT_DAYS_LIMIT-1, min_cagr=CFG_FINAL_FOF_MIN_CAGR,
                             VALID_FUND_CODES=[]):
    """
    人类视角最终报表生成。
    What & Why: 将冰冷的六位数字转化为具有业务含义的中文缩写，融合最终的权重打分并提取带有特定主题（如海外配资）的精选组合池。
    """
    log("[择优/初始化] 启动复合资产解析与重排工序...")
    report_df = pd.read_csv(report_path)
    active_code_df = pd.read_csv(active_code_path)

    overseas_mask = active_code_df['基金类型'].str.contains('QDII|海外', na=False)
    overseas_set = set(active_code_df[overseas_mask]['基金代码'].astype(str).str.zfill(6).tolist())

    report_df = report_df.merge(active_code_df, on='基金代码', how='right')
    all_codes = report_df['文件'].str.extract(r'(\d{6})')[0].dropna().unique().tolist()
    valid_codes = set(str(c).zfill(6) for c in VALID_FUND_CODES)
    invalid_set = set(all_codes) - valid_codes

    all_df_list = []
    for dim in dims:
        df_dim = load_and_merge_parquet_by_dim(dimension=dim, min_days=min_days, min_score=0)
        if df_dim.empty: continue
        df_dim['Total_Score'] = df_dim['Total_Score'] * 10000
        df_dim = df_dim[df_dim['CAGR'] > min_cagr].reset_index(drop=True)
        all_df_list.append(df_dim)

    if not all_df_list:
        log("[择优/阻断] 各层级维度均无满足 CAGR 基线的有效组合", level="ERROR")
        return pd.DataFrame()

    all_df = pd.concat(all_df_list, ignore_index=True).sort_values(by='Total_Score', ascending=False)

    if invalid_set:
        pattern = re.compile(r'\d{6}')
        mask = [invalid_set.isdisjoint(pattern.findall(fn)) if type(fn) is str else False for fn in
                all_df['组合文件名'].tolist()]
        all_df_filter = all_df[mask].reset_index(drop=True)
    else:
        all_df_filter = all_df.copy()

    all_df_filter['score'] = (all_df_filter['CAGR'] * 10) ** 3 * all_df_filter['Total_Score']
    all_df_filter = all_df_filter.sort_values(by='score', ascending=False).reset_index(drop=True)
    all_df_filter['calmar_diff'] = all_df_filter['Calmar_Ratio'] - all_df_filter['Calmar_Baseline']

    padded_active_codes = active_code_df['基金代码'].astype(str).str.zfill(6)
    code_to_name = dict(zip(padded_active_codes, active_code_df['基金简称'].apply(_simplify_fund_name)))

    all_df_filter['基金简称组合'] = [
        '_'.join([str(code_to_name.get(c, c)) for c in fn.split('_')]) if type(fn) is str else fn for fn in
        all_df_filter['组合文件名'].tolist()]
    all_df_filter.insert(0, '基金简称组合', all_df_filter.pop('基金简称组合'))

    tail_cols = ['组合文件名', 'Start_Date', 'End_Date', 'Total_Days']
    cols = [c for c in all_df_filter.columns if c not in tail_cols] + [c for c in tail_cols if
                                                                       c in all_df_filter.columns]
    all_df_filter = all_df_filter[cols]

    dc = all_df_filter['Downside_Correlation']
    all_df_filter['组合数量'] = all_df_filter['组合文件名'].apply(
        lambda x: len(str(x).split('_')) if type(x) is str else 0)
    all_df_filter['score_corr_down'] = all_df_filter['Total_Score'] / (dc + 1)
    all_df_filter['score_corr_down1'] = all_df_filter['score'] / (dc + 1)

    all_df_filter['包含海外基金'] = all_df_filter['组合文件名'].apply(
        lambda x: any(c in overseas_set for c in str(x).split('_')))
    result_df = all_df_filter[all_df_filter['包含海外基金']].reset_index(drop=True)

    log(f"[择优/完成] 资产排序终稿 | 候选总池: [{len(all_df_filter)}] 组 | 包含海外战略配资池: [{len(result_df)}] 组")
    return all_df_filter


# ====================================================================
# [总控枢纽] 算法维度爬坡与主控流程
# ====================================================================
def run_backest_process():
    RESULT_CSV = 'fund_data/all_funds_result.csv'
    CORR_CSV = 'fund_data/fund_correlations_300.csv'
    DOWNSIDE_CSV = 'fund_data/fof_evaluation_results_2d_pool3917.csv'

    _warmup_numba()

    log("[主控/池化] 执行基础 Base Pool 构建...")
    if not os.path.exists(RESULT_CSV):
        log(f"[主控/阻断] 缺乏底层基金净值体检数据 | 缺失文件: [{RESULT_CSV}]", level="ERROR")
        return

    df_results = pd.read_csv(RESULT_CSV)
    df_filtered = filter_fund_pool(df_results, min_annual_return=CFG_BASE_POOL_MIN_CAGR, min_day=FIXED_MIN_DAY)
    if df_filtered.empty:
        log("[主控/阻断] 本期无基金满足初筛要求，流程退出", level="WARN")
        return

    matched_downside = glob.glob(DOWNSIDE_CSV)
    ACTUAL_DOWNSIDE_CSV = matched_downside[
        0] if matched_downside else 'fund_data/fof_evaluation_results_2d_downside_300.csv'
    global_corr_matrix, global_downside_corr_matrix = precompute_correlations(
        result_csv=RESULT_CSV, corr_csv=CORR_CSV, downside_csv=ACTUAL_DOWNSIDE_CSV,
        max_workers=CFG_GLOBAL_MAX_WORKERS, df_filtered=df_filtered)

    if global_corr_matrix.empty:
        base_pool_files = df_filtered['adj_nav_file'].dropna().tolist()
    else:
        base_pool_files = _greedy_correlation_filter(
            df_filtered, global_corr_matrix, global_downside_corr_matrix, CFG_CORR_THRESHOLD,
            CFG_DOWNSIDE_CORR_THRESHOLD)

    master_df, base_pool_codes = _build_master_matrix(base_pool_files, CFG_GLOBAL_MAX_WORKERS)
    base_pool_codes_set = set(base_pool_codes)
    save_json("fund_data/base_pool_codes.json", base_pool_codes)
    final_fund_count = len(base_pool_codes)

    log(f"[主控/确池] 核心标的池已锁定 | 入选规模: [{final_fund_count}]只")
    if final_fund_count < 2:
        log("[主控/阻断] 基金数太少无法发生裂变组合，流程退出", level="WARN")
        return

    N = 2
    while N <= CFG_MAX_DIMENSION:
        log(f"\n[主控/多维演化] =============== 启动维度爬坡: 【{N}D】 ===============")
        output_parquet = f'fund_data/fof_evaluation_results_{N}d_pool{final_fund_count}_min_day_{FIXED_MIN_DAY}.parquet'

        if os.path.exists(output_parquet):
            log(f"[爬坡/越过] 该维度已计算完成 | 现有文件: [{os.path.basename(output_parquet)}]")
            N += 1
            continue

        good_prev_combos = set()
        if N > 2:
            good_prev_combos = get_previous_good_combos(N - 1, base_pool_codes_set, target_dim=N,
                                                        target_max_combos=CFG_DIM_UPGRADE_TARGET_COMBOS)
            if not good_prev_combos:
                log(f"[爬坡/阻断] 无优良降维种子支撑，算法自然收敛终止", level="WARN")
                break

        computed_set, cache_file = load_or_init_computed_set(N)

        if N == 2:
            uncomputed_combos, global_skipped, total_combos = [], 0, 0
            for c in itertools.combinations(base_pool_codes, 2):
                total_combos += 1
                if c not in computed_set:
                    uncomputed_combos.append(c)
                else:
                    global_skipped += 1
        else:
            uncomputed_combos, total_combos, global_skipped = generate_next_dimension_combos_apriori(
                N=N, good_prev_combos=good_prev_combos, computed_set=computed_set, strict_mode=True)

        if total_combos == 0:
            log(f"[爬坡/阻断] 该维度理论生成集为空，流程终止", level="WARN")
            break

        skip_ratio = global_skipped / total_combos * 100
        log(f"[爬坡/下发任务] 池深: [{total_combos:,}]组 | 缓存直达: [{global_skipped:,}]组 ({skip_ratio:.2f}%) | 等待核算: [{len(uncomputed_combos):,}]组")

        if not uncomputed_combos:
            N += 1
            continue

        ensure_dir(output_parquet)
        writer, global_computed = None, 0

        with ProcessPoolExecutor(max_workers=CFG_GLOBAL_MAX_WORKERS, initializer=_init_worker,
                                 initargs=(master_df,)) as executor:
            with tqdm(total=total_combos, desc=f"【{N}D】演算集群", unit="组") as pbar:
                if global_skipped > 0: pbar.update(global_skipped)
                for i in range(0, len(uncomputed_combos), BATCH_SIZE):
                    batch = uncomputed_combos[i:i + BATCH_SIZE]
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
                        if writer is None: writer = pq.ParquetWriter(output_parquet, table.schema, compression='zstd')
                        writer.write_table(table)

                    del futures, batch_results, batch, chunks

        if writer is not None: writer.close()
        _print_final_report(final_fund_count, N, total_combos, global_skipped, global_computed, output_parquet,
                            writer is not None)
        _update_cache_from_new_file(output_parquet, cache_file, computed_set)

        N += 1


if __name__ == '__main__':
    log("\n=== 【STAGE 1: 本地资产库清洗重建】 ===")

    # # 放开注释就会删除之前所有的数据，重新拉取最新的数据进行分析，谨慎操作
    # # if os.path.exists('fund_data'):
    # #     shutil.rmtree('fund_data')
    active_codes = get_active_fund_codes()
    fetch_and_save_fund_data(active_codes, year=CFG_HOLDINGS_YEAR, test_mode=False, max_workers=2)
    judge_fund_df(head_count=CFG_RECENT_DAYS_LIMIT, max_workers=CFG_GLOBAL_MAX_WORKERS, force_update=True)

    log("\n=== 【STAGE 2: 高维组合推演及记忆挂载】 ===")
    run_backest_process()

    log("\n=== 【STAGE 3: 择优决断与输出排期】 ===")
    valid_fund_codes = read_json("fund_data/base_pool_codes.json")
    fof_candidates = analyze_fof_combinations(VALID_FUND_CODES=valid_fund_codes)
    print()