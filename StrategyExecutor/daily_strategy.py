# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/19 18:14
:last_date:
    2023/10/19 18:14
:description:

"""
import numpy as np
import pandas as pd
import talib

from StrategyExecutor.MyTT import *
from StrategyExecutor.common import load_data
from StrategyExecutor.zuhe_daily_strategy import gen_signal, gen_all_basic_signal, gen_full_all_basic_signal


def gen_daily_buy_signal_one(data):
    """
    股价创新低，跳过前面的100个数据,换手率大于0.5
    timestamp: 20231107191930
    trade_count: 45255
    total_profit: 659593.0
    size of result_df: 7904
    ratio: 0.1746547342835046
    average days_held: 8.155960667329577
    average profit: 14.575030383383051
    :param data:
    :return:
    """
    # 效果：
    # 概率增加，持有天数也增加

    # 还需继续优化
    data.loc[100:, 'Buy_Signal'] = (data['收盘'] == data['LL']) & ((data['换手率'] > 0.5))


def gen_daily_buy_signal_two(data):
    """
    昨日股价创新历史新低，今天收阳或者上涨，跳过前面的100个数据,换手率大于0.5
    timestamp: 20231107191851
    trade_count: 22786
    total_profit: 396846.0
    size of result_df: 3839
    ratio: 0.16848064601070833
    average days_held: 5.401079610287018
    average profit: 17.416220486263494
    :param data:
    :return:
    """
    # 效果：
    # 概率增加，持有天数减少

    # 还需继续优化

    # 初始化所有的元素为False
    data['Buy_Signal'] = False

    # 为第101个元素及之后的元素计算Buy_Signal
    data.loc[100:, 'Buy_Signal'] = (data['收盘'].shift(1) == data['LL']) & \
                                   ((data['涨跌幅'] >= 0) | (data['收盘'] >= data['开盘'])) & ((data['换手率'] > 0.5))


def gen_daily_buy_signal_three(data):
    """
    操盘做T + 操盘线小于3 + 涨跌幅小于最大值的80%,换手率大于0.5
    timestamp: 20231107193756
    trade_count: 11783
    total_profit: 132593.0
    size of result_df: 1773
    ratio: 0.15047101756768225
    average days_held: 4.3834337605024185
    average profit: 11.252906730034796
    :param data:
    :return:
    """
    max_chaopan = 3
    data['Buy_Signal'] = (data['操盘线'] < max_chaopan) & \
                         (data['必涨']) & \
                         (data['涨跌幅'].shift(1) > -data['Max_rate'] * 0.8) & \
                         (data['涨跌幅'] > -data['Max_rate'] * 0.8) & ((data['换手率'] > 0.5))


def gen_daily_buy_signal_four(data):
    """
    今日股价创新60天新低，今日收阳,换手率大于0.5
    timestamp: 20231107193944
    trade_count: 18692
    total_profit: 473428.0
    size of result_df: 2735
    ratio: 0.14631928097581853
    average days_held: 6.30119837363578
    average profit: 25.327840787502677
    :param data:
    :return:
    """
    # 效果：
    # 任然无法避免买入后继续跌，概率增加，持有天数减少

    # 还需继续优化
    die_rate = 2
    data['Buy_Signal'] = (data['收盘'] == data['收盘'].rolling(window=60).min()) & \
                         (data['收盘'] > data['开盘']) & \
                         (data['涨跌幅'] > -5) & ((data['换手率'] > 0.5))


def gen_daily_buy_signal_five(data):
    """
    前辈指标,换手率大于0.5
    timestamp: 20231107194223
    trade_count: 185002
    total_profit: 5138929.0
    size of result_df: 27317
    ratio: 0.14765786315823612
    average days_held: 8.663911741494687
    average profit: 27.77769429519681
    :param data:
    :return:
    """
    # 效果：
    # 任然无法避免买入后继续跌，概率增加，持有天数减少
    n = 9
    m1 = 3
    m2 = 3
    # 计算N日内的最高价、最低价
    data['low_n'] = data['收盘'].rolling(window=n).min()
    data['high_n'] = data['收盘'].rolling(window=n).max()

    # 计算未成熟随机值 (RSV)
    data['rsv'] = (data['收盘'] - data['low_n']) / (data['high_n'] - data['low_n']) * 100

    # 使用递归公式计算K值和D值
    data['K'] = data['rsv'].ewm(span=m1).mean()
    data['D'] = data['K'].ewm(span=m2).mean()

    # 计算J值
    data['J'] = 3 * data['K'] - 2 * data['D']
    K, D, J = KDJ(data['收盘'], data['最高'], data['最低'])
    data['J'] = J
    data['买'] = data['J'].apply(lambda x: 10 if x < 0 else 0)
    data['Buy_Signal'] = (9.9 < data['买'].shift(1)) & (9.9 > data['买']) & ((data['换手率'] > 0.5))
    return data


def gen_daily_buy_signal_six(data):
    """
    主力追踪,换手率大于0.5
    timestamp: 20231107194300
    trade_count: 172794
    total_profit: 5679969.0
    size of result_df: 22605
    ratio: 0.13082051460120142
    average days_held: 11.77368427144461
    average profit: 32.8713323379284
    :param data:
    :return:
    """
    # 效果：
    # 任然无法避免买入后继续跌，概率增加，持有天数减少

    # Calculate moving averages for the trends
    data['上趋势'] = data['最低'].rolling(window=20).mean() * 1.2
    data['次上趋势'] = data['最低'].rolling(window=20).mean() * 1.1
    data['次下趋势'] = data['最高'].rolling(window=20).mean() * 0.9
    data['下趋势'] = data['最高'].rolling(window=20).mean() * 0.8

    # Calculate daily returns as a percentage
    data['ZD'] = (data['收盘'] - data['收盘'].shift(1)) / data['收盘'].shift(1) * 100

    # Calculate HDZF
    data['HDZF'] = (data['最高'].rolling(window=20).max() - data['收盘']) / (
            data['最高'].rolling(window=20).max() - data['最低'].rolling(window=20).min())

    # Define trend strength
    conditions = [
        (data['收盘'] > data['次上趋势']) & (data['收盘'] <= data['上趋势']),
        (data['收盘'] > data['次下趋势']) & (data['收盘'] <= data['次上趋势']),
        (data['收盘'] < data['下趋势']),
        (data['收盘'] > data['上趋势'])
    ]
    choices = [3, 2, 0, 4]
    data['趋势强度'] = np.select(conditions, choices, default=1)

    # Define the selection criteria
    data['Buy_Signal'] = (data['趋势强度'].shift(1) == 3) & (data['趋势强度'] == 2) & ((data['换手率'] > 0.5))


def gen_daily_buy_signal_seven(data):
    """
    尾盘买入选股
    timestamp: 20231107200541
    trade_count: 63620
    total_profit: 1890255.0
    size of result_df: 7185
    ratio: 0.11293618359006602
    average days_held: 9.05276642565231
    average profit: 29.71164728072933
    :param data:
    :return:
    """
    # Calculate the moving averages and other conditions
    data['M'] = data['收盘'].ewm(span=50, adjust=False).mean()
    data['N'] = (data['收盘'].rolling(window=5).mean() > data['收盘'].rolling(window=10).mean()) & \
                (data['收盘'].rolling(window=10).mean() > data['收盘'].rolling(window=20).mean())

    # Calculate the angle in degrees
    data['角度'] = np.arctan((data['M'] / data['M'].shift(1) - 1) * 100) * (180 / np.pi)
    data['均角'] = data['角度'].rolling(window=7).mean()

    data.loc[200:, 'Buy_Signal'] = (data['M'] > data['M'].shift(1)) & \
                                   (data['收盘'] < data['收盘'].shift(1) * 0.97) & \
                                   (data['最低'] <= data['M'] * 1.03) & \
                                   (data['收盘'] >= data['M'] * 0.98) & \
                                   (data['收盘'] == data['收盘'].rolling(window=3).min()) & \
                                   (data['收盘'] > data['收盘'].shift(1) * 0.90) & \
                                   (data['最高'] < (data['最高'].shift(1) + 0.07))


def gen_daily_buy_signal_eight(data):
    """
    阴量换手选股
    timestamp: 20231107202755
    trade_count: 71487
    total_profit: 2020658.0
    size of result_df: 12372
    ratio: 0.173066431658903
    average days_held: 16.061927343433073
    average profit: 28.26609033810343
    :param data:
    :return:
    """

    # Conditions for selection
    data['XG1'] = data['换手率'] > data['换手率'].shift(1) * 1.3
    data['XG2'] = data['开盘'] >= data['收盘']

    # V1, V2, V3 as defined
    data['V1'] = (data['收盘'] - data['收盘'].shift(1)) / data['收盘'].shift(1)
    data['V2'] = data['V1'] < -0.06
    data['V3'] = data['V1'] > 0.008

    # V4 is a bit ambiguous because of the curly braces. Assuming you want V3 and not V2,
    # it looks like a typo in the formula and should probably be just 'V3'.
    data['V4'] = data['V2']

    # Selecting stocks based on conditions V4, XG2, and XG1
    data['Buy_Signal'] = data['V4'] & data['XG2'] & data['XG1']


def gen_daily_buy_signal_nine(data):
    """
    超跌安全选股
    timestamp: 20231107204720
    trade_count: 130383
    total_profit: 4266658.0
    size of result_df: 20098
    ratio: 0.1541458625741086
    average days_held: 4.647062883964934
    average profit: 32.72403610900194
    :param data:
    :return:
    """
    data['X_1'] = data['收盘'] / data['收盘'].rolling(window=60).mean() * 100 < 75
    data['X_2'] = data['收盘'] / data['收盘'].rolling(window=40).mean() * 100 < 80
    data['X_3'] = data['最高'] > data['最低'] * 1.052
    data['X_4'] = data['X_3'] & (data['X_3'].rolling(window=5).sum() > 1)
    data['X_5'] = (data['收盘'] == data['最低']) & (data['最高'] == data['最低'])
    data['X_6'] = data['X_4'] & (data['X_2'] | data['X_1']) & ~data['X_5']

    # For X_7, 'EXIST' would mean that the condition was true at least once in the last 10 days.
    # We use rolling.apply with a custom lambda function to check this.
    data['X_7'] = (data['收盘'] > data['开盘']) & \
                  (data['X_6'].rolling(window=10).apply(lambda x: x.any(), raw=True)) & \
                  (data['收盘'].ewm(span=5).mean() > data['收盘'].ewm(span=5).mean().shift(1))

    # The final condition for selection
    data['Buy_Signal'] = data['X_6'] & ~data['X_7'].astype(bool)


def gen_daily_buy_signal_ten(data):
    """
    一触即发选股
    timestamp: 20231107210904
    trade_count: 7323
    total_profit: 273736.0
    size of result_df: 854
    ratio: 0.11661887204697528
    average days_held: 4.94141745186399
    average profit: 37.38030861668715
    :param data:
    :return:
    """
    # Calculate moving averages and other indicators
    X_1 = data['收盘'].rolling(window=27).mean()
    X_2 = (data['收盘'] - X_1) / X_1 * 100
    X_3 = X_2.rolling(window=2).mean()

    # Define the CROSS function
    # 定义CROSS函数
    def CROSS(series, constant):
        return (series.shift(1) < constant) & (series > constant)

    # 定义BARSLAST函数
    def BARSLAST(condition):
        # condition is a boolean Series, where True indicates the condition is met
        # cumsum() will reset at each True (condition met)
        # then we find the idxmax which will give us the index of the last True
        # finally, we subtract this index from the cumulative count (range index) to get the bars since last True
        return (condition[::-1].cumsum() == 1)[::-1].idxmax()

    X_3_cross_minus_10 = CROSS(X_3, -10)
    X_4 = data.apply(lambda row: BARSLAST(X_3_cross_minus_10.loc[:row.name]), axis=1)

    X_5 = (X_3 < -10) & (X_4 > 3)
    X_6 = X_5.replace(False, 0) * X_3.abs()
    X_7 = X_6 > 0
    X_8 = data['收盘'] / (
            data[['收盘', '最低', '最高']].mean(axis=1).ewm(span=3).mean().ewm(span=26).mean() * 0.9) < 0.95
    X_9 = ((data['收盘'] - data['收盘'].rolling(window=21).mean()) / data['收盘'].rolling(window=21).mean()).rolling(
        window=3).mean() * 100
    X_10 = X_9 < -15
    X_11 = (data['收盘'] - data['收盘'].rolling(window=28).mean()) / data['收盘'].rolling(
        window=28).mean() * 100 < -23
    X_12 = (data['收盘'] / data['收盘'].rolling(window=10).min().shift(1) < 0.96) & \
           (data['最低'] / data['最低'].rolling(window=10).min().shift(1) < 0.99) & \
           (data['收盘'] != data['最低']) & \
           (data['收盘'] / data['最低'] > 1.005)

    # Define the FILTER function
    def FILTER(condition, period):
        # 标记所有符合条件的周期
        marks = condition[condition].index
        # 过滤掉那些在指定周期内的重复标记
        filtered = marks[~marks.duplicated(keep='first')]
        # 保留间隔大于等于period的标记
        to_keep = [idx for i, idx in enumerate(filtered) if i == 0 or filtered[i] - filtered[i - 1] >= period]
        # 返回过滤后的条件序列
        filtered_condition = pd.Series(False, index=condition.index)
        filtered_condition[to_keep] = True
        return filtered_condition

    # Apply the filter to get the final selection
    data['Buy_Signal'] = FILTER(X_7 & X_8 & X_10 & X_11 & X_12, 10)


def gen_daily_buy_signal_eleven(data):
    """
    今买明卖选股
    timestamp: 20231107210904
    trade_count: 7323
    total_profit: 273736.0
    size of result_df: 854
    ratio: 0.11661887204697528
    average days_held: 4.94141745186399
    average profit: 37.38030861668715
    :param data:
    :return:
    """
    df = pd.DataFrame(data)
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)

    # 计算XYZ_1：分步骤计算以避免使用shift函数在lambda表达式中
    high_low = df['最高'] - df['最低']
    close_high = abs(df['收盘'].shift(1) - df['最高'])
    close_low = abs(df['收盘'].shift(1) - df['最低'])
    # 计算XYZ_1
    df['XYZ_1'] = df.apply(lambda row: max(row['最高'] - row['最低'],
                                           abs(df['收盘'].shift(1).loc[row.name] - row['最高']),
                                           abs(df['收盘'].shift(1).loc[row.name] - row['最低'])), axis=1)

    df['XYZ_2'] = (df['最高'] + df['最低']) / 2 + df['XYZ_1'].rolling(window=2).mean()
    df['XYZ_3'] = (df['最高'] + df['最低']) / 2 - df['XYZ_1'].rolling(window=2).mean()
    df['XYZ_4'] = df['XYZ_2'].shift(1).where(df['XYZ_2'] <= df['XYZ_2'].shift(1))
    df['XYZ_5'] = df['XYZ_2'].rolling(window=3).min()
    # 预先计算XYZ_4和XYZ_5的移动值
    df['XYZ_4_shifted'] = df['XYZ_4'].shift(1)
    df['XYZ_5_shifted'] = df['XYZ_5'].shift(1)

    # 计算XYZ_6
    df['XYZ_6'] = df.apply(
        lambda row: row['XYZ_4'] if row['XYZ_5'] != row['XYZ_5_shifted'] and row['XYZ_4'] < row['XYZ_4_shifted'] else (
            row['XYZ_4'] if row['XYZ_4'] == row['XYZ_5'] else row['XYZ_5']), axis=1)

    df['XYZ_7'] = (df['XYZ_2'] == df['XYZ_6']).astype(int).rolling(window=2).sum()
    # 首先计算交叉点
    cross_points = (df['收盘'].shift(1) < df['XYZ_6']) & (df['收盘'] > df['XYZ_6']) | (df['收盘'].shift(1) > df['XYZ_6']) & (
            df['收盘'] < df['XYZ_6'])
    df['XYZ_7'] = df['XYZ_7'].fillna(0)
    # 计算XYZ_8
    df['XYZ_8'] = 0
    for index, value in df.iterrows():
        window_size = int(df.loc[index, 'XYZ_7'])
        if window_size > 0:
            df.at[index, 'XYZ_8'] = cross_points.loc[:index].tail(window_size).sum()

    # 请在您的环境中尝试运行这段代码。
    # 初始化XYZ_9列
    df['XYZ_9'] = np.nan

    # 设置最后一次XYZ_8大于0的日期索引
    last_index = None
    for index, row in df.iterrows():
        if row['XYZ_8'] > 0:
            last_index = index
        df.at[index, 'XYZ_9'] = (index - last_index).days if last_index is not None else np.nan

    # 确保XYZ_9中的NaN值被处理，例如替换为0
    df['XYZ_9'] = df['XYZ_9'].fillna(0)

    # 计算XYZ_10
    # 首先检查XYZ_9的最大值，并据此确定合理的窗口大小
    max_XYZ_9 = df['XYZ_9'].max()
    window_size = int(max_XYZ_9) + 1 if max_XYZ_9 >= 0 else 1  # 确保窗口大小是正整数

    # 使用确定的窗口大小计算XYZ_10
    df['XYZ_10'] = (df['XYZ_3'].rolling(window=window_size).max() > df['收盘']).astype(int).cumsum()

    df['XYZ_11'] = df['收盘'].rolling(window=18).mean()
    df['XYZ_12'] = df['收盘'] >= df['XYZ_11'] * 1.004
    df['XYZ_13'] = df['XYZ_11'] >= df['XYZ_11'].shift(1)
    df['XYZ_14'] = df['XYZ_12'] & df['XYZ_13']
    df['XYZ_15'] = df.apply(
        lambda row: (row['收盘'] - row['最低']) / (row['最高'] - row['最低']) if row['收盘'] < row['开盘'] else 0, axis=1)

    # 应用选股条件
    data['Buy_Signal'] = (df['XYZ_9'] < df['XYZ_10']) & \
                         (df['收盘'] / df['最低'].rolling(window=3).min().shift(1) < 1) & \
                         (df['收盘'] < df['开盘']) & \
                         (df['收盘'] != df['最低']) & \
                         df['XYZ_14'] & \
                         (df['XYZ_15'] > 0.03) & (df['XYZ_15'] < 0.3)
    return data


def gen_daily_buy_signal_twelve(data):
    """
    下跌反弹选股策略
    今日创40日新低，今日成交量相较昨日3日平均成交额下降一倍
    trade_count: 27195
    total_profit: 411826.0
    size of result_df: 5622
    ratio: 0.20672917815774958
    average days_held: 12.319213090641663
    average profit: 15.14344548630263
    :param data:
    :return:
    """
    data['Buy_Signal'] = (data['收盘'] <= data['收盘'].rolling(window=40).min()) & \
                         (data['成交额'] < data['成交额'].shift(1).rolling(window=5).mean() / 2)


def gen_daily_buy_signal_thirteen(data):
    """
    超跌不停选股策略
    跌幅大于8小于9.8
    timestamp: 20231107210904
    trade_count: 7323
    total_profit: 273736.0
    size of result_df: 854
    ratio: 0.11661887204697528
    average days_held: 4.94141745186399
    average profit: 37.38030861668715
    效果:
        待优化
    :param data:
    :return:
    """
    data['Buy_Signal'] = (-9.8 <= data['涨跌幅']) & (data['涨跌幅'] <= -8) & (
            data['收盘'] < data['收盘'].rolling(window=5).mean())


def gen_daily_buy_signal_fourteen(data):
    """
    下影线大于实体选股策略
    timestamp: 20231112234314
    trade_count: 15308
    total_profit: 378041.0
    size of result_df: 2685
    ratio: 0.1753984844525738
    average days_held: 11.088189182126992
    average profit: 24.695649333681736
    效果:
        待优化
    :param data:
    :return:
    """
    data['Buy_Signal'] = ((data['开盘'] - data['收盘']) * 4 < (data['收盘'] - data['最低'])) \
                         & ((data['开盘'] - data['收盘']) > 0) \
                         & (data['换手率'] > 0.5)


def gen_daily_buy_signal_fiveteen(data):
    """
    开盘即最低选股策略
    timestamp: 20231113010041
    trade_count: 50020
    total_profit: 1295070.0
    size of result_df: 5777
    ratio: 0.1154938024790084
    average days_held: 8.255117952818873
    average profit: 25.891043582566972
    :param data:
    :return:
    """
    data['Buy_Signal'] = (data['开盘'] == data['最低']) & (data['收盘'] > data['开盘']) \
                         & (data['换手率'] > 0.5) & (data['BAR'] == data['BAR'].rolling(window=10).min()) & (
                                 data['涨跌幅'] < 5) & (data['BAR'] < 0)


def gen_daily_buy_signal_sixteen(data):
    """
    收盘最低选股策略
    timestamp: 20231113003457
    trade_count: 71207
    total_profit: 1399947.0
    size of result_df: 13422
    ratio: 0.1884927043689525
    average days_held: 13.088502534863146
    average profit: 19.66024407712725
    :param data:
    :return:
    """
    data['Buy_Signal'] = (data['收盘'] == data['最低']) & (data['换手率'] > 0.5) \
                         & (data['收盘'] == data['收盘'].rolling(window=20).min())


def gen_daily_buy_signal_seventeen(data):
    """
    macd最低选股策略
    示例情形：
        韵达股份 002120 20230926
    macd创10日新低，且macd值小于0，也是绝对值的最大值
    并且macd差值差距变大
    并且是阴线
    timestamp: 20231114013709
    trade_count: 78581
    total_profit: 2656420.0
    size of result_df: 7746
    ratio: 0.09857344650742546
    average days_held: 7.831409628281646
    average profit: 33.8048637711406
    :param data:
    :return:
    """
    data['Buy_Signal'] = (data['BAR'] == data['BAR'].rolling(window=10).min()) & (
            data['abs_BAR'] == data['abs_BAR'].rolling(window=10).max()) \
                         & ((data['macd_cha'].shift(1) - 0.01) > data['macd_cha']) \
                         & (data['macd_cha'].shift(1) < 0) & (data['涨跌幅'] > -9) & (
                                 (data['开盘'].shift(1) - data['收盘'].shift(1)) > 0)
    return data


def gen_daily_buy_signal_eighteen(data):
    """
    macd陡降选股策略
    示例情形：
        韵达股份 002120 20230320
    :param data:
    :return:
    """
    data['Buy_Signal'] = (data['macd_cha'] < 0) & (data['macd_cha'].shift(1) > data['macd_cha'])


def gen_daily_buy_signal_nineteen(data):
    """
    大阴线选股策略
    示例情形：
        东方电子 000682 20230828
    :param data:
    :return:
    """
    data['Buy_Signal'] = ((data['开盘'] - data['收盘']) > 2 * (data['开盘'].shift(1) - data['收盘'].shift(1))) & (
            data['开盘'] > data['收盘']) & ((data['开盘'] - data['收盘']) > 0.04 * data['收盘'])


def gen_daily_buy_signal_twenty(data):
    """
    最高小于昨日收盘选股策略
    示例情形：

    timestamp: 20231117000838
    trade_count: 546559
    total_profit: 14281917.0
    size of result_df: 72256
    ratio: 0.13220164703170198
    average days_held: 8.932537932775784
    average profit: 26.130604381228743
    :param data:
    :return:
    """
    data['Buy_Signal'] = (data['最高'] < data['收盘'].shift(1)) & (data['涨跌幅'] > -data['Max_rate'] * 0.8)


def gen_daily_buy_signal_21(data):
    """
    涨跌幅小于0.5收盘选股策略
    示例情形：

    timestamp: 20231117001921
    trade_count: 1376805
    total_profit: 26313725.0
    size of result_df: 204926
    ratio: 0.1488417023471007
    average days_held: 7.7690304727248956
    average profit: 19.112165484582057
    :param data:
    :return:
    """
    data['Buy_Signal'] = (data['涨跌幅'] >= -0.5) & (data['涨跌幅'] <= 0.5)


def gen_daily_buy_signal_22(data):
    """
    振幅大于当然最大比例收盘选股策略
    示例情形：
    timestamp: 20231117002451
    trade_count: 506685
    total_profit: 14190804.0
    size of result_df: 92374
    ratio: 0.18231050850133712
    average days_held: 15.275893306492199
    average profit: 28.007152372775984
    :param data:
    :return:
    """
    data['Buy_Signal'] = ((data['最高'] - data['最低']) > data['Max_rate'] * 0.01 * data['收盘'])


def gen_daily_buy_signal_23(data):
    """
    高低持平选股策略
    今日收盘和昨日开盘持平，今日开盘和昨日收盘持平
    示例情形：
    timestamp: 20231117003236
    trade_count: 290485
    total_profit: 2512478.0
    size of result_df: 54245
    ratio: 0.18673941855861748
    average days_held: 7.306370380570425
    average profit: 8.649252112845758
    :param data:
    :return:
    """
    data['Buy_Signal'] = (abs((data['开盘'] - data['收盘'].shift(1))) < (0.01 + 0.001 * data['收盘'].shift(1))) & (
            abs((data['开盘'].shift(1) - data['收盘'])) < (0.01 + 0.001 * data['收盘'].shift(1)))


def gen_daily_buy_signal_24(data):
    """
    游击战术策略
    最低超过下轨
    但是实体高价大于昨日实体高价
    或者实体低价大于昨日实体低价
    示例情形：
    timestamp: 20231117003236
    trade_count: 290485
    total_profit: 2512478.0
    size of result_df: 54245
    ratio: 0.18673941855861748
    average days_held: 7.306370380570425
    average profit: 8.649252112845758
    :param data:
    :return:
    """
    low_zhi = 96

    def SMA(S, N, M=1):  # Chinese style SMA
        return pd.Series(S).ewm(alpha=M / N, adjust=False).mean().values

    df = data
    df['DIR'] = abs(df['收盘'] - df['收盘'].shift(10))
    df['VIR'] = df['收盘'].diff().abs().rolling(window=10).sum()
    df['ER'] = df['DIR'] / df['VIR']
    df['CS'] = SMA(df['ER'] * (2 / 3 - 2 / 14) + 2 / 14, 3, 1)
    df['CQ'] = df['CS'] ** 3
    # Correct window size calculation for '裁决'

    window_sizes = (10 - df['CS'] * 10).apply(np.floor).fillna(1).astype(int).clip(lower=1)
    df['ma'] = [MA(df['收盘'], w)[i] if i >= w else np.nan for i, w in enumerate(window_sizes)]

    df['裁决'] = EMA(df['ma'], 2)

    df['CD'] = np.nan_to_num(df['收盘'] / df['裁决']) * 100
    df['OD'] = np.nan_to_num(df['开盘'] / df['裁决']) * 100
    df['OH'] = np.nan_to_num(df['最高'] / df['裁决']) * 100
    df['OL'] = np.nan_to_num(df['最低'] / df['裁决']) * 100
    data['Buy_Signal'] = (df['OD'] < low_zhi) & ((df['OD'] > df['OD'].shift(1))) \
                         & (data['收盘'] < data['收盘'].rolling(window=40).mean()) \
                         & (data['涨跌幅'] < 0) \
                         & (data['换手率'] > 0.5) & (data['涨跌幅'].shift(1) < 0) & (df['CD'] < low_zhi) & (
                                 data['涨跌幅'] > -0.95 * data['Max_rate']) & \
                         (data['开盘'].shift(1) > data['收盘'].shift(1))
    return data


def EXPMA(S, N):
    return pd.Series(S).ewm(alpha=2 / (N + 1), adjust=False).mean().values

def can_not_buy(data):
    """
    不能买的股票
    :param data:
    :return:
    """
    data['Buy_Signal'] = data['涨跌幅'] > 0.95 * data['Max_rate']

def gen_daily_buy_signal_25(data):
    """
    金牛选股策略
    timestamp: 20231129010053
    trade_count: 47753
    total_profit: 272905.48374484706
    size of result_df: 10796
    ratio: 0.22608003685632316
    average days_held: 2.341863338428999
    average profit: 5.714939035135951
    :param data:
    :return:
    """

    def MA(S, N):
        return pd.Series(S).rolling(N).mean()

    def EXPMA(S, N):
        return pd.Series(S).ewm(alpha=2 / (N + 1), adjust=False).mean().values

    # 将列名分配给变量
    close = data['收盘']
    open_price = data['开盘']
    volume = data['成交量']

    # 公式中的其他计算
    M11 = 5
    M44 = 60
    P = volume
    LB = volume / MA(volume, 5).shift(1)
    LJC = MA(P, M11) > MA(P, M44) * 0.9

    M1, M2, M3, M4, M5, M6 = 4, 6, 9, 13, 18, 24
    MA5 = MA(close, 60)
    PB1 = (EXPMA(close, M1) + MA(close, M1 * 2) + MA(close, M1 * 4)) / 3
    PB2 = (EXPMA(close, M2) + MA(close, M2 * 2) + MA(close, M2 * 4)) / 3
    PB3 = (EXPMA(close, M3) + MA(close, M3 * 2) + MA(close, M3 * 4)) / 3
    PB4 = (EXPMA(close, M4) + MA(close, M4 * 2) + MA(close, M4 * 4)) / 3
    PB5 = (EXPMA(close, M5) + MA(close, M5 * 2) + MA(close, M5 * 4)) / 3
    PB6 = (EXPMA(close, M6) + MA(close, M6 * 2) + MA(close, M6 * 4)) / 3

    AAA = MIN(MIN(MIN(MIN(MIN(PB1, PB2), PB3), PB4), PB5), PB6)
    BBB = MAX(MAX(MAX(MAX(MAX(PB1, PB2), PB3), PB4), PB5), PB6)
    XG = (IF(open_price < AAA, True, False) | IF(open_price.shift(1) < AAA, True, False)) & (close > BBB)
    P6 = ABS(PB1 - PB2) + ABS(PB1 - PB3) + ABS(PB1 - PB4) + ABS(PB1 - PB5) + ABS(PB1 - PB6) + \
         ABS(PB2 - PB3) + ABS(PB2 - PB4) + ABS(PB2 - PB5) + ABS(PB2 - PB6) + ABS(PB3 - PB4) + \
         ABS(PB3 - PB5) + ABS(PB3 - PB6) + ABS(PB4 - PB5) + ABS(PB4 - PB6) + ABS(PB5 - PB6)
    LXZH = P6 / close < 0.20
    JL = (close > MA5) & LJC & ((LB > 2.2) | (LB.shift(1) > 2.2)) & (XG | XG.shift(1)) & \
         (close > MA5 * 1.02) & (close < MA5 * 1.15) & (close > open_price * 1.02) & \
         (LXZH | LXZH.shift(1)) & ((close - close.shift(1)) / close.shift(1) > 0.03)
    PD = JL | JL.shift(1)
    XGG = PD & (close > PB1 * 0.88)

    # 将结果添加到数据框
    data['Buy_Signal'] = (data['涨跌幅'] < 0.95 * data['Max_rate']) & (XGG | (XG & (LB > 2.2) & (close > MA5 * 1.02) & (close < MA5 * 1.15)))
    return data

def gen_daily_buy_signal_26(data):
    """
    超级短线选股策略
    timestamp: 20231203205742
    trade_count: 24977
    total_profit: 189286.75811711114
    total_cost: 33135064.24188291
    size of result_df: 1799
    ratio: 0.07202626416302998
    average days_held: 1.2541538215157946
    average profit: 7.5784424917768805
    average 1w profit: 57.12581594390018
    :param data:
    :return:
    """
    X_2 = data['最高'].rolling(window=20).max().shift(1)
    X_3 = data['最低'].rolling(window=10).min().shift(1)
    X_4 = data['最低'] > X_3
    X_5 = data['最高'] < X_2
    X_6 = data.index.to_series().apply(lambda x: x > 30)
    X_7 = np.maximum(np.maximum(data['最高'] - data['最低'], abs(data['收盘'].shift(1) - data['最高'])), abs(data['收盘'].shift(1) - data['最低']))
    X_8 = X_7.rolling(window=14).mean()
    X_9 = ((data['最高'] + data['最低']) / 2).rolling(window=1).mean()
    X_10 = X_7.rolling(window=2).mean()
    X_11 = X_9 + X_10
    X_12 = X_9 - X_10
    X_13 = data['收盘'] < X_12.rolling(window=3).max()
    X_14 = (data['收盘'] < data['收盘'].shift(1)) & (data['收盘'] < data['开盘']) & (data['收盘'] != data['最低']) & (data['收盘'] > data['收盘'].rolling(window=18).mean())
    X_15 = data['收盘'].rolling(window=20).mean() > data['收盘'].rolling(window=20).mean().shift(1)
    X_16 = data['收盘'] < data['最低'].rolling(window=3).min().shift(1)
    X_17 = (X_8 > X_8.shift(1)) & (X_7 > X_7.shift(1))
    X_18 = (data['收盘'] - data['最低']) / (data['最高'] - data['最低'])
    X_19 = X_18 < 0.3
    X_20 = (data['最高'] - data['开盘']) / (data['最高'] - data['最低'])
    X_21 = X_20 < 0.4

    data['Buy_Signal'] &= X_14 & X_15 & X_16 & X_17 & X_19 & X_13 & X_4 & X_5 & X_6

    return data

def gen_daily_buy_signal_27(data):
    """
    小楷尾盘淘金选股策略
    trade_count: 4145
    total_profit: 27710.67343010754
    size of result_df: 516
    ratio: 0.12448733413751507
    average days_held: 1.4651387213510254
    average profit: 6.685325314863098
    :param data:
    :return:
    """
    VAR1 = (data['开盘'].shift(3) > data['收盘'].shift(3)) & (data['开盘'].shift(2) > data['收盘'].shift(2)) & (data['开盘'].shift(1) < data['收盘'].shift(1))
    VAR2 = data['开盘'] > data['收盘']
    VAR3 = data['收盘'].shift(1) / data['收盘'] >= 1.03
    VAR4 = data['收盘'] / data['最低'] < 1.03
    VAR5 = (data['成交量'] < data['成交量'].shift(3) * 1.2) & (data['成交量'] > data['成交量'].shift(3) * 0.9)
    VAR6 = data['最高'] < data['最高'].shift(1)
    VAR7 = data['收盘'].ewm(span=5, adjust=False).mean() > data['收盘'].rolling(window=14).mean() * 1.017
    VAR8 = (data['开盘'].shift(3) < data['收盘'].shift(3)) & (data['开盘'].shift(2) < data['收盘'].shift(2)) & (data['开盘'].shift(1) > data['收盘'].shift(1))
    VAR9 = data['收盘'].shift(2) < data['收盘'].shift(1) * 1.01
    VAR10 = data['成交量'] < data['成交量'].rolling(window=16).min() * 9.5
    VAR11 = (data['收盘'] - data['最低']) < (data['开盘'] - data['收盘']) * 0.6
    VAR12 = data['收盘'].shift(1) / data['收盘'].shift(2) < 1.0995
    VAR13 = data['最高'].shift(1) / data['收盘'] < 1.07
    VAR14 = data['最高'] - data['开盘'] > (data['收盘'] - data['最低'])
    VAR15 = data['最高'].shift(3) / data[['开盘', '收盘']].min(axis=1).shift(3) > 1.011
    X_15 = np.where(data['收盘'] < data['开盘'], (data['收盘'] - data['最低']) / (data['最高'] - data['最低']), 0)
    涨幅 = data['收盘'] / data['收盘'].shift(1)

    XG1 = VAR8 & VAR2 & (涨幅 < 1.098) & VAR3 & VAR4 & VAR6 & VAR11 & VAR12 & VAR10 & VAR13 & VAR14 & VAR15 & VAR9 & (X_15 > 0.03) & (X_15 < 0.3)
    XG2 = VAR1 & VAR2 & VAR3 & VAR4 & VAR5 & VAR6 & VAR7

    data['Buy_Signal'] = (XG1 | XG2)

    return data


def gen_daily_buy_signal_28(data):
    """
    九五至尊选股策略
    timestamp: 20231129022402
    trade_count: 9813
    total_profit: 160647.18787170682
    size of result_df: 1818
    ratio: 0.18526444512381535
    average days_held: 2.119229593396515
    average profit: 16.37085375233943
    :param data:
    :return:
    """
    X_3 = 0.01 * data['成交额'].ewm(span=13, adjust=False).mean() / data['成交量'].ewm(span=13, adjust=False).mean()
    JWZZ = (data['收盘'] - X_3) / X_3 * 1000

    # CROSS(JWZZ, 95) - Indicates where JWZZ crosses above 95
    data['Buy_Signal'] &= (JWZZ.shift(1) < 95) & (JWZZ >= 95)

def gen_daily_buy_signal_29(data):
    """
    昨日涨停今日最低跌停选股策略
    :param data:
    :return:
    """

    # CROSS(JWZZ, 95) - Indicates where JWZZ crosses above 95
    data['Buy_Signal'] = (data['涨跌幅'].shift(1) > 9.5) & (data['最低'] < (100 - 9.5) * data['收盘'].shift(1) / 100) & (data['日期'] > '2023-01-01')
    return data

def gen_daily_buy_signal_30(data):
    """
    曾跌停，但收阳
    :param data:
    :return:
    """

    # CROSS(JWZZ, 95) - Indicates where JWZZ crosses above 95
    data['Buy_Signal'] = (data['Max_rate'] > 0) & (data['最低'] <= data['收盘'].shift(1) * 0.95 * 0.9) & (data['收盘'] > data['开盘'])& (data['收盘'] > 2)
    return data

def select_stocks(data):
    def SMA(S, N, M=1):  # Chinese style SMA
        return pd.Series(S).ewm(alpha=M / N, adjust=False).mean().values
    def CROSS(S, constant):
        return (pd.Series(S).shift(1) < constant) & (pd.Series(S) > constant)

    def CROSS_s(S1, S2):  # 判断向上金叉穿越 CROSS(MA(C,5),MA(C,10))  判断向下死叉穿越 CROSS(MA(C,10),MA(C,5))
        return (pd.Series(S1).shift(1) < pd.Series(S2).shift(1)) & (pd.Series(S1) > pd.Series(S2))

    def BARSLAST(S):  # 上一次条件成立到当前的周期, BARSLAST(C/REF(C,1)>=1.1) 上一次涨停到今天的天数
        M = np.concatenate(([0], np.where(S, 1, 0)))
        for i in range(1, len(M)):  M[i] = 0 if M[i] else M[i - 1] + 1
        return M[1:]

        # Assuming data columns are in the order: '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'
    close = data['收盘']
    low = data['最低']
    high = data['最高']

    # V1, V2, V3
    V1 = EMA(close, 5) - EMA(close, 340)
    V2 = EMA(V1, 144)
    V3 = (close - LLV(low, 27)) / (HHV(high, 27) - LLV(low, 27)) * 100

    # GUP0
    GUP0 = IF(CROSS(V3, 5) & (V1 < V2), 20, 0) + 20

    # VARMM1 to VARMM5
    VARMM1 = REF(low, 1)
    VARMM2 = SMA(ABS(low - VARMM1), 13) / SMA(MAX(low - VARMM1, 0), 13) * 4
    VARMM3 = EMA(VARMM2, 13)
    VARMM4 = LLV(low, 34)
    VARMM5 = EMA(IF(low <= VARMM4, VARMM3, 0), 3)

    # 主力 and 主力进场
    主力 = VARMM5 > REF(VARMM5, 1)
    主力进场 = SUM(主力, 20)

    # 快线 and 慢线
    快线 = (close - LLV(low, 9)) / (HHV(high, 9) - LLV(low, 9)) * 100
    慢线 = SMA(快线, 3)

    # BB
    BB = IF(BARSLAST(CROSS(慢线, 快线)) >= 3 & CROSS(快线, 慢线) & (慢线 < 30), 20, 0)
    # data['慢_chaunshang'] = BARSLAST(CROSS_s(慢线, 快线))
    data['慢_shang'] = pd.Series(CROSS_s(慢线, 快线))
    data['快_shang'] = CROSS_s(快线, 慢线)
    data['BB'] = BB
    data['慢线'] = 慢线
    data['快线'] = 快线
    data['主力'] = 主力
    data['GUP0'] = GUP0
    data['V3'] = V3
    data['主力进场'] = 主力进场


    # 买点
    买点 = IF(BB>0 & (GUP0 > 25) & (主力进场 > 5), 35, 0)

    # 底
    VAR1J = 3
    VAR2J = ((3 * SMA((((close - LLV(low, 27)) / (HHV(high, 27) - LLV(low, 27))) * 100), 5)) - (2 * SMA(SMA((((close - LLV(low, 27)) / (HHV(high, 27) - LLV(low, 27))) * 100), 5), 3)))
    data['VAR2J'] = VAR2J
    data['底'] = IF(CROSS(VAR2J, VAR1J), 1, 0)
    data['买点'] = 买点
    # sell
    sell = 买点 | data['底']
    data['Buy_Signal'] = sell
    return sell


def gen_daily_buy_signal_last(data):
    """
    在产生信号后，后一天下跌再买入
    :param data:
    :return:
    """
    data_1 = data.copy()
    gen_daily_buy_signal_24(data_1)
    data['Buy_Signal'] = data_1['Buy_Signal'].shift(1) & (data['涨跌幅'] < 0)


def mix(data):
    """
    买入信号
    :param data:
    :return:
    """
    # 复制一份数据

    data_1 = data.copy()
    data_2 = data.copy()
    data_1 = gen_full_all_basic_signal(data_1)
    gen_signal(data_1,'收盘_5日_大极值signal:换手率_大于_5_日均线_signal:最低_5日_大极值signal:振幅_大于_10_日均线_signal:BAR_小于_5_日均线_signal_yes:最高_20日_小极值_signal_yes:振幅_10日_大极值signal_yes'.split(':'))




    data['Buy_Signal'] = data_1['Buy_Signal']


def gen_true(data):
    """
    都是买入信号
    :param data:
    :return:
    """
    data['Buy_Signal'] = (data['涨跌幅'] < 0.95 * data['Max_rate'])
