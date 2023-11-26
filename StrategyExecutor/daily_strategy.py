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
    gen_signal(data_1, '最高_5日_大极值signal:最高_小于_5_日均线_signal'.split(':'))
    data['Buy_Signal'] = data_1['Buy_Signal']


def gen_true(data):
    """
    都是买入信号
    :param data:
    :return:
    """
    data['Buy_Signal'] = (data['涨跌幅'] < 0.95 * data['Max_rate'])
