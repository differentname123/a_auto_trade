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

from StrategyExecutor.MyTT import KDJ


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
    max_chaopan= 3
    data['Buy_Signal'] = (data['操盘线'] < max_chaopan) & \
                            (data['必涨']) & \
                         (data['涨跌幅'].shift(1) > -data['Max_rate'] * 0.8) & \
                         (data['涨跌幅'] > -data['Max_rate'] * 0.8)  & ((data['换手率'] > 0.5))

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
    K,D,J = KDJ(data['收盘'],data['最高'],data['最低'])
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

    data.loc[200:, 'Buy_Signal'] =(data['M'] > data['M'].shift(1)) & \
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
    data['V4'] =data['V2']

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
    data['X_7'] = (data['收盘'] >data['开盘']) & \
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

def mix(data):
    """
    买入信号
    :param data:
    :return:
    """
    # 复制一份数据
    data_1 = data.copy()
    data_2 = data.copy()
    gen_daily_buy_signal_six(data_1)
    gen_daily_buy_signal_five(data_2)
    data['Buy_Signal'] = data_1['Buy_Signal'] & data_2['Buy_Signal']

def gen_true(data):
    """
    都是买入信号
    :param data:
    :return:
    """
    data['Buy_Signal'] = True
