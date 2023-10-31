# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/19 18:14
:last_date:
    2023/10/19 18:14
:description:

"""
from StrategyExecutor.MyTT import KDJ


def gen_daily_buy_signal_one(data):
    """
    股价创新低
    :param data:
    :return:
    """
    # 效果：
    # 概率增加，持有天数也增加

    # 还需继续优化
    data['Buy_Signal'] = (data['收盘'] == data['LL'])

def gen_daily_buy_signal_two(data):
    """
    昨日股价创新历史新低，今天收阳或者上涨，跳过前面的100个数据
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
                                   ((data['涨跌幅'] >= 0) | (data['收盘'] >= data['开盘']))


def gen_daily_buy_signal_three(data):
    """
    操盘做T + 操盘线小于3
    :param data:
    :return:
    """
    max_chaopan= 3
    data['Buy_Signal'] = (data['操盘线'] < max_chaopan) & \
                            (data['必涨']) & \
                         (data['涨跌幅'].shift(1) > -data['Max_rate'] * 0.8) & \
                         (data['涨跌幅'] > -data['Max_rate'] * 0.8)

def gen_daily_buy_signal_four(data):
    """
    今日股价创新60天新低，今日收阳

    :param data:
    :return:
    """
    # 效果：
    # 任然无法避免买入后继续跌，概率增加，持有天数减少

    # 还需继续优化
    die_rate = 2
    data['Buy_Signal'] = (data['收盘'] == data['收盘'].rolling(window=60).min()) & \
                         (data['收盘'] > data['开盘']) & \
                         (data['涨跌幅'] > -5)


def calculate_kdj(data, n=9, m1=3, m2=3):
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

    return data

def gen_daily_buy_signal_five(data):
    """
    前辈指标

    :param data:
    :return:
    """
    # 效果：
    # 任然无法避免买入后继续跌，概率增加，持有天数减少

    # 还需继续优化
    K,D,J = KDJ(data['收盘'],data['最高'],data['最低'])
    data['J'] = J
    data['买'] = data['J'].apply(lambda x: 10 if x < 0 else 0)
    data['Buy_Signal'] = (9.9 < data['买'].shift(1)) & (9.9 > data['买']) & (data['日期'] > '2013-01-01') & (data['涨跌幅'] < 0) # 效果最好
    # data['Buy_Signal'] = (9.9 < data['买'].shift(2)) & (9.9 > data['买'].shift(1)) & (data['涨跌幅'] < 0) & (data['日期'] > '2013-01-01')
    # 再往前判断前两天是否有穿越，并且连续两天是否都下跌
    # data['Buy_Signal'] = (9.9 < data['买'].shift(3)) & (9.9 > data['买'].shift(2)) & (data['涨跌幅'] < 0) & (data['日期'] > '2013-01-01')
    return data

def gen_true(data):
    """
    都是买入信号
    :param data:
    :return:
    """
    data['Buy_Signal'] = True
