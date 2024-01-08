# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/19 18:14
:last_date:
    2023/10/19 18:14
:description:

"""
from StrategyExecutor.basic_daily_strategy import *


def gen_basic_daily_zhishu_buy_signal_1(data):
    """
    收盘值相关买入信号，一个是固定值，另一个是均线，还有新低或者新高
    :param data:
    :return:
    """

    data = gen_multiple_daily_buy_signal_fix(data, '上证指数开盘', [2900, 3000, 3100])
    data = gen_multiple_daily_buy_signal_ma(data, '上证指数开盘', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_max_min(data, '上证指数开盘', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_fix(data, '深证指数开盘', [9200, 9650, 10200])
    data = gen_multiple_daily_buy_signal_ma(data, '深证指数开盘', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_max_min(data, '深证指数开盘', [5, 10, 20])
    return data

def gen_basic_daily_zhishu_buy_signal_2(data):
    """
    收盘值相关买入信号，一个是固定值，另一个是均线，还有新低或者新高
    :param data:
    :return:
    """

    data = gen_multiple_daily_buy_signal_fix(data, '上证指数最高', [2900, 3000, 3100])
    data = gen_multiple_daily_buy_signal_ma(data, '上证指数最高', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_max_min(data, '上证指数最高', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_fix(data, '深证指数最高', [9200, 9650, 10200])
    data = gen_multiple_daily_buy_signal_ma(data, '深证指数最高', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_max_min(data, '深证指数最高', [5, 10, 20])
    return data

def gen_basic_daily_zhishu_buy_signal_3(data):
    """
    收盘值相关买入信号，一个是固定值，另一个是均线，还有新低或者新高
    :param data:
    :return:
    """

    data = gen_multiple_daily_buy_signal_fix(data, '上证指数最低', [2900, 3000, 3100])
    data = gen_multiple_daily_buy_signal_ma(data, '上证指数最低', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_max_min(data, '上证指数最低', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_fix(data, '深证指数最低', [9200, 9650, 10200])
    data = gen_multiple_daily_buy_signal_ma(data, '深证指数最低', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_max_min(data, '深证指数最低', [5, 10, 20])
    return data

def gen_basic_daily_zhishu_buy_signal_4(data):
    """
    收盘值相关买入信号，一个是固定值，另一个是均线，还有新低或者新高
    :param data:
    :return:
    """

    data = gen_multiple_daily_buy_signal_fix(data, '上证指数收盘', [2900, 3000, 3100])
    data = gen_multiple_daily_buy_signal_ma(data, '上证指数收盘', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_max_min(data, '上证指数收盘', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_fix(data, '深证指数收盘', [9200, 9650, 10200])
    data = gen_multiple_daily_buy_signal_ma(data, '深证指数收盘', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_max_min(data, '深证指数收盘', [5, 10, 20])
    return data

def gen_basic_daily_zhishu_buy_signal_5(data):
    """
    收盘值相关买入信号，一个是固定值，另一个是均线，还有新低或者新高
    :param data:
    :return:
    """

    MACD, MACD_Signal, MACD_Hist = talib.MACD(data['上证指数收盘'], fastperiod=12, slowperiod=26,
                                                                      signalperiod=9)
    data['上证指数BAR'] = (MACD - MACD_Signal) * 2
    data = gen_multiple_daily_buy_signal_ma(data, '上证指数BAR', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_max_min(data, '上证指数BAR', [5, 10, 20])

    MACD, MACD_Signal, MACD_Hist = talib.MACD(data['深证指数收盘'], fastperiod=12, slowperiod=26,
                                                                      signalperiod=9)
    data['深证指数BAR'] = (MACD - MACD_Signal) * 2
    data = gen_multiple_daily_buy_signal_ma(data, '深证指数BAR', [5, 10, 20])
    data = gen_multiple_daily_buy_signal_max_min(data, '深证指数BAR', [5, 10, 20])
    return data

def gen_basic_daily_zhishu_buy_signal_6(data):
    """
    产生股价['收盘', '开盘', '最高', '最低']相较于昨日的比较
    :param data:
    :return:
    """
    data = gen_multiple_daily_buy_signal_compare(data, ['上证指数收盘', '上证指数开盘', '上证指数最高', '上证指数最低'])
    data = gen_multiple_daily_buy_signal_compare(data, ['深证指数收盘', '深证指数开盘', '深证指数最高', '深证指数最低'])
    return data