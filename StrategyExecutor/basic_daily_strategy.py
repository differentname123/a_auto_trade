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


def gen_multiple_daily_buy_signal_1(data, key, value_list, ma_list, max_min_list):
    """
    产生公共结构的买入信号
    列如:
        涨跌幅大于或者小于固定的值1 3 5等，或者是大于或者小于5 10 20平均数的值
    :param ma_list:
    :param data:
    :return:
    """

    # 产生固定值的区间信号
    if value_list:
        sorted_values = sorted(value_list)

        # 为每个区间创建列
        for i in range(len(sorted_values)):
            if i == 0:
                # 处理小于第一个值的情况
                column_name = f'{key}_小于_{sorted_values[i]}_固定区间_signal'
                data[column_name] = data[key] < sorted_values[i]
            else:
                # 处理区间值
                lower = sorted_values[i - 1]
                upper = sorted_values[i]
                column_name = f'{key}_{lower}_到_{upper}_固定区间_signal'
                data[column_name] = data[key].apply(lambda x: lower <= x < upper)

        # 处理大于最后一个值的情况
        column_name = f'{key}_大于_{sorted_values[-1]}_固定区间_signal'
        data[column_name] = data[key] > sorted_values[-1]

    # 产生平均值的区间信号
    if ma_list:
        for value in ma_list:
            column_name = f'{key}_大于_{value}_日均线_signal'
            data[column_name] = data[key] > data[key].rolling(window=value).mean()
            column_name = f'{key}_小于_{value}_日均线_signal'
            data[column_name] = data[key] <= data[key].rolling(window=value).mean()

    # 产生极值的信号
    if max_min_list:
        for value in max_min_list:
            column_name = f'{key}_{value}日_大极值signal'
            data[column_name] = data[key] == data[key].rolling(window=value).max()
            column_name = f'{key}_{value}日_小极值_signal'
            data[column_name] = data[key] == data[key].rolling(window=value).min()

    return data

def gen_basic_daily_buy_signal_yesterday(data,key):
    """
    找出昨日包含指定key的字段为true的字段赋值为true
    :param data:
    :param key:
    :return:
    """
    # 找出data中包含key的字段
    columns = [column for column in data.columns if key in column]
    # 找出昨日包含key的字段为true的字段赋值为true
    for column in columns:
        data[column+'yesterday'] = (data[column].shift(1) == True)
    return data

def fun(data):
    data = gen_multiple_daily_buy_signal_1(data, '收盘', [1, 5, 10], [5, 10, 20], [5, 10, 20])
    data = gen_basic_daily_buy_signal_yesterday(data,'极值')
    return data


def gen_basic_daily_buy_signal_1(data):
    """
    收盘值相关买入信号，一个是固定值，另一个是均线，还有新低或者新高
    :param data:
    :return:
    """

    data = gen_multiple_daily_buy_signal_1(data, '收盘', [1, 5, 10], [5, 10, 20], [5, 10, 20])
    return data


def gen_basic_daily_buy_signal_2(data):
    """
    换手率相关买入信号，一个是固定值，另一个是均线，还有新低或者新高
    :param data:
    :return:
    """
    data = gen_multiple_daily_buy_signal_1(data, '换手率', [1, 5, 10], [5, 10, 20], [5, 10, 20])
    return data

def gen_basic_daily_buy_signal_3(data):
    """
    阴线或者阳线
    :param data:
    :return:
    """
    data['实体_阴线_signal'] = data['收盘'] < data['开盘']
    return data