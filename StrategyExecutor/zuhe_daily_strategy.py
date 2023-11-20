# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/19 18:14
:last_date:
    2023/10/19 18:14
:description:

"""
import itertools

import numpy as np
import pandas as pd

from StrategyExecutor.MyTT import *
from StrategyExecutor.basic_daily_strategy import *
from StrategyExecutor.common import load_data


def gen_combination(value_columns):
    """
    生成value_columns的所有组合的二维列表，每个元素是一个列表，包含value_columns字段的组合
    :param value_columns:
    :return:
    """
    value_columns_combination = []
    for i in range(1, len(value_columns) + 1):
        value_columns_combination.extend(list(itertools.combinations(value_columns, i)))
    return value_columns_combination

def split_columns(columns, keyword):
    """
    根据关键词拆分列并构建字典
    """
    filtered_columns = [column for column in columns if keyword in column]
    columns_dict = {}
    for column in filtered_columns:
        key = column.split('_')[0]
        columns_dict.setdefault(key, []).append(column)
    return columns_dict

def generate_combinations(columns_dict):
    """
    生成列组合
    """
    # 为每个列表添加一个表示“不选择”（None）的选项
    values_lists = [values + [None] for values in columns_dict.values()]
    # 使用 itertools.product 生成所有可能的组合
    combinations = list(itertools.product(*values_lists))
    # 过滤掉所有元素都是 None 的组合，并移除组合中的 None
    return [tuple(v for v in combo if v is not None)
            for combo in combinations if not all(v is None for v in combo)]

def generate_final_combinations(*combinations_lists):
    """
    从多个组合列表中生成最终的组合
    :param combinations_lists: 可变数量的组合列表
    :return: 最终的组合列表
    """
    # 为每个组合列表添加一个表示“不选择”（None）的选项
    modified_lists = [lst + [None] for lst in combinations_lists]

    # 生成所有可能的组合
    final_combinations = list(itertools.product(*modified_lists))

    # 移除组合中的 None，并合并元组中的列表
    filtered_final_combinations = []
    for combo in final_combinations:
        # 过滤掉 None 并合并列表
        merged_list = [item for sublist in combo if sublist is not None for item in sublist]
        filtered_final_combinations.append(merged_list)

    return filtered_final_combinations

def deal_columns(data, columns):
    """
    按照相应条件处理相应的列
    :param data:
    :param columns:
    :return:
    """

    # 处理固定区间列
    value_columns_dict = split_columns(columns, '固定区间')
    value_columns_combination = generate_combinations(value_columns_dict)

    # 处理日均线列，特殊处理逻辑
    ma_columns = [column for column in columns if '日均线' in column]
    # 类似于对value_columns的处理，对ma_columns进行处理
    ma_columns_dict = {}
    for column in ma_columns:
        key = column.split('_')[0]
        if key not in ma_columns_dict:
            ma_columns_dict[key] = []
        ma_columns_dict[key].append(column)
    # 将ma_columns_dict中的value转换成所有的组合
    ma_columns_combination = []
    for key, value in ma_columns_dict.items():
        ma_columns_combination.append(gen_combination(value))

    # 处理极值列
    max_min_columns_dict = split_columns(columns, '极值')
    max_min_columns_combination = generate_combinations(max_min_columns_dict)

    processed_columns = [col for cols in [value_columns_dict, ma_columns_dict, max_min_columns_dict]
                         for col in cols.values()]
    flattened_processed_columns = [item for sublist in processed_columns for item in sublist]

    # 处理其他列
    other_columns = [column for column in columns if column not in flattened_processed_columns]
    other_columns_dict = split_columns(other_columns, '')
    other_columns_combination = generate_combinations(other_columns_dict)

    final_combinations = generate_final_combinations(value_columns_combination, ma_columns_combination,
                                                     max_min_columns_combination, other_columns_combination)

    return final_combinations


def deal_colu1mns(data, columns):
    """
    按照相应条件处理相应的列
    :param data:
    :param columns:
    :return:
    """

    value_columns = [column for column in columns if '固定区间' in column]
    # 将value_columns按照列名用_分割，取第一个元素，作为字典的key，值是一个列表，列表中的值是拥有共同以key开头的value_columns的元素
    value_columns_dict = {}
    for column in value_columns:
        key = column.split('_')[0]
        if key not in value_columns_dict:
            value_columns_dict[key] = []
        value_columns_dict[key].append(column)

    # 将value_columns_dict中的value转换成所有的组合
    value_columns_combination = []
    for key, value in value_columns_dict.items():
        value_columns_combination.append(gen_combination(value))


    ma_columns = [column for column in columns if '日均线' in column]
    # 类似于对value_columns的处理，对ma_columns进行处理
    ma_columns_dict = {}
    for column in ma_columns:
        key = column.split('_')[0]
        if key not in ma_columns_dict:
            ma_columns_dict[key] = []
        ma_columns_dict[key].append(column)
    # 将ma_columns_dict中的value转换成所有的组合
    ma_columns_combination = []
    for key, value in ma_columns_dict.items():
        ma_columns_combination.append(gen_combination(value))

    max_min_columns = [column for column in columns if '极值' in column]

    # 类似于对value_columns的处理，对max_min_columns进行处理
    max_min_columns_dict = {}
    for column in max_min_columns:
        key = column.split('_')[0]
        if key not in max_min_columns_dict:
            max_min_columns_dict[key] = []
        max_min_columns_dict[key].append(column)

    # 为每个列表添加一个表示“不选择”（None）的选项
    values_lists = [values + [None] for values in max_min_columns_dict.values()]
    # 使用 itertools.product 生成所有可能的组合
    combinations = list(itertools.product(*values_lists))
    # 过滤掉所有元素都是 None 的组合，并移除组合中的 None
    max_min_columns_combination = [tuple(v for v in combo if v is not None)
                             for combo in combinations if not all(v is None for v in combo)]


    other_columns = [column for column in columns if column not in value_columns + ma_columns + max_min_columns]

    other_columns_dict = {}
    for column in other_columns:
        key = column.split('_')[0]
        if key not in other_columns_dict:
            other_columns_dict[key] = []
        other_columns_dict[key].append(column)

    # 为每个列表添加一个表示“不选择”（None）的选项
    values_lists = [values + [None] for values in other_columns_dict.values()]
    # 使用 itertools.product 生成所有可能的组合
    combinations = list(itertools.product(*values_lists))
    # 过滤掉所有元素都是 None 的组合，并移除组合中的 None
    other_columns_combination = [tuple(v for v in combo if v is not None)
                             for combo in combinations if not all(v is None for v in combo)]


def gen_all_signal(file_path):
    """
    生成所有的信号
    :param data:
    :return:
    """
    data = load_data(file_path)
    gen_basic_daily_buy_signal_1(data)
    gen_basic_daily_buy_signal_2(data)
    gen_basic_daily_buy_signal_3(data)
    gen_basic_daily_buy_signal_4(data)
    signal_columns = [column for column in  data.columns if 'signal' in column]
    result = deal_columns(data, signal_columns)
    return result


def zuhe_fun(data):
    """
    注意先将同一个值的ma组合和value组合得到之后再和其它的字段进行组合
    :return:
    """