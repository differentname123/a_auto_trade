# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/19 18:14
:last_date:
    2023/10/19 18:14
:description:

"""
import inspect
import itertools
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from StrategyExecutor import basic_daily_strategy
from common import *
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


def untie_combinations(merged_list):
    """
    merge_list中的元素有些不是str而是列表，需要将它们展开
    :param merged_list:
    :return:
    """
    untied_list = []
    for item in merged_list:
        if isinstance(item, str):
            untied_list.append(item)
        else:
            untied_list.extend(item)
    # 将untied_list进行排序
    untied_list.sort()
    return untied_list


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
        filtered_final_combinations.append(untie_combinations(merged_list))

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
    ma_columns_dict = split_columns(columns, '日均线')
    ma_columns_combination = generate_combinations(ma_columns_dict)

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
    # 将final_combinations按照元素长度排序
    final_combinations.sort(key=lambda x: len(x), reverse=False)
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


def gen_signal(data, combination):
    """
    生成信号
    :param data:
    :param combination:
    :return:
    """
    # 获取combination中的列名，然后作为key进行与操作

    for column in combination:
        data['Buy_Signal'] = data['Buy_Signal'] & data[column]
    return data


def is_combination_in_zero_combination(combination, zero_combination):
    """
    检查 combination 是否为 zero_combination 中任意列表的子集
    :param combination: 一个列表
    :param zero_combination: 一个二维列表
    :return: 如果 combination 是 zero_combination 中任意列表的子集，则返回 True，否则返回 False
    """
    set_combination = set(combination)
    for zero_comb in zero_combination:
        if set(zero_comb).issubset(set_combination):
            return True
    return False


def gen_all_signal(data, final_combinations, backtest_func=backtest_strategy_low_profit, threshold_day=1):
    """
    Generate all signals based on final_combinations.
    :param data: DataFrame containing the data.
    :param final_combinations: Iterable of combinations to generate signals for.
    :param backtest_func: Function to use for backtesting. Default is backtest_strategy_low_profit.
    :param threshold_day: Threshold for 'Days Held' to filter the results.
    :return: None, writes results to a JSON file.
    """
    file_name = Path('../back/zuhe') / f"{data['名称'].iloc[0]}.json"
    file_name.parent.mkdir(parents=True, exist_ok=True)
    gen_all_basic_signal(data)
    result_df_dict = read_json(file_name)
    zero_combination = set()  # Using a set for faster lookups
    try:
        total_combinations = len(final_combinations)
        for index, combination in enumerate(final_combinations, start=1):
            # Print progress every 100 combinations to reduce print overhead
            if index % 100 == 0:
                print(f"Processing {file_name} combination {index} of {total_combinations}...")

            combination_key = ':'.join(combination)
            if combination_key in result_df_dict or any(comb in zero_combination for comb in combination):
                continue

            signal_data = gen_signal(data, combination)
            results_df = backtest_func(signal_data)
            if results_df.empty:
                zero_combination.update(combination_key)
                continue

            processed_result = process_results(results_df, threshold_day)
            if processed_result:
                result_df_dict[combination_key] = processed_result
    except KeyboardInterrupt:
        print("KeyboardInterrupt, saving results...")
    finally:
        write_json(file_name, result_df_dict)


def gen_all_signal_processing(args, threshold_day=1):
    """
    Generate all signals based on final_combinations.
    :param data: DataFrame containing the data.
    :param final_combinations: Iterable of combinations to generate signals for.
    :param backtest_func: Function to use for backtesting. Default is backtest_strategy_low_profit.
    :param threshold_day: Threshold for 'Days Held' to filter the results.
    :return: None, writes results to a JSON file.
    """
    try:
        try:
            full_name, final_combinations, gen_signal_func, backtest_func = args
            data = load_data(full_name)
            gen_signal_func(data)
            file_name = Path('../back/zuhe') / f"{data['名称'].iloc[0]}.json"
            file_name.parent.mkdir(parents=True, exist_ok=True)

            result_df_dict = read_json(file_name)
            zero_combination = set()  # Using a set for faster lookups
            for index, combination in enumerate(final_combinations, start=1):
                combination_key = ':'.join(combination)
                if combination_key in result_df_dict or any(comb in zero_combination for comb in combination):
                    continue

                signal_data = gen_signal(data, combination)
                results_df = backtest_func(signal_data)
                if results_df.empty:
                    zero_combination.add(combination_key)
                    continue

                processed_result = process_results(results_df, threshold_day)
                if processed_result:
                    result_df_dict[combination_key] = processed_result
        except Exception as e:
            print(e)
            print(full_name)
        finally:
            write_json(file_name, result_df_dict)
    except Exception as e:
        print(e)


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def process_results(results_df, threshold_day):
    result = results_df.sort_values(by='Days Held', ascending=True)
    if result.empty:
        return None

    result_df = result[result['Days Held'] > threshold_day]
    result_shape = result.shape[0]
    if result_df.empty:
        total_days_held = 0
    else:
        total_days_held = result_df['Days Held'].sum()
    return {
        'trade_count': result_shape,
        'total_profit': round(result['Total_Profit'].iloc[-1], 4),
        'size_of_result_df': result_df.shape[0],
        'total_days_held': int(total_days_held)
    }


def gen_all_basic_signal(data):
    gen_basic_daily_buy_signal_1(data)
    gen_basic_daily_buy_signal_2(data)
    gen_basic_daily_buy_signal_3(data)
    gen_basic_daily_buy_signal_4(data)
    return data


def gen_basic_daily_buy_signal_yesterday(data, key):
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
        data[column + 'yesterday'] = (data[column].shift(1) == True)
    return data


def back_zuhe_all(file_path, backtest_func=backtest_strategy_low_profit):
    """
    生成所有的组合及进行相应的回测
    :param data:
    :return:
    """
    data = load_data('../daily_data_exclude_new/龙洲股份_002682.txt')
    gen_all_basic_signal(data)
    signal_columns = [column for column in data.columns if 'signal' in column]
    final_combinations = deal_columns(data, signal_columns)

    # 准备多进程处理的任务列表
    tasks = []
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            fullname = os.path.join(root, f)
            tasks.append((fullname, final_combinations, backtest_func))

    # 使用进程池处理任务
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)

    total_files = len(tasks)
    for i, _ in enumerate(pool.imap_unordered(gen_all_signal_processing, tasks), 1):
        print(f"Processing file {i} of {total_files}...")

    pool.close()
    pool.join()
    statistics_zuhe('../back/zuhe')


def gen_full_all_basic_signal(data):
    """
    扫描basic_daily_strategy.py文件，生成所有的基础信号,过滤出以gen_basic_daily_buy_signal开头的函数，并应用于data
    :param data: 
    :return: 
    """
    # 扫描basic_daily_strategy.py文件，生成所有的基础信号
    for name, func in inspect.getmembers(basic_daily_strategy, inspect.isfunction):
        if name.startswith('gen_basic_daily_buy_signal'):
            func(data)
    return data


def back_sigle_all(file_path, gen_signal_func=gen_full_all_basic_signal, backtest_func=backtest_strategy_low_profit):
    """
    回测每个基础指标，方便后续过滤某些指标
    :param data:
    :return:
    """
    data = load_data('../daily_data_exclude_new/龙洲股份_002682.txt')
    gen_signal_func(data)
    signal_columns = [column for column in data.columns if 'signal' in column]
    # 将每个信号单独组成一个组合
    final_combinations = [[column] for column in signal_columns]
    # 准备多进程处理的任务列表
    tasks = []
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            fullname = os.path.join(root, f)
            tasks.append((fullname, final_combinations, gen_signal_func, backtest_func))

    # 使用进程池处理任务
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)

    total_files = len(tasks)
    for i, _ in enumerate(pool.imap_unordered(gen_all_signal_processing, tasks), 1):
        print(f"Processing file {i} of {total_files}...")

    pool.close()
    pool.join()
    statistics_zuhe('../back/zuhe')


def statistics_zuhe(file_path):
    """
    读取file_path下的所有.json文件，将有相同key的数据进行合并
    :param file_path:
    :return:
    """
    result = {}
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            fullname = os.path.join(root, f)
            data = read_json(fullname)
            for key, value in data.items():
                if key in result:
                    result[key]['trade_count'] += value['trade_count']
                    result[key]['total_profit'] += value['total_profit']
                    result[key]['size_of_result_df'] += value['size_of_result_df']
                    result[key]['total_days_held'] += value['total_days_held']
                else:
                    result[key] = value
    # 再计算result每一个key的平均值
    for key, value in result.items():
        value['ratio'] = value['size_of_result_df'] / value['trade_count']
        value['average_days_held'] = value['total_days_held'] / value['trade_count']
        value['average_profit'] = value['total_profit'] / value['trade_count']
        # 将value['ratio']保留4位小数
        value['ratio'] = round(value['ratio'], 4)
        value['average_profit'] = round(value['average_profit'], 4)
        value['average_days_held'] = round(value['average_days_held'], 4)
        value['total_profit'] = round(value['total_profit'], 4)
    # 将resul trade_count降序排序，然后在此基础上再按照ratio升序排序
    result = sorted(result.items(), key=lambda x: x[1]['trade_count'], reverse=True)
    result = sorted(result, key=lambda x: x[1]['ratio'])
    # 将result写入file_path上一级文件
    file_name = Path(file_path).parent / 'statistics.json'
    write_json(file_name, result)
    return result


def back_zuhe(file_path, backtest_func=backtest_strategy_low_profit):
    """
    生成所有的组合及进行相应的回测
    :param data:
    :return:
    """
    data = load_data(file_path)
    gen_all_basic_signal(data)
    signal_columns = [column for column in data.columns if 'signal' in column]
    final_combinations = deal_columns(data, signal_columns)
    gen_all_signal(data, final_combinations, backtest_func)
    return data
