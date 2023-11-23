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
import traceback
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
    data['Buy_Signal'] = (data['涨跌幅'] < 0.95 * data['Max_rate'])
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


def gen_all_signal_processing(args, threshold_day=1, is_skip=False):
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
            full_name, final_combinations, gen_signal_func, backtest_func, zero_combination = args
            data = load_data(full_name)
            data = gen_signal_func(data)
            file_name = Path('../back/zuhe') / f"{data['名称'].iloc[0]}.json"
            file_name.parent.mkdir(parents=True, exist_ok=True)

            result_df_dict = read_json(file_name)
            for index, combination in enumerate(final_combinations, start=1):
                combination_key = ':'.join(combination)
                if is_skip:
                    if combination_key in result_df_dict:
                        print(f"combination {combination_key} in result_df_dict")
                        continue
                if any(comb in zero_combination for comb in combination):
                    print(f"combination {combination_key} in zero_combination")
                    continue

                signal_data = gen_signal(data, combination)
                # 如果signal_data的Buy_Signal全为False，则不进行回测
                if not signal_data['Buy_Signal'].any():
                    zero_combination.add(combination_key)
                    result_df_dict[combination_key] = {
                        'trade_count': 0,
                        'total_profit': 0,
                        'size_of_result_df': 0,
                        'total_days_held': 0
                    }
                    continue
                results_df = backtest_func(signal_data)
                processed_result = process_results(results_df, threshold_day)
                if processed_result:
                    result_df_dict[combination_key] = processed_result
        except Exception as e:
            traceback.print_exc()
            print(full_name)
        finally:
            write_json(file_name, result_df_dict)
    except Exception as e:
        # 输出异常栈
        traceback.print_exc()


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
        return {
            'trade_count': 0,
            'total_profit': 0,
            'size_of_result_df': 0,
            'total_days_held': 0
        }
    total_days_held = result['Days Held'].sum()
    result_df = result[result['Days Held'] > threshold_day]
    result_shape = result.shape[0]
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
        if name.startswith('gen_basic_daily_buy_signal_1'):
            data = func(data)
    return data

def get_combination_list(basic_indicators, newest_indicators, exist_combinations_set, zero_combinations_set):
    """
    获取所有的组合,并进行去重操作
    :param basic_indicators:
    :param newest_indicators:
    :param exist_combinations:
    :return:
    """
    # 定义结果集合
    result_combination_list = []
    full_combination_list = []

    # 如果 newest_indicators 为空，仅处理 basic_indicators
    if newest_indicators == [[]]:
        for basic_indicator in basic_indicators:
            basic_indicator_key = ':'.join(basic_indicator)
            if basic_indicator_key not in exist_combinations_set:
                result_combination_list.append(basic_indicator)
            full_combination_list.append(basic_indicator)
    else:
        # 将 indicators 转换为集合以提高效率
        # basic_indicators是一个二维数组，先将每一个元素以':'连接，然后再转换为集合
        basic_indicators_sets = [set(':'.join(basic_indicator).split(':')) for basic_indicator in basic_indicators]
        newest_indicators_sets = [set(':'.join(newest_indicator).split(':')) for newest_indicator in newest_indicators]

        # 生成组合
        for basic_set in basic_indicators_sets:
            for newest_set in newest_indicators_sets:
                if not basic_set & newest_set:  # 检查交集
                    combination = ':'.join(sorted(basic_set | newest_set))  # 合并集合并排序
                    combination_list = combination.split(':')
                    full_combination_list.append(combination_list)
                    if combination not in exist_combinations_set:
                        # 如果combination被包含在zero_combinations_set元素中，则不添加到result_combination_list中
                        if not any([set(combination_list) >= set(zero_combination.split(':')) for zero_combination in zero_combinations_set]):
                            result_combination_list.append(combination_list)
    # 对result_combination_list和full_combination_list进行去重
    result_combination_list = list(set([tuple(t) for t in result_combination_list]))
    full_combination_list = list(set([tuple(t) for t in full_combination_list]))
    return result_combination_list, full_combination_list

def filter_combination_list(combination_list, statistics, zero_combinations_set, trade_count_threshold=1000):
    """
    过滤掉交易次数小于1000的组合
    :param combination_list:
    :param statistics:
    :param trade_count_threshold:
    :return:
    """
    result_combination_list = []
    for combination in combination_list:
        combination_key = ':'.join(combination)
        if combination_key not in statistics or statistics[combination_key]['trade_count'] >= trade_count_threshold:
            if not any([set(combination) >= set(zero_combination.split(':')) for zero_combination in
                        zero_combinations_set]):
                result_combination_list.append(combination)
    return result_combination_list

def back_layer_all(file_path, gen_signal_func=gen_full_all_basic_signal, backtest_func=backtest_strategy_low_profit):
    """
    分层进行回测
    1.先获取第一层指标(带signal的)，并且读取已有的统计信息
    2.排除掉已统计好的指标，计算剩下的指标，并将新计算的结果加入统计文件里面
    3.读取统计信息，找到满足条件的指标(交易数量大于1000)
    4.将基础指标和满足要求的指标(最新一层)进行两两组合（需要进行重复性判断:1.以‘:’进行分割后变成集合，如果长度减小那就说明有重复 2.过了第一个条件之后又拼接成字符串，判断是否已经在指标中）
    5.计算这些新的指标的回测结果，并将结果写入统计文件
    6.如果新指标没有满足条件的话那么完成统计，否则继续第3步
    :param data:
    :return:
    """
    level = 0
    # 读取statistics.json文件，得到已有的key
    statistics = read_json('../back/statistics.json')
    # 找到trade_count为0的指标，将key保持在zero_combinations_set中
    zero_combinations_set = set()
    for key, value in statistics.items():
        if value['trade_count'] == 0:
            zero_combinations_set.add(key)
    exist_combinations_set = set(statistics.keys())
    data = load_data('../daily_data_exclude_new/龙洲股份_002682.txt')
    data = gen_signal_func(data)
    signal_columns = [column for column in data.columns if 'signal' in column]
    # 将每个信号单独组成一个组合
    basic_indicators = [[column] for column in signal_columns]
    basic_indicators = filter_combination_list(basic_indicators, statistics, zero_combinations_set)
    newest_indicators = [[]]
    while True:
        result_combination_list, full_combination_list = get_combination_list(basic_indicators, newest_indicators, exist_combinations_set, zero_combinations_set)

        print(f'level:{level}, basic_indicators:{len(basic_indicators)}, newest_indicators:{len(newest_indicators)}, result_combination_list:{len(result_combination_list)}, full_combination_list:{len(full_combination_list)}')
        level += 1
        newest_indicators = filter_combination_list(full_combination_list, statistics, zero_combinations_set)
        if result_combination_list == []:
            if newest_indicators == []:
                break
            continue
        # 筛选出full_combination_list中的指标在statistics中对应的数据
        # 准备多进程处理的任务列表
        tasks = []
        # gen_all_signal_processing(('../daily_data_exclude_new/ST国嘉_600646.txt', final_combinations, gen_signal_func, backtest_func))
        for root, ds, fs in os.walk(file_path):
            for f in fs:
                fullname = os.path.join(root, f)
                tasks.append((fullname, result_combination_list, gen_signal_func, backtest_func, zero_combinations_set))

        # 使用进程池处理任务
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)

        total_files = len(tasks)
        for i, _ in enumerate(pool.imap_unordered(gen_all_signal_processing, tasks), 1):
            print(f"Processing file {i} of {total_files}...")

        pool.close()
        pool.join()
        statistics_zuhe('../back/zuhe')
        statistics = read_json('../back/statistics.json')
        exist_combinations_set = set(statistics.keys())
        zero_combinations_set = set()
        for key, value in statistics.items():
            if value['trade_count'] == 0:
                zero_combinations_set.add(key)

def back_sigle_all(file_path, gen_signal_func=gen_full_all_basic_signal, backtest_func=backtest_strategy_low_profit):
    """
    回测每个基础指标，方便后续过滤某些指标
    :param data:
    :return:
    """
    data = load_data('../daily_data_exclude_new/龙洲股份_002682.txt')
    data = gen_signal_func(data)
    signal_columns = [column for column in data.columns if 'signal' in column]
    # 将每个信号单独组成一个组合
    final_combinations = [[column] for column in signal_columns]
    # 准备多进程处理的任务列表
    tasks = []
    # gen_all_signal_processing(('../daily_data_exclude_new/ST国嘉_600646.txt', final_combinations, gen_signal_func, backtest_func))
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
        if value['trade_count'] != 0:
            value['ratio'] = value['size_of_result_df'] / value['trade_count']
            value['average_days_held'] = value['total_days_held'] / value['trade_count']
            value['average_profit'] = value['total_profit'] / value['trade_count']
            # 将value['ratio']保留4位小数
            value['ratio'] = round(value['ratio'], 4)
            value['average_profit'] = round(value['average_profit'], 4)
            value['average_days_held'] = round(value['average_days_held'], 4)
            value['total_profit'] = round(value['total_profit'], 4)
        else:
            value['ratio'] = 1
            value['average_profit'] = 0
            value['average_days_held'] = 0
            value['total_profit'] = 0
    # 将resul trade_count降序排序，然后在此基础上再按照ratio升序排序
    result = sorted(result.items(), key=lambda x: x[1]['trade_count'], reverse=True)
    result = sorted(result, key=lambda x: x[1]['ratio'])
    # 将result转换成dict
    result = dict(result)
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
