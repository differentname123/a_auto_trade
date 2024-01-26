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
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import os
import json
import concurrent.futures
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from StrategyExecutor import basic_daily_strategy, basic_zhishu_strategy
from common import *
from StrategyExecutor.MyTT import *
from StrategyExecutor.basic_daily_strategy import *
from StrategyExecutor.common import load_data

def filter_combinations_good(zero_combinations_set, good_keys):
    """
    过滤已存在和无效的组合
    """
    final_combinations_set = good_keys
    return [comb for comb in final_combinations_set if
            not any(frozenset(comb) >= zc for zc in zero_combinations_set)], zero_combinations_set

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
    优化版本的生成信号函数
    :param data: 输入的 DataFrame
    :param combination: 列名组合的列表
    :return: 带有 Buy_Signal 列的 DataFrame
    """
    # 预先计算涨跌幅条件
    buy_condition = ((data['涨跌幅'] >= -(data['Max_rate'] - 1.0 / data['收盘'])) & (data['涨跌幅'] <= (data['Max_rate'] - 1.0 / data['收盘'])) & (data['收盘'] >= 3) & (data['Max_rate'] > 0.1))

    # 使用 reduce 函数结合所有的条件
    combined_condition = np.logical_and.reduce([data[col] for col in combination])

    # 应用所有条件生成 Buy_Signal
    data['Buy_Signal'] = buy_condition & combined_condition

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


def gen_all_signal_processing_good(args, threshold_day=1, is_skip=False):
    """
    Generate all signals based on final_combinations.
    Optimized to reduce file operations and improve efficiency in loops.
    """
    start_time = time.time()
    try:
        zero_combination = set()  # Using a set for faster lookups
        full_name, final_combinations, gen_signal_func, backtest_func = args

        data = pd.read_csv(full_name, low_memory=False)
        # 获取full_name这个路径的文件名，去掉后缀，再去掉_后面的内容，得到名称
        mingcheng = os.path.basename(full_name).split('.')[0].split('_')[0]
        # 如果data长度小于100，不生成信号
        # data = gen_signal_func(data)
        data['Buy Date'] = pd.to_datetime(data['Buy Date'])
        file_name = Path('../final_zuhe/zuhe') / f"{mingcheng}.json"
        # aother_file_name = Path('../back/gen/zuhe') / f"{data['名称'].iloc[0]}.json"
        # file_name = Path('../back/zuhe') / f"C夏厦.json"
        file_name.parent.mkdir(parents=True, exist_ok=True)

        # 一次性读取JSON
        result_df_dict = {}
        # aother_result_df_dict = read_json(aother_file_name)
        # 将aother_result_df_dict合并到result_df_dict
        # 如果data长度小于100，不生成信号
        # if data.shape[0] < 100:
        #     # 打印相应的日志
        #     print(f"Processing {full_name} length {data.shape[0]} less than 100, skip...")
        #     return
        if is_skip:
            # 优化：过滤和处理逻辑提取为单独的函数
            final_combinations, zero_combination = filter_combinations(result_df_dict, final_combinations)

        # 处理每个组合
        for combination in final_combinations:
            combination_key = ':'.join(combination)
            signal_data = gen_signal(data, combination)

            # results_df = backtest_func(signal_data)
            results_df = signal_data[signal_data['Buy_Signal'] == True]
            processed_result = process_results_with_every_period(results_df, threshold_day)

            if processed_result:
                result_df_dict[combination_key] = processed_result
        # aother_result_df_dict.update(result_df_dict)
        write_json(file_name, result_df_dict)
        # write_json(aother_file_name, aother_result_df_dict)
        end_time = time.time()
        print(
            f"{full_name} 耗时：{end_time - start_time}秒 data长度{data.shape[0]} zero_combination len: {len(zero_combination)} final_combinations len: {len(final_combinations)}")


    except Exception as e:
        traceback.print_exc()
    finally:
        # 写入文件
        pass

def gen_all_signal_processing_gen(args, threshold_day=1, is_skip=True):
    """
    Generate all signals based on final_combinations.
    Optimized to reduce file operations and improve efficiency in loops.
    """
    start_time = time.time()
    try:
        zero_combination = set()  # Using a set for faster lookups
        full_name, final_combinations, gen_signal_func, backtest_func = args
        mingcheng = full_name.split('/')[-1].split('.')[0].split('_')[0]
        data = load_data(full_name)
        data = gen_signal_func(data)
        file_name = Path('../back/gen/zuhe') / f"{mingcheng}.json"
        # file_name = Path('../back/zuhe') / f"C夏厦.json"
        file_name.parent.mkdir(parents=True, exist_ok=True)

        # 一次性读取JSON
        result_df_dict = read_json(file_name)

        if is_skip:
            # 优化：过滤和处理逻辑提取为单独的函数
            final_combinations, zero_combination = filter_combinations(result_df_dict, final_combinations)

        # 处理每个组合
        for combination in final_combinations:
            combination_key = ':'.join(combination)
            signal_data = gen_signal(data, combination)

            # 如果Buy_Signal全为False，则不进行回测
            if not signal_data['Buy_Signal'].any():
                result_df_dict[combination_key] = create_empty_result()
                continue

            results_df = backtest_func(signal_data)
            processed_result = process_results_with_year(results_df, threshold_day)

            if processed_result:
                result_df_dict[combination_key] = processed_result

    except Exception as e:
        traceback.print_exc()
    finally:
        # 写入文件
        write_json(file_name, result_df_dict)
        end_time = time.time()
        print(
            f"{full_name} 耗时：{end_time - start_time}秒 data长度{data.shape[0]} zero_combination len: {len(zero_combination)} final_combinations len: {len(final_combinations)}")

def gen_all_signal_processing_with_back_data(args, threshold_day=1, is_skip=True):
    """
    会将本次结果单独存储一份
    """
    start_time = time.time()
    try:
        zero_combination = set()  # Using a set for faster lookups
        origin_file_name, task_data, final_combinations = args

        data = task_data
        file_name = '../back/gen/with_back_data/' + origin_file_name
        recent_file_name = '../back/gen/with_back_data_single/' + origin_file_name
        # file_name = Path('../back/zuhe') / f"C夏厦.json"
        # file_name.parent.mkdir(parents=True, exist_ok=True)

        recent_result_df_dict = {}
        # 处理每个组合
        for combination in final_combinations:
            combination_key = ':'.join(combination)
            signal_data = gen_signal(data, combination)

            # 如果Buy_Signal全为False，则不进行回测
            if not signal_data['Buy_Signal'].any():
                # result_df_dict[combination_key] = create_empty_result()
                # recent_result_df_dict[combination_key] = create_empty_result()
                continue
            # 获取signal_data中Buy_Signal为True的数据
            results_df = signal_data[signal_data['Buy_Signal'] == True]
            processed_result = process_results_with_year(results_df, threshold_day)

            if processed_result:
                recent_result_df_dict[combination_key] = processed_result
        # result_df_dict = read_json(file_name)
        # result_df_dict.update(recent_result_df_dict)
        write_json(recent_file_name, recent_result_df_dict)
        # write_json(file_name, result_df_dict)

    except Exception as e:
        traceback.print_exc()
    finally:
        # 写入文件
        end_time = time.time()
        print(
            f"{file_name} 耗时：{end_time - start_time}秒 data长度{data.shape[0]} zero_combination len: {len(zero_combination)} final_combinations len: {len(final_combinations)}")


def gen_all_signal_processing_gen_single_file(args, threshold_day=1, is_skip=False):
    """
    会将本次结果单独存储一份
    """
    start_time = time.time()
    try:
        zero_combination = set()  # Using a set for faster lookups
        full_name, final_combinations, gen_signal_func, backtest_func = args

        data = pd.read_csv(full_name, low_memory=False)
        # data = gen_signal_func(data)
        data['Buy Date'] = pd.to_datetime(data['Buy Date'])
        mingcheng = full_name.split('/')[-1].split('.')[0].split('_')[0]
        # file_name = Path('../back/gen/zuhe') / f"{data['名称'].iloc[0]}.json"
        recent_file_name = Path('../back/gen/single') / f"{mingcheng}.json"
        # file_name = Path('../back/zuhe') / f"C夏厦.json"
        # file_name.parent.mkdir(parents=True, exist_ok=True)
        if is_skip:
            recent_result_df_dict = read_json(recent_file_name)
            # 获取full_name的父目录
            out_put_file_path = os.path.dirname(full_name)
            false_columns_output_filename = os.path.join('{}_false'.format(out_put_file_path), '{}false_columns.txt'.format(os.path.basename(full_name)))
            false_columns = set()
            if os.path.exists(false_columns_output_filename):
                with open(false_columns_output_filename, 'r') as lines:
                    for line in lines:
                        # 假设每行是以逗号分隔的元素
                        elements = line.strip().split(',')
                        false_columns.add(frozenset(elements))

            final_combinations, zero_combination = filter_combinations_good(false_columns, final_combinations)
            final_combinations, zero_combination = filter_combinations(recent_result_df_dict, final_combinations)


        recent_result_df_dict = {}
        # 处理每个组合
        for combination in final_combinations:
            combination_key = ':'.join(combination)
            signal_data = gen_signal(data, combination)

            # 如果Buy_Signal全为False，则不进行回测
            # if not signal_data['Buy_Signal'].any():
            #     # result_df_dict[combination_key] = create_empty_result()
            #     recent_result_df_dict[combination_key] = create_empty_result()
            #     continue

            # origin_results_df = backtest_func(signal_data)
            results_df = signal_data[signal_data['Buy_Signal'] == True]
            # if results_df.shape[0] != origin_results_df.shape[0]:
            #     print('diff:')
            #     print(origin_results_df)
            #     print(results_df)
            processed_result = process_results_with_every_period(results_df, threshold_day)

            if processed_result:
                recent_result_df_dict[combination_key] = processed_result
        # result_df_dict = read_json(file_name)
        # result_df_dict.update(recent_result_df_dict)
        write_json(recent_file_name, recent_result_df_dict)
        # write_json(file_name, result_df_dict)

    except Exception as e:
        traceback.print_exc()
    finally:
        # 写入文件
        end_time = time.time()
        # 获取当前时间,保留到分钟
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        print(
            f"{now}_{recent_file_name} 耗时：{end_time - start_time}秒 data长度{data.shape[0]} zero_combination len: {len(zero_combination)} final_combinations len: {len(final_combinations)}")


def gen_all_signal_processing_op(args, threshold_day=1, is_skip=True):
    """
    Generate all signals based on final_combinations.
    Optimized to reduce file operations and improve efficiency in loops.
    """
    start_time = time.time()
    try:
        zero_combination = set()  # Using a set for faster lookups
        full_name, final_combinations, gen_signal_func, backtest_func = args

        data = load_data(full_name)
        data = gen_signal_func(data)
        file_name = Path('../back/zuhe') / f"{data['名称'].iloc[0]}.json"
        # file_name = Path('../back/zuhe') / f"C夏厦.json"
        file_name.parent.mkdir(parents=True, exist_ok=True)

        # 一次性读取JSON
        result_df_dict = read_json(file_name)

        if is_skip:
            # 优化：过滤和处理逻辑提取为单独的函数
            final_combinations, zero_combination = filter_combinations(result_df_dict, final_combinations)

        # 处理每个组合
        for combination in final_combinations:
            combination_key = ':'.join(combination)
            signal_data = gen_signal(data, combination)

            # 如果Buy_Signal全为False，则不进行回测
            if not signal_data['Buy_Signal'].any():
                result_df_dict[combination_key] = create_empty_result()
                continue

            results_df = backtest_func(signal_data)
            processed_result = process_results(results_df, threshold_day)

            if processed_result:
                result_df_dict[combination_key] = processed_result

    except Exception as e:
        traceback.print_exc()
    finally:
        # 写入文件
        write_json(file_name, result_df_dict)
        end_time = time.time()
        print(
            f"{full_name} 耗时：{end_time - start_time}秒 data长度{data.shape[0]} zero_combination len: {len(zero_combination)} final_combinations len: {len(final_combinations)}")


def filter_combinations(result_df_dict, final_combinations):
    """
    过滤已存在和无效的组合
    """
    exist_combinations_set = set(result_df_dict.keys())
    final_combinations_set = {':'.join(combination) for combination in final_combinations}
    final_combinations_set -= exist_combinations_set
    # 获取result_df_dict中trade_count为0的组合
    zero_combination = {combination for combination, result in result_df_dict.items() if result['trade_count'] == 0}
    zero_combinations_set = {frozenset(z.split(':')) for z in zero_combination}
    return [comb.split(':') for comb in final_combinations_set if
            not any(frozenset(comb.split(':')) >= zc for zc in zero_combinations_set)], zero_combinations_set

def filter_combinations_op(result_df_dict, final_combinations):
    """
    过滤已存在和无效的组合
    """
    exist_combinations_set = set(result_df_dict.keys())
    final_combinations_set = {':'.join(combination) for combination in final_combinations}

    # 过滤掉已存在的组合
    filtered_combinations = final_combinations_set - exist_combinations_set

    # 获取result_df_dict中trade_count为0的组合，并转换为frozenset
    zero_combinations = {frozenset(comb.split(':')) for comb, result in result_df_dict.items() if result['trade_count'] == 0}

    # 过滤掉那些包含或等于trade_count为0的组合的组合
    valid_combinations = [comb.split(':') for comb in filtered_combinations
                          if not any(zero_comb.issubset(comb.split(':')) for zero_comb in zero_combinations)]

    return valid_combinations, zero_combinations


def create_empty_result():
    """
    创建空结果字典
    """
    return {'trade_count': 0,
            'total_profit': 0,
            'size_of_result_df': 0,
            'total_days_held': 0,
            'total_cost': 0,
            'one_befor_year_count': 0,
            'two_befor_year_count': 0,
            'three_befor_year_count_thread': 0,
            'three_befor_year_count': 0,
            'all_date': []}


def gen_all_signal_processing(args, threshold_day=1, is_skip=True):
    """
    Generate all signals based on final_combinations.
    :param data: DataFrame containing the data.
    :param final_combinations: Iterable of combinations to generate signals for.
    :param backtest_func: Function to use for backtesting. Default is backtest_strategy_low_profit.
    :param threshold_day: Threshold for 'Days Held' to filter the results.
    :return: None, writes results to a JSON file.
    """
    # 计时开始
    start_time = time.time()
    try:
        try:
            full_name, final_combinations, gen_signal_func, backtest_func, zero_combination = args

            data = load_data(full_name)
            data = gen_signal_func(data)
            file_name = Path('../back/zuhe') / f"{data['名称'].iloc[0]}.json"
            file_name.parent.mkdir(parents=True, exist_ok=True)

            result_df_dict = read_json(file_name)
            if is_skip:
                exist_combinations_set = set(result_df_dict.keys())
                # 将final_combinations转换为集合,需要将元素按照':'拼接，然后过滤掉exist_combinations_set中已经存在的组合
                final_combinations_set = {':'.join(combination) for combination in final_combinations}
                final_combinations_set = final_combinations_set - exist_combinations_set
                final_combinations = [combination.split(':') for combination in final_combinations_set]
                for key, value in result_df_dict.items():
                    if value['trade_count'] == 0:
                        zero_combination.add(key)
            zero_combinations_set = {frozenset(zero_combination.split(':')) for zero_combination in
                                     zero_combination}
            # 这里也需要加载对final_combinations的过滤
            for index, combination in enumerate(final_combinations, start=1):
                combination_key = ':'.join(combination)
                combination_set = frozenset(combination)
                if is_skip:
                    if combination_key in result_df_dict:
                        print(f"combination {combination_key} in result_df_dict")
                        continue
                if is_skip:
                    if any(combination_set >= zero_comb for zero_comb in zero_combinations_set):
                        # print(f"combination {combination_key} in zero_combination")
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
        finally:
            # 计时结束
            end_time = time.time()
            # 先删除文件，再写入文件，避免写入文件失败导致文件内容为空
            write_json(file_name, result_df_dict)
            print(
                f"{full_name} 耗时：{end_time - start_time}秒 data长度{data.shape[0]} zero_combination len: {len(zero_combination)}")
    except Exception as e:
        # 输出异常栈
        traceback.print_exc()


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path} exception: {e}")
        return {}


def write_json(file_path, data):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing {file_path} exception: {e}")

def process_results_with_every_year(results_df, threshold_day, dimension='year'):
    """
    统计每一年的回测情况
    :param results_df:
    :param threshold_day:
    :return:
    """
    result = results_df.sort_values(by='Days Held', ascending=True)
    if result.empty:
        print("不应该啊")
        return {
            'trade_count': 0,
            'total_profit': 0,
            'total_cost': 0,
            'size_of_result_df': 0,
            'one_befor_year_count': 0,
            'two_befor_year_count': 0,
            'three_befor_year_count': 0,
            'three_befor_year_count_thread':0,
            'total_days_held': 0,
            'all_date': []
        }
    # 将'Days Held'转换为时间格式
    result['Buy Date'] = pd.to_datetime(result['Buy Date'])
    # 统计'Buy Date'大于三年前的数据
    three_year = datetime.datetime.now() - relativedelta(years=2)
    # 将three_year只保留到年份，不要月和日
    three_befor_year = three_year.year
    two_year = datetime.datetime.now() - relativedelta(years=1)
    # 将three_year只保留到年份，不要月和日
    two_befor_year = two_year.year
    one_befor_year = datetime.datetime.now().year
    # 统计result中'Buy Date'大于three_year的个数
    three_befor_year_count = result[result['Buy Date'].dt.year >= three_befor_year].shape[0]
    two_befor_year_count = result[result['Buy Date'].dt.year >= two_befor_year].shape[0]
    one_befor_year_count = result[result['Buy Date'].dt.year >= one_befor_year].shape[0]
    total_days_held = result['Days Held'].sum()
    total_cost = result['total_cost'].sum()
    # Total_Profit=所有Profit的和
    Total_Profit = result['Profit'].sum()
    result_df = result[result['Days Held'] > threshold_day]
    result_shape = result.shape[0]
    three_befor_year_count_thread = result_df[result_df['Buy Date'].dt.year >= three_befor_year].shape[0]
    # 先将result['Buy Date']变成字符串类型
    result['Buy Date'] = result['Buy Date'].astype(str)
    # 获取results_df的所有日期
    all_date = result['Buy Date'].tolist()
    return {
        'trade_count': result_shape,
        'total_profit': round(Total_Profit, 4),
        'total_cost': round(total_cost, 4),
        'size_of_result_df': result_df.shape[0],
        'one_befor_year_count': one_befor_year_count,
        'two_befor_year_count': two_befor_year_count,
        'three_befor_year_count': three_befor_year_count,
        'three_befor_year_count_thread':three_befor_year_count_thread,
        'total_days_held': int(total_days_held),
        'all_date': all_date
    }

def process_results_with_every_period(results_df, threshold_day, dimension='year'):
    """
    :param results_df: DataFrame containing the results data
    :param threshold_day: Threshold for filtering based on 'Days Held'
    :param dimension: Dimension to group by ('year', 'month', or 'day')
    :return: Dictionary containing aggregated statistics
    """
    result = results_df.sort_values(by='Days Held', ascending=True)
    if result.empty:
        return {'all': {
            'trade_count': 0,
            'total_profit': 0,
            'total_cost': 0,
            'size_of_result_df': 0,
            'total_days_held': 0,
            'all_date': []
        }}
    result['Buy Date'] = pd.to_datetime(result['Buy Date'])
    # Group by the specified dimension
    if dimension == 'year':
        grouper = pd.Grouper(key='Buy Date', freq='Y')
    elif dimension == 'month':
        grouper = pd.Grouper(key='Buy Date', freq='M')
    else:  # 'day'
        grouper = pd.Grouper(key='Buy Date', freq='D')

    grouped_result = result.groupby(grouper)

    # Aggregate data and format as per the required output
    aggregated_data = {}
    for name, group in grouped_result:
        key = name.strftime('%Y') if dimension == 'year' else name.strftime('%Y-%m') if dimension == 'month' else name.strftime('%Y-%m-%d')
        aggregated_data[key] = {
            'trade_count': int(group.shape[0]),
            'total_profit': float(round(group['Profit'].sum(), 4)),
            'total_cost': float(round(group['total_cost'].sum(), 4)),
            'size_of_result_df': int(group[group['Days Held'] > threshold_day].shape[0]),
            'total_days_held': int(group['Days Held'].sum()),
            # 'all_date': group['Buy Date'].dt.strftime('%Y-%m-%d').tolist()
        }

    # Add overall statistics
    aggregated_data['all'] = {
        'trade_count': int(result.shape[0]),
        'total_profit': float(round(result['Profit'].sum(), 4)),
        'total_cost': float(round(result['total_cost'].sum(), 4)),
        'size_of_result_df': int(result[result['Days Held'] > threshold_day].shape[0]),
        'total_days_held': int(result['Days Held'].sum()),
        # 'all_date': result['Buy Date'].dt.strftime('%Y-%m-%d').tolist()
    }

    return aggregated_data
def process_results_with_year(results_df, threshold_day):
    """
    增加最近三年内的信号个数
    :param results_df:
    :param threshold_day:
    :return:
    """
    result = results_df.sort_values(by='Days Held', ascending=True)
    if result.empty:
        print("不应该啊")
        return {
            'trade_count': 0,
            'total_profit': 0,
            'total_cost': 0,
            'size_of_result_df': 0,
            'one_befor_year_count': 0,
            'two_befor_year_count': 0,
            'three_befor_year_count': 0,
            'three_befor_year_count_thread':0,
            'total_days_held': 0,
            'all_date': []
        }
    # 统计'Buy Date'大于三年前的数据
    three_year = datetime.datetime.now() - relativedelta(years=2)
    # 将three_year只保留到年份，不要月和日
    three_befor_year = three_year.year
    two_year = datetime.datetime.now() - relativedelta(years=1)
    # 将three_year只保留到年份，不要月和日
    two_befor_year = two_year.year
    one_befor_year = datetime.datetime.now().year
    # 统计result中'Buy Date'大于three_year的个数
    three_befor_year_count = result[result['Buy Date'].dt.year >= three_befor_year].shape[0]
    two_befor_year_count = result[result['Buy Date'].dt.year >= two_befor_year].shape[0]
    one_befor_year_count = result[result['Buy Date'].dt.year >= one_befor_year].shape[0]
    total_days_held = result['Days Held'].sum()
    total_cost = result['total_cost'].sum()
    # Total_Profit=所有Profit的和
    Total_Profit = result['Profit'].sum()
    result_df = result[result['Days Held'] > threshold_day]
    result_shape = result.shape[0]
    three_befor_year_count_thread = result_df[result_df['Buy Date'].dt.year >= three_befor_year].shape[0]
    # 先将result['Buy Date']变成字符串类型
    result['Buy Date'] = result['Buy Date'].astype(str)
    # 获取results_df的所有日期
    all_date = result['Buy Date'].tolist()
    return {
        'trade_count': result_shape,
        'total_profit': round(Total_Profit, 4),
        'total_cost': round(total_cost, 4),
        'size_of_result_df': result_df.shape[0],
        'one_befor_year_count': one_befor_year_count,
        'two_befor_year_count': two_befor_year_count,
        'three_befor_year_count': three_befor_year_count,
        'three_befor_year_count_thread':three_befor_year_count_thread,
        'total_days_held': int(total_days_held),
        'all_date': all_date
    }


def process_results(results_df, threshold_day):
    result = results_df.sort_values(by='Days Held', ascending=True)
    if result.empty:
        return {
            'trade_count': 0,
            'total_profit': 0,
            'total_cost': 0,
            'size_of_result_df': 0,
            'total_days_held': 0
        }
    total_days_held = result['Days Held'].sum()
    total_cost = result['total_cost'].sum()
    Total_Profit = results_df['Total_Profit'].iloc[-1]
    result_df = result[result['Days Held'] > threshold_day]
    result_shape = result.shape[0]
    return {
        'trade_count': result_shape,
        'total_profit': round(Total_Profit, 4),
        'total_cost': round(total_cost, 4),
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
    data = load_data('../daily_data_exclude_new_can_buy/龙洲股份_002682.txt')
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
        if name.startswith('gen_basic_daily_buy_signal_'):
            data = func(data)
    data = gen_multiple_daily_buy_signal_yes(data)
    # data = clear_other_clo(data)
    data = data.fillna(False)
    return data

def gen_full_zhishu_basic_signal(data, is_need_pre=False):
    """
    扫描basic_daily_zhishu_strategy.py文件，生成所有的基础信号,过滤出以gen_basic_daily_buy_signal开头的函数，并应用于data
    :param data:
    :return:
    """
    # 扫描basic_daily_strategy.py文件，生成所有的基础信号
    for name, func in inspect.getmembers(basic_zhishu_strategy, inspect.isfunction):
        if name.startswith('gen_basic_daily_zhishu_buy_signal_'):
            data = func(data)
    if is_need_pre:
        data = gen_multiple_daily_buy_signal_yes(data)
    # 将data在所有的nan值替换为False
    data = data.fillna(False)
    return data

def gen_full_all_dimension_signal(data):
    """
    生成所有维度（天，周，月，上证）的信号
    :param data:
    :return:
    """
    # 扫描basic_daily_strategy.py文件，生成所有的天基础信号
    for name, func in inspect.getmembers(basic_daily_strategy, inspect.isfunction):
        if name.startswith('gen_basic_daily_buy_signal_'):
            data = func(data)
    data = gen_multiple_daily_buy_signal_yes(data)
    return data

def gen_full_all_basic_signal_gen(data):
    """
    扫描basic_daily_strategy.py文件，生成所有的基础信号,过滤出以gen_basic_daily_buy_signal开头的函数，并应用于data
    :param data:
    :return:
    """
    # 扫描basic_daily_strategy.py文件，生成所有的基础信号
    for name, func in inspect.getmembers(basic_daily_strategy, inspect.isfunction):
        if name == 'gen_basic_daily_buy_signal_gen':
            data = func(data)
    data = gen_multiple_daily_buy_signal_yes(data)
    return data


def get_combination_list_optimized(basic_indicators, newest_indicators, exist_combinations_set, zero_combinations_set):
    result_combination_list = []
    full_combination_list = []
    seen_combinations = set()  # 用于跟踪已经添加的组合

    # Convert indicators to sets in advance
    basic_indicators_sets = [frozenset(basic_indicator) for basic_indicator in basic_indicators]
    newest_indicators_sets = [frozenset(newest_indicator) for newest_indicator in
                              newest_indicators] if newest_indicators != [[]] else [frozenset()]

    for basic_set in basic_indicators_sets:
        for newest_set in newest_indicators_sets:
            if not basic_set & newest_set:  # Check for intersection
                combination = basic_set | newest_set
                if combination not in seen_combinations:
                    seen_combinations.add(combination)
                    combination_list = list(combination)
                    if ':'.join(combination_list) not in exist_combinations_set:
                        result_combination_list.append(combination_list)
                    full_combination_list.append(combination_list)

    return result_combination_list, full_combination_list


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
                        if not any([set(combination_list) >= set(zero_combination.split(':')) for zero_combination in
                                    zero_combinations_set]):
                            result_combination_list.append(combination_list)
    # 对result_combination_list和full_combination_list进行去重
    result_combination_list = list(set([tuple(t) for t in result_combination_list]))
    full_combination_list = list(set([tuple(t) for t in full_combination_list]))
    return result_combination_list, full_combination_list


def filter_combination_list(combination_list, statistics, zero_combinations_set, trade_count_threshold=10000):
    """
    过滤掉交易次数小于10000的组合,不过滤已存在
    :param combination_list:
    :param statistics:
    :param trade_count_threshold:
    :return:
    """
    zero_combinations_set = {frozenset(zero_combination.split(':')) for zero_combination in zero_combinations_set}
    result_combination_list = []
    for combination in combination_list:
        combination_key = ':'.join(combination)
        combination_set = set(combination)
        if combination_key not in statistics or statistics[combination_key]['trade_count'] >= trade_count_threshold:
            if not any(combination_set >= zero_comb for zero_comb in zero_combinations_set):
                result_combination_list.append(combination)
    return result_combination_list


def back_layer_all_op_gen(file_path, result_combination_list, gen_signal_func=gen_full_all_basic_signal,
                          backtest_func=backtest_strategy_low_profit, target_key='target_key'):
    """
    分层进行回测的优化版函数
    """
    # 优化5: 使用并行处理
    if result_combination_list:
        process_combinations_gen(result_combination_list, file_path, gen_signal_func, backtest_func, target_key)


def back_layer_all_good(file_path, result_combination_list, gen_signal_func=gen_full_all_basic_signal,
                        backtest_func=backtest_strategy_low_profit, target_key='target_key'):
    """
    分层进行回测指定的指标，用于最终的策略
    """
    # 优化5: 使用并行处理
    if result_combination_list:
        process_combinations_good(result_combination_list, file_path, gen_signal_func, backtest_func, target_key)


def back_layer_all_op(file_path, gen_signal_func=gen_full_all_basic_signal, backtest_func=backtest_strategy_low_profit,
                      target_key='target_key'):
    """
    分层进行回测的优化版函数
    """
    statistics_file_path = '../back/' + f'statistics_{target_key.replace(":", "_")}.json'
    level = 0
    # 优化1: 一次性读取statistics
    statistics = read_json(statistics_file_path)

    # 优化2: 改进对statistics的处理
    zero_combinations_set = {key for key, value in statistics.items() if value['trade_count'] == 0}
    exist_combinations_set = set(statistics.keys())

    # 优化3: 减少文件操作，如果可能的话，改进load_data和gen_signal_func以减少重复计算
    data = load_data('../daily_data_exclude_new_can_buy/龙洲股份_002682.txt')
    data = gen_signal_func(data)
    signal_columns = [column for column in data.columns if 'signal' in column]
    basic_indicators = [[column] for column in signal_columns]

    # 优化4: 减少重复的filter_combination_list调用
    basic_indicators = filter_combination_list(basic_indicators, statistics, zero_combinations_set)
    newest_indicators = [[]]
    # temp_list = []
    # for column in signal_columns:
    #     if [column] not in basic_indicators:
    #         temp_list.append([column])

    while True:
        # 将newest_indicators按照100个一组进行分组
        if level == 2 and os.path.exists(f'../back/combination_list_{level}.json'):
            temp_dict = read_json(f'../back/combination_list_{level}.json')
            result_combination_list = temp_dict['result_combination_list']
            full_combination_list = temp_dict['full_combination_list']
            newest_indicators = temp_dict['newest_indicators']
        else:
            result_combination_list, full_combination_list = get_combination_list(basic_indicators, newest_indicators,
                                                                                  exist_combinations_set,
                                                                                  zero_combinations_set)
            newest_indicators = filter_combination_list(full_combination_list, statistics, zero_combinations_set)

            combination_list_path = f'../back/combination_list_{level}.json'
            temp_dict = {'result_combination_list': result_combination_list,
                         'full_combination_list': full_combination_list,
                         'newest_indicators': newest_indicators}
            write_json(combination_list_path, temp_dict)

        # ... 省略部分代码 ...
        print(
            f'level:{level}, zero_combinations_set:{len(zero_combinations_set)}, basic_indicators:{len(basic_indicators)}, newest_indicators:{len(newest_indicators)}, result_combination_list:{len(result_combination_list)}, full_combination_list:{len(full_combination_list)}')
        # 优化5: 使用并行处理
        if result_combination_list:
            process_combinations(result_combination_list, file_path, gen_signal_func, backtest_func, target_key)
            # 更新statistics和zero_combinations_set
            statistics = read_json(statistics_file_path)
            exist_combinations_set = set(statistics.keys())
            zero_combinations_set = {key for key, value in statistics.items() if value['trade_count'] == 0}
            newest_indicators = filter_combination_list(full_combination_list, statistics, zero_combinations_set)

        # 检查是否继续循环
        if not newest_indicators:
            break
        level += 1

def back_layer_all_op_gen_single(file_path, result_combination_list, gen_signal_func=gen_full_all_basic_signal,
                          backtest_func=backtest_strategy_low_profit, target_key='target_key'):
    """
    分层进行回测的优化版函数,单独存储一份数据
    """
    # 优化5: 使用并行处理
    if result_combination_list:
        process_combinations_gen_single(result_combination_list, file_path, gen_signal_func, backtest_func, target_key)

def back_layer_all_with_back_data(task_with_back_data, result_combination_list, gen_signal_func=gen_full_all_basic_signal,
                          backtest_func=backtest_strategy_low_profit, target_key='target_key'):
    """
    直接将有回测的数据进行指标效果统计
    """
    # 优化5: 使用并行处理
    if result_combination_list:
        process_combinations_with_back_data(result_combination_list, task_with_back_data, gen_signal_func, backtest_func, target_key)

def process_combinations_good(result_combination_list, file_path, gen_signal_func, backtest_func, target_key):
    """
    回测最终好指标的函数
    """
    # 按照一定数量分割list
    split_size = 1000000
    result_combination_lists = [result_combination_list[i:i + split_size] for i in
                                range(0, len(result_combination_list), split_size)]
    total_len = 0
    # 删除'../back/gen/sublist.json'
    if os.path.exists('../final_zuhe/sublist.json'):
        os.remove('../final_zuhe/sublist.json')
    sublist_json = []
    for sublist in result_combination_lists:
        # 开始计时
        start_time = time.time()
        # 读取sublist
        total_len += len(sublist)
        # 打印进度
        print(f"Processing {total_len} files... of {len(result_combination_list)}")
        tasks = prepare_tasks(sublist, file_path, gen_signal_func, backtest_func)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            total_files = len(tasks)
            for i, _ in enumerate(pool.imap_unordered(gen_all_signal_processing_good, tasks), 1):
                print(f"Processing file {i} of {total_files}...")
        # 结束计时
        end_time = time.time()
        # 将sublist增加到sublist_json,并写入文件
        sublist_json.extend(sublist)
        sublist_json = deduplicate_2d_list(sublist_json)
        write_json('../final_zuhe/sublist.json', sublist_json)

        statistics_zuhe_good('../final_zuhe/zuhe')
        print(f"Time taken: {end_time - start_time:.2f} seconds")

def process_combinations_with_back_data(result_combination_list, task_with_back_data, gen_signal_func, backtest_func, target_key):
    """
    直接将有回测的数据进行指标效果统计
    """
    # 按照一定数量分割list
    split_size = 30000
    result_combination_lists = [result_combination_list[i:i + split_size] for i in
                                range(0, len(result_combination_list), split_size)]
    total_len = 0
    sublist_json = []
    for sublist in result_combination_lists:
        # 开始计时
        start_time = time.time()
        # 读取sublist
        total_len += len(sublist)
        # 打印进度
        print(f"Processing {total_len} files... of {len(result_combination_list)}")
        tasks = prepare_task_with_back_data(sublist, task_with_back_data)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            total_files = len(tasks)
            for i, _ in enumerate(pool.imap_unordered(gen_all_signal_processing_with_back_data, tasks), 1):
                print(f"Processing file {i} of {total_files}...")
        # 结束计时
        end_time = time.time()
        # 将sublist增加到sublist_json,并写入文件
        sublist_json.extend(sublist)
        sublist_json = deduplicate_2d_list(sublist_json)
        write_json('../back/gen/sublist.json', sublist_json)

        # statistics_zuhe_gen_with_back_data('../back/gen/with_back_data_single', target_key='all')
        print(f"Time taken: {end_time - start_time:.2f} seconds")

def process_combinations_gen_single(result_combination_list, file_path, gen_signal_func, backtest_func, target_key):
    """
    使用多进程处理组合
    """
    # 按照一定数量分割list
    split_size = 200000
    result_combination_lists = [result_combination_list[i:i + split_size] for i in
                                range(0, len(result_combination_list), split_size)]
    total_len = 0
    sublist_json = []
    for sublist in result_combination_lists:
        # 开始计时
        start_time = time.time()
        # 读取sublist
        total_len += len(sublist)
        # 打印进度
        print(f"Processing {total_len} files... of {len(result_combination_list)}")
        tasks = prepare_tasks(sublist, file_path, gen_signal_func, backtest_func)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            total_files = len(tasks)
            for i, _ in enumerate(pool.imap_unordered(gen_all_signal_processing_gen_single_file, tasks), 1):
                print(f"Processing file {i} of {total_files}...")
        # 结束计时
        end_time = time.time()
        # 将sublist增加到sublist_json,并写入文件
        sublist_json.extend(sublist)
        sublist_json = deduplicate_2d_list(sublist_json)
        write_json('../back/gen/sublist.json', sublist_json)

        statistics_zuhe_gen_both_single_every_period('../back/gen/single')
        print(f"Time taken: {end_time - start_time:.2f} seconds")

def process_combinations_gen(result_combination_list, file_path, gen_signal_func, backtest_func, target_key):
    """
    使用多进程处理组合
    """
    # 按照一定数量分割list
    split_size = 10000
    result_combination_lists = [result_combination_list[i:i + split_size] for i in
                                range(0, len(result_combination_list), split_size)]
    total_len = 0
    # 删除'../back/gen/sublist.json'
    if os.path.exists('../back/gen/sublist.json'):
        os.remove('../back/gen/sublist.json')
    sublist_json = []
    for sublist in result_combination_lists:
        # 开始计时
        start_time = time.time()
        # 读取sublist
        total_len += len(sublist)
        # 打印进度
        print(f"Processing {total_len} files... of {len(result_combination_list)}")
        tasks = prepare_tasks(sublist, file_path, gen_signal_func, backtest_func)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            total_files = len(tasks)
            for i, _ in enumerate(pool.imap_unordered(gen_all_signal_processing_gen, tasks), 1):
                print(f"Processing file {i} of {total_files}...")
        # 结束计时
        end_time = time.time()
        # 将sublist增加到sublist_json,并写入文件
        sublist_json.extend(sublist)
        sublist_json = deduplicate_2d_list(sublist_json)
        write_json('../back/gen/sublist.json', sublist_json)

        statistics_zuhe_gen_both('../back/gen/zuhe', target_key='all')
        print(f"Time taken: {end_time - start_time:.2f} seconds")


def process_combinations(result_combination_list, file_path, gen_signal_func, backtest_func, target_key):
    """
    使用多进程处理组合
    """
    # 按照一定数量分割list
    split_size = 10000
    result_combination_lists = [result_combination_list[i:i + split_size] for i in
                                range(0, len(result_combination_list), split_size)]
    total_len = 0
    for sublist in result_combination_lists:
        # 开始计时
        start_time = time.time()
        # 读取sublist
        sublist_json = read_json('../back/sublist.json')
        total_len += len(sublist)
        # 如果sublist是空的，赋值为[]
        if not sublist_json:
            sublist_json = []
        # 打印进度
        print(f"Processing {total_len} files... of {len(result_combination_list)}")
        tasks = prepare_tasks(sublist, file_path, gen_signal_func, backtest_func)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            total_files = len(tasks)
            for i, _ in enumerate(pool.imap_unordered(gen_all_signal_processing_op, tasks), 1):
                print(f"Processing file {i} of {total_files}...")
        # 结束计时
        end_time = time.time()
        # 将sublist增加到sublist_json,并写入文件
        sublist_json.extend(sublist)
        sublist_json = deduplicate_2d_list(sublist_json)
        write_json('../back/sublist.json', sublist_json)

        statistics_zuhe('../back/zuhe', target_key=target_key)
        print(f"Time taken: {end_time - start_time:.2f} seconds")


def deduplicate_2d_list(lst):
    """
    This function removes duplicate sublists from a given 2D list. It uses a set to track unique elements,
    which requires the sublists to be converted to a tuple (since lists are not hashable).
    """
    unique_sublists = set()
    deduplicated_list = []

    for sublist in lst:
        # Convert the sublist to a tuple to make it hashable
        tuple_sublist = tuple(sublist)
        if tuple_sublist not in unique_sublists:
            unique_sublists.add(tuple_sublist)
            deduplicated_list.append(sublist)

    return deduplicated_list


def prepare_tasks(combination_list, file_path, gen_signal_func, backtest_func):
    """
    准备多进程任务
    """
    tasks = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            full_path = os.path.join(root, file)
            tasks.append((full_path, combination_list, gen_signal_func, backtest_func))
    tasks.reverse()  # 逆序排列任务
    return tasks

def prepare_task_with_back_data(combination_list, task_with_back_data):
    """
    准备多进程任务
    """
    tasks = []
    count = 0
    for task_data in task_with_back_data:
        count += 1
        file_name = str(count) + '.json'
        tasks.append((file_name, task_data, combination_list))
    return tasks

def back_layer_all(file_path, gen_signal_func=gen_full_all_basic_signal, backtest_func=backtest_strategy_low_profit,
                   target_key='all'):
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
    statistics_file_path = '../back/' + f'statistics_{target_key.replace(":", "_")}.json'
    level = 0
    # 读取statistics.json文件，得到已有的key
    statistics = read_json(statistics_file_path)
    # 找到trade_count为0的指标，将key保持在zero_combinations_set中
    zero_combinations_set = set()
    for key, value in statistics.items():
        if value['trade_count'] == 0:
            zero_combinations_set.add(key)
    exist_combinations_set = set(statistics.keys())
    data = load_data('../daily_data_exclude_new_can_buy/龙洲股份_002682.txt')
    data = gen_signal_func(data)
    signal_columns = [column for column in data.columns if 'signal' in column]
    # 将每个信号单独组成一个组合
    basic_indicators = [[column] for column in signal_columns]
    basic_indicators = filter_combination_list(basic_indicators, statistics, zero_combinations_set)
    newest_indicators = [[]]
    while True:
        # 如果对应的层级的组合文件已经存在，那么直接读取
        if os.path.exists(f'../back/combination_list_{level}.json'):
            temp_dict = read_json(f'../back/combination_list_{level}.json')
            result_combination_list = temp_dict['result_combination_list']
            full_combination_list = temp_dict['full_combination_list']
            newest_indicators = temp_dict['newest_indicators']
        else:
            result_combination_list, full_combination_list = get_combination_list(basic_indicators, newest_indicators,
                                                                                  exist_combinations_set,
                                                                                  zero_combinations_set)
            newest_indicators = filter_combination_list(full_combination_list, statistics, zero_combinations_set)
            temp_dict = {}
            temp_dict['result_combination_list'] = result_combination_list
            temp_dict['full_combination_list'] = full_combination_list
            temp_dict['newest_indicators'] = newest_indicators
            write_json(f'../back/combination_list_{level}.json', temp_dict)
        print(
            f'level:{level}, zero_combinations_set:{len(zero_combinations_set)}, basic_indicators:{len(basic_indicators)}, newest_indicators:{len(newest_indicators)}, result_combination_list:{len(result_combination_list)}, full_combination_list:{len(full_combination_list)}')
        # result_combination_list = ['BAR_20日_小极值_signal:收盘_大于_20_固定区间_signal'.split(':')]
        if result_combination_list == []:
            if newest_indicators == []:
                break
            continue
        total_count = len(result_combination_list)
        current_count = 0
        # 将result_combination_list按照100000个一组进行分组
        result_combination_lists = [result_combination_list[i:i + 50000] for i in
                                    range(0, len(result_combination_list), 50000)]
        for result_combination_list in result_combination_lists:
            current_count += len(result_combination_list)
            # 输出当前的进度
            print(
                f'level:{level}, current_count:{current_count}, total_count:{total_count}, progress:{current_count / total_count * 100:.2f}%')
            tasks = []
            # gen_all_signal_processing(('../daily_data_exclude_new_can_buy/ST国嘉_600646.txt', final_combinations, gen_signal_func, backtest_func))
            for root, ds, fs in os.walk(file_path):
                for f in fs:
                    fullname = os.path.join(root, f)
                    tasks.append(
                        (fullname, result_combination_list, gen_signal_func, backtest_func, zero_combinations_set))
            # 将tasks逆序排列，这样可以让最后一个文件先处理，这样可以先看到结果
            tasks.reverse()
            # 使用进程池处理任务
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)

            total_files = len(tasks)
            for i, _ in enumerate(pool.imap_unordered(gen_all_signal_processing, tasks), 1):
                print(f"Processing file {i} of {total_files}...")

            pool.close()
            pool.join()
            statistics_zuhe('../back/zuhe')
            statistics = read_json(statistics_file_path)
            exist_combinations_set = set(statistics.keys())
            zero_combinations_set = set()
            for key, value in statistics.items():
                if value['trade_count'] == 0:
                    zero_combinations_set.add(key)
            newest_indicators = filter_combination_list(full_combination_list, statistics, zero_combinations_set)
            level += 1


def back_sigle_all(file_path, gen_signal_func=gen_full_all_basic_signal, backtest_func=backtest_strategy_low_profit):
    """
    回测每个基础指标，方便后续过滤某些指标
    :param data:
    :return:
    """
    data = load_data('../daily_data_exclude_new_can_buy/龙洲股份_002682.txt')
    data = gen_signal_func(data)
    signal_columns = [column for column in data.columns if 'signal' in column]
    # 将每个信号单独组成一个组合
    final_combinations = [[column] for column in signal_columns]
    # 准备多进程处理的任务列表
    tasks = []
    # gen_all_signal_processing(('../daily_data_exclude_new_can_buy/ST国嘉_600646.txt', final_combinations, gen_signal_func, backtest_func))
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


@timeit
def statistics_zuhe_good(file_path):
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
                    for key1, value1 in value.items():
                        if key1 in result[key]:
                            for key2, value2 in value1.items():
                                if key2 == 'all_date':
                                    result[key][key1][key2].extend(value2)
                                    # 去重
                                    result[key][key1][key2] = list(set(result[key][key1][key2]))
                                    continue

                                if key2 in result[key][key1]:
                                    result[key][key1][key2] += value2
                                else:
                                    result[key][key1][key2] = value2
                        else:
                            result[key][key1] = value1
                else:
                    result[key] = value

    for key1, value1 in result.items():
        for key, value in value1.items():
            if value['trade_count'] != 0:
                # value['all_date'] = list(set(value['all_date']))
                value['average_1w_profit'] = round(10000 * value['total_profit'] / value['total_cost'], 4)
                value['ratio'] = value['size_of_result_df'] / value['trade_count']
                value['average_days_held'] = value['total_days_held'] / value['trade_count']
                value['average_profit'] = value['total_profit'] / value['trade_count']
                # value['date_count'] = len(value['all_date'])
                # value['date_ratio'] = round(value['date_count'] / value['trade_count'], 4)
                # value['start_date'] = min(value['all_date'])
                # value['end_date'] = max(value['all_date'])
                # 将value['ratio']保留4位小数
                value['ratio'] = round(value['ratio'], 4)
                value['average_profit'] = round(value['average_profit'], 4)
                value['average_days_held'] = round(value['average_days_held'], 4)
                value['total_profit'] = round(value['total_profit'], 4)
            else:
                value['ratio'] = 1
                value['average_profit'] = 0
                value['average_1w_profit'] = 0
                value['average_days_held'] = 0
                value['total_profit'] = 0
                value['date_count'] = 0
                value['date_ratio'] = 0
                # 找到value['all_date']中的起始和结束时间
                value['start_date'] = "1970"
                value['end_date'] = "2100"
            value['all_date'] = []
    # 将resul trade_count降序排序，然后在此基础上再按照ratio升序排序
    result = dict(sorted(result.items(), key=lambda x: (-x[1]['all']['ratio'], x[1]['all']['trade_count']), reverse=True))
    target_key = 'target_key'
    # 将result写入file_path上一级文件
    file_name = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}.json'
    # 写入一份备份文件
    file_name_backup = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}_backup.json'
    if os.path.exists(file_name_backup):
        os.remove(file_name_backup)
    write_json(file_name_backup, result)
    # 先删除原来的statistics.json文件
    result = dict(sorted(result.items(), key=lambda x: (-x[1]['all']['ratio'], x[1]['all']['trade_count']), reverse=True))
    if os.path.exists(file_name):
        os.remove(file_name)
    write_json(file_name, result)

    # 读取'../back/gen/statistics_all.json'文件，然后将result中的数据合并到statistics_all.json中
    statistics_all = read_json('../back/gen/statistics_all.json')
    statistics_all.update(result)
    statistics_all_backup = '../back/gen/statistics_all_backup.json'
    if os.path.exists(statistics_all_backup):
        old_data = read_json(statistics_all_backup)
        if len(old_data) < len(statistics_all):
            os.remove(statistics_all_backup)
            write_json(statistics_all_backup, statistics_all)

    # 将statistics_all.json写入文件
    write_json('../back/gen/statistics_all.json', statistics_all)
    return result

@timeit
def statistics_zuhe_gen_both(file_path, target_key='all'):
    """
    读取file_path下的所有.json文件，将有相同key的数据进行合并
    :param file_path:
    :return:
    """
    result = {}
    sublist_list = read_json('../back/gen/sublist.json')
    sublist_set = set()
    for sublist in sublist_list:
        temp_key = ':'.join(sublist)
        sublist_set.add(temp_key)
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            fullname = os.path.join(root, f)
            data = read_json(fullname)
            for key, value in data.items():
                if 'three_befor_year_count_thread' not in value:
                    value['one_befor_year_count'] = 0
                    value['two_befor_year_count'] = 0
                    value['three_befor_year_count'] = 0
                    value['three_befor_year_count_thread'] = 0
                if key in result:
                    result[key]['trade_count'] += value['trade_count']
                    result[key]['total_profit'] += value['total_profit']
                    result[key]['total_cost'] += value['total_cost']
                    result[key]['size_of_result_df'] += value['size_of_result_df']
                    result[key]['total_days_held'] += value['total_days_held']
                    result[key]['one_befor_year_count'] += value['one_befor_year_count']
                    result[key]['two_befor_year_count'] += value['two_befor_year_count']
                    result[key]['three_befor_year_count'] += value['three_befor_year_count']
                    result[key]['three_befor_year_count_thread'] += value['three_befor_year_count_thread']
                else:
                    result[key] = value
    # 再计算result每一个key的平均值
    for key, value in result.items():
        if value['trade_count'] != 0:
            if 'total_cost' not in value:
                value['average_1w_profit'] = 0
                value['three_befor_year_count'] = 0
                value['two_befor_year_count'] = 0
                value['one_befor_year_count'] = 0
                value['three_befor_year_count_thread'] = 0
            else:
                value['average_1w_profit'] = round(10000 * value['total_profit'] / value['total_cost'], 4)
            value['ratio'] = value['size_of_result_df'] / value['trade_count']
            value['average_days_held'] = value['total_days_held'] / value['trade_count']
            value['average_profit'] = value['total_profit'] / value['trade_count']

            # 将value['ratio']保留4位小数
            value['ratio'] = round(value['ratio'], 4)
            value['average_profit'] = round(value['average_profit'], 4)
            value['average_days_held'] = round(value['average_days_held'], 4)
            value['total_profit'] = round(value['total_profit'], 4)
            value['one_befor_year_rate'] = round(value['one_befor_year_count'] / value['trade_count'], 4)
            value['two_befor_year_rate'] = round(value['two_befor_year_count'] / value['trade_count'], 4)
            value['three_befor_year_rate'] = round(value['three_befor_year_count'] / value['trade_count'], 4)
            if value['three_befor_year_count'] != 0:
                value['three_befor_year_count_thread_ratio'] = round(value['three_befor_year_count_thread'] / value['three_befor_year_count'], 4)
            else:
                value['three_befor_year_count_thread_ratio'] = 1
        else:
            value['ratio'] = 1
            value['average_profit'] = 0
            value['average_1w_profit'] = 0
            value['average_days_held'] = 0
            value['total_profit'] = 0
    # 将resul trade_count降序排序，然后在此基础上再按照ratio升序排序
    result = dict(sorted(result.items(), key=lambda x: (-x[1]['ratio'], x[1]['trade_count']), reverse=True))
    # 将result写入file_path上一级文件
    target_key = 'all'
    # 先写入一份到备份文件
    file_name_backup = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}_backup.json'
    if os.path.exists(file_name_backup):
        os.remove(file_name_backup)
    write_json(file_name_backup, result)
    file_name = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}.json'

    # 先删除原来的statistics.json文件
    if os.path.exists(file_name):
        os.remove(file_name)
    write_json(file_name, result)
    return result

@timeit
def statistics_zuhe_gen_with_back_data(file_path, target_key='all'):
    """
    读取file_path下的所有.json文件，将有相同key的数据进行合并
    统计的是单次新增，然后将结果加入之前的all.json文件中
    :param file_path:
    :return:
    """
    result = {}
    sublist_list = read_json('../back/gen/sublist.json')
    sublist_set = set()
    for sublist in sublist_list:
        temp_key = ':'.join(sublist)
        sublist_set.add(temp_key)
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            fullname = os.path.join(root, f)
            data = read_json(fullname)
            for key, value in data.items():
                if 'three_befor_year_count_thread' not in value:
                    value['one_befor_year_count'] = 0
                    value['two_befor_year_count'] = 0
                    value['three_befor_year_count'] = 0
                    value['three_befor_year_count_thread'] = 0
                if key in result:
                    result[key]['trade_count'] += value['trade_count']
                    result[key]['total_profit'] += value['total_profit']
                    result[key]['total_cost'] += value['total_cost']
                    result[key]['size_of_result_df'] += value['size_of_result_df']
                    result[key]['total_days_held'] += value['total_days_held']
                    result[key]['one_befor_year_count'] += value['one_befor_year_count']
                    result[key]['two_befor_year_count'] += value['two_befor_year_count']
                    result[key]['three_befor_year_count'] += value['three_befor_year_count']
                    result[key]['three_befor_year_count_thread'] += value['three_befor_year_count_thread']
                else:
                    result[key] = value
    # 再计算result每一个key的平均值
    for key, value in result.items():
        if value['trade_count'] != 0:
            if 'total_cost' not in value:
                value['average_1w_profit'] = 0
                value['three_befor_year_count'] = 0
                value['two_befor_year_count'] = 0
                value['one_befor_year_count'] = 0
                value['three_befor_year_count_thread'] = 0
            else:
                value['average_1w_profit'] = round(10000 * value['total_profit'] / value['total_cost'], 4)
            value['ratio'] = value['size_of_result_df'] / value['trade_count']
            value['average_days_held'] = value['total_days_held'] / value['trade_count']
            value['average_profit'] = value['total_profit'] / value['trade_count']

            # 将value['ratio']保留4位小数
            value['ratio'] = round(value['ratio'], 4)
            value['average_profit'] = round(value['average_profit'], 4)
            value['average_days_held'] = round(value['average_days_held'], 4)
            value['total_profit'] = round(value['total_profit'], 4)
            value['one_befor_year_rate'] = round(value['one_befor_year_count'] / value['trade_count'], 4)
            value['two_befor_year_rate'] = round(value['two_befor_year_count'] / value['trade_count'], 4)
            value['three_befor_year_rate'] = round(value['three_befor_year_count'] / value['trade_count'], 4)
            if value['three_befor_year_count'] != 0:
                value['three_befor_year_count_thread_ratio'] = round(value['three_befor_year_count_thread'] / value['three_befor_year_count'], 4)
            else:
                value['three_befor_year_count_thread_ratio'] = 1
        else:
            value['ratio'] = 1
            value['average_profit'] = 0
            value['average_1w_profit'] = 0
            value['average_days_held'] = 0
            value['total_profit'] = 0
    # 将resul trade_count降序排序，然后在此基础上再按照ratio升序排序
    result = dict(sorted(result.items(), key=lambda x: (x[1]['trade_count'], -x[1]['ratio']), reverse=True))
    # 写入target_key.json
    target_key_result = result
    target_key = 'target_key'
    file_name = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}.json'

    # 先删除原来的statistics.json文件
    if os.path.exists(file_name):
        os.remove(file_name)
    write_json(file_name, target_key_result)


    # 将result写入file_path上一级文件
    target_key = 'all'
    file_name = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}.json'
    # 读取file_name
    if os.path.exists(file_name):
        old_result = read_json(file_name)
        old_result.update(target_key_result)
        result = old_result
    result = dict(sorted(result.items(), key=lambda x: (-x[1]['ratio'], x[1]['trade_count']), reverse=True))
    # 再写入一份到备份文件
    file_name_back = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}_backup.json'
    write_json(file_name_back, result)
    # 先删除原来的statistics.json文件
    if os.path.exists(file_name):
        os.remove(file_name)
    write_json(file_name, result)


    return result

def statistics_zuhe_gen_both_single_every_period(file_path):
    """
    读取file_path下的所有.json文件，将有相同key的数据进行合并
    统计的是单次新增，然后将结果加入之前的all.json文件中
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
                    for key1, value1 in value.items():
                        if key1 in result[key]:
                            for key2, value2 in value1.items():
                                # if key2 == 'all_date':
                                #     result[key][key1][key2].extend(value2)
                                #     # 去重
                                #     result[key][key1][key2] = list(set(result[key][key1][key2]))
                                #     continue

                                if key2 in result[key][key1]:
                                    result[key][key1][key2] += value2
                                else:
                                    result[key][key1][key2] = value2
                        else:
                            result[key][key1] = value1
                else:
                    result[key] = value

    for key1, value1 in result.items():
        for key, value in value1.items():
            if value['trade_count'] != 0:
                # value['all_date'] = list(set(value['all_date']))
                value['average_1w_profit'] = round(10000 * value['total_profit'] / value['total_cost'], 4)
                value['ratio'] = value['size_of_result_df'] / value['trade_count']
                value['average_days_held'] = value['total_days_held'] / value['trade_count']
                value['average_profit'] = value['total_profit'] / value['trade_count']
                # value['date_count'] = len(value['all_date'])
                # value['date_ratio'] = round(value['date_count'] / value['trade_count'], 4)
                # value['start_date'] = min(value['all_date'])
                # value['end_date'] = max(value['all_date'])
                # 将value['ratio']保留4位小数
                value['ratio'] = round(value['ratio'], 4)
                value['average_profit'] = round(value['average_profit'], 4)
                value['average_days_held'] = round(value['average_days_held'], 4)
                value['total_profit'] = round(value['total_profit'], 4)
            else:
                value['ratio'] = 1
                value['average_profit'] = 0
                value['average_1w_profit'] = 0
                value['average_days_held'] = 0
                value['total_profit'] = 0
                value['date_count'] = 0
                value['date_ratio'] = 0
                # 找到value['all_date']中的起始和结束时间
                value['start_date'] = "1970"
                value['end_date'] = "2100"
            value['all_date'] = []
    # 将resul trade_count降序排序，然后在此基础上再按照ratio升序排序
    result = dict(sorted(result.items(), key=lambda x: (-x[1]['all']['ratio'], x[1]['all']['trade_count']), reverse=True))
    # 写入target_key.json
    target_key_result = result
    target_key = 'target_key'
    file_name = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}.json'

    # 先删除原来的statistics.json文件
    if os.path.exists(file_name):
        os.remove(file_name)
    write_json(file_name, target_key_result)


    # 将result写入file_path上一级文件
    target_key = 'all'
    file_name = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}.json'
    # 读取file_name
    if os.path.exists(file_name):
        old_result = read_json(file_name)
        old_result.update(target_key_result)
        result = old_result
    if os.path.exists(file_name):
        os.remove(file_name)
    write_json(file_name, result)


    return result


@timeit
def statistics_zuhe_gen_both_single(file_path, target_key='all'):
    """
    读取file_path下的所有.json文件，将有相同key的数据进行合并
    统计的是单次新增，然后将结果加入之前的all.json文件中
    :param file_path:
    :return:
    """
    result = {}
    sublist_list = read_json('../back/gen/sublist.json')
    sublist_set = set()
    for sublist in sublist_list:
        temp_key = ':'.join(sublist)
        sublist_set.add(temp_key)
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            fullname = os.path.join(root, f)
            data = read_json(fullname)
            for key, value in data.items():
                if 'three_befor_year_count_thread' not in value:
                    value['one_befor_year_count'] = 0
                    value['two_befor_year_count'] = 0
                    value['three_befor_year_count'] = 0
                    value['three_befor_year_count_thread'] = 0
                if key in result:
                    result[key]['trade_count'] += value['trade_count']
                    result[key]['total_profit'] += value['total_profit']
                    result[key]['total_cost'] += value['total_cost']
                    result[key]['size_of_result_df'] += value['size_of_result_df']
                    result[key]['total_days_held'] += value['total_days_held']
                    result[key]['one_befor_year_count'] += value['one_befor_year_count']
                    result[key]['two_befor_year_count'] += value['two_befor_year_count']
                    result[key]['three_befor_year_count'] += value['three_befor_year_count']
                    result[key]['three_befor_year_count_thread'] += value['three_befor_year_count_thread']
                else:
                    result[key] = value
    # 再计算result每一个key的平均值
    for key, value in result.items():
        if value['trade_count'] != 0:
            if 'total_cost' not in value:
                value['average_1w_profit'] = 0
                value['three_befor_year_count'] = 0
                value['two_befor_year_count'] = 0
                value['one_befor_year_count'] = 0
                value['three_befor_year_count_thread'] = 0
            else:
                value['average_1w_profit'] = round(10000 * value['total_profit'] / value['total_cost'], 4)
            value['ratio'] = value['size_of_result_df'] / value['trade_count']
            value['average_days_held'] = value['total_days_held'] / value['trade_count']
            value['average_profit'] = value['total_profit'] / value['trade_count']

            # 将value['ratio']保留4位小数
            value['ratio'] = round(value['ratio'], 4)
            value['average_profit'] = round(value['average_profit'], 4)
            value['average_days_held'] = round(value['average_days_held'], 4)
            value['total_profit'] = round(value['total_profit'], 4)
            value['one_befor_year_rate'] = round(value['one_befor_year_count'] / value['trade_count'], 4)
            value['two_befor_year_rate'] = round(value['two_befor_year_count'] / value['trade_count'], 4)
            value['three_befor_year_rate'] = round(value['three_befor_year_count'] / value['trade_count'], 4)
            if value['three_befor_year_count'] != 0:
                value['three_befor_year_count_thread_ratio'] = round(value['three_befor_year_count_thread'] / value['three_befor_year_count'], 4)
            else:
                value['three_befor_year_count_thread_ratio'] = 1
        else:
            value['ratio'] = 1
            value['average_profit'] = 0
            value['average_1w_profit'] = 0
            value['average_days_held'] = 0
            value['total_profit'] = 0
    # 将resul trade_count降序排序，然后在此基础上再按照ratio升序排序
    result = dict(sorted(result.items(), key=lambda x: (-x[1]['ratio'], x[1]['trade_count']), reverse=True))
    # 写入target_key.json
    target_key_result = result
    target_key = 'target_key'
    file_name = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}.json'

    # 先删除原来的statistics.json文件
    if os.path.exists(file_name):
        os.remove(file_name)
    write_json(file_name, target_key_result)


    # 将result写入file_path上一级文件
    target_key = 'all'
    file_name = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}.json'
    # 读取file_name
    if os.path.exists(file_name):
        old_result = read_json(file_name)
        old_result.update(target_key_result)
        result = old_result
    result = dict(sorted(result.items(), key=lambda x: (-x[1]['ratio'], x[1]['trade_count']), reverse=True))
    # 再写入一份到备份文件
    # file_name_back = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}_backup.json'
    # write_json(file_name_back, result)
    # 先删除原来的statistics.json文件
    if os.path.exists(file_name):
        os.remove(file_name)
    write_json(file_name, result)


    return result


@timeit
def statistics_zuhe_gen(file_path, target_key='all'):
    """
    读取file_path下的所有.json文件，将有相同key的数据进行合并
    :param file_path:
    :return:
    """
    result = {}
    if target_key == 'all':
        for root, ds, fs in os.walk(file_path):
            for f in fs:
                fullname = os.path.join(root, f)
                data = read_json(fullname)
                for key, value in data.items():
                    if key in result:
                        result[key]['trade_count'] += value['trade_count']
                        result[key]['total_profit'] += value['total_profit']
                        result[key]['total_cost'] += value['total_cost']
                        result[key]['size_of_result_df'] += value['size_of_result_df']
                        result[key]['total_days_held'] += value['total_days_held']
                    else:
                        result[key] = value
    else:
        sublist_list = read_json('../back/gen/sublist.json')
        sublist_set = set()
        for sublist in sublist_list:
            temp_key = ':'.join(sublist)
            sublist_set.add(temp_key)

        for root, ds, fs in os.walk(file_path):
            for f in fs:
                fullname = os.path.join(root, f)
                data = read_json(fullname)
                for key, value in data.items():
                    if key in sublist_set:
                        if key in result:
                            result[key]['trade_count'] += value['trade_count']
                            result[key]['total_profit'] += value['total_profit']
                            result[key]['total_cost'] += value['total_cost']
                            result[key]['size_of_result_df'] += value['size_of_result_df']
                            result[key]['total_days_held'] += value['total_days_held']
                        else:
                            result[key] = value
    # 再计算result每一个key的平均值
    for key, value in result.items():
        if value['trade_count'] != 0:
            if 'total_cost' not in value:
                value['average_1w_profit'] = 0
            else:
                value['average_1w_profit'] = round(10000 * value['total_profit'] / value['total_cost'], 4)
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
            value['average_1w_profit'] = 0
            value['average_days_held'] = 0
            value['total_profit'] = 0
    # 将resul trade_count降序排序，然后在此基础上再按照ratio升序排序
    result = dict(sorted(result.items(), key=lambda x: (-x[1]['ratio'], x[1]['trade_count']), reverse=True))
    # 将result写入file_path上一级文件
    file_name = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}.json'

    # 先删除原来的statistics.json文件
    if os.path.exists(file_name):
        os.remove(file_name)
    write_json(file_name, result)
    return result


def statistics_zuhe(file_path, target_key='all'):
    """
    读取file_path下的所有.json文件，将有相同key的数据进行合并
    :param file_path:
    :return:
    """
    result = {}
    if target_key == 'all':
        for root, ds, fs in os.walk(file_path):
            for f in fs:
                fullname = os.path.join(root, f)
                data = read_json(fullname)
                for key, value in data.items():
                    if key in result:
                        result[key]['trade_count'] += value['trade_count']
                        result[key]['total_profit'] += value['total_profit']
                        # result[key]['total_cost'] += value['total_cost']
                        result[key]['size_of_result_df'] += value['size_of_result_df']
                        result[key]['total_days_held'] += value['total_days_held']
                    else:
                        result[key] = value
    else:
        sublist_list = read_json('../back/sublist.json')
        sublist_set = set()
        for sublist in sublist_list:
            temp_key = ':'.join(sublist)
            sublist_set.add(temp_key)

        for root, ds, fs in os.walk(file_path):
            for f in fs:
                fullname = os.path.join(root, f)
                data = read_json(fullname)
                for key, value in data.items():
                    if key in sublist_set:
                        if key in result:
                            result[key]['trade_count'] += value['trade_count']
                            result[key]['total_profit'] += value['total_profit']
                            # result[key]['total_cost'] += value['total_cost']
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
            # value['average_1w_profit'] = round(value['total_profit'] / value['total_cost'], 4)
            # 将value['ratio']保留4位小数
            value['ratio'] = round(value['ratio'], 4)
            value['average_profit'] = round(value['average_profit'], 4)
            value['average_days_held'] = round(value['average_days_held'], 4)
            value['total_profit'] = round(value['total_profit'], 4)
        else:
            value['ratio'] = 1
            value['average_profit'] = 0
            value['average_1w_profit'] = 0
            value['average_days_held'] = 0
            value['total_profit'] = 0
    # 将resul trade_count降序排序，然后在此基础上再按照ratio升序排序
    result = dict(sorted(result.items(), key=lambda x: (-x[1]['ratio'], x[1]['trade_count']), reverse=True))
    # 将result写入file_path上一级文件
    file_name = Path(file_path).parent / f'statistics_{target_key.replace(":", "_")}.json'

    # 先删除原来的statistics.json文件
    if os.path.exists(file_name):
        os.remove(file_name)
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
