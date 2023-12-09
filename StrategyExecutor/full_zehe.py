# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023-11-24 15:30
:last_date:
    2023-11-24 15:30
:description:
    
"""
import multiprocessing
import os
import random

import pandas as pd
from multiprocessing import Pool
import collections
import inspect
import itertools
import json
import math
import multiprocessing
import os
import shutil
import time
import timeit
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from StrategyExecutor.daily_strategy import mix, gen_daily_buy_signal_26, gen_daily_buy_signal_27, \
    gen_daily_buy_signal_25, gen_daily_buy_signal_28, gen_true, gen_daily_buy_signal_29, gen_daily_buy_signal_30, \
    select_stocks, mix_back
from StrategyExecutor.strategy import back_all_stock, strategy
from StrategyExecutor.zuhe_daily_strategy import gen_full_all_basic_signal, filter_combinations, filter_combinations_op, \
    create_empty_result, process_results_with_year

pd.options.mode.chained_assignment = None  # 关闭SettingWithCopyWarning

def get_sell_price(buy_price):
    """
    根据买入价格，返回卖出价格
    卖出价格为买入价格的1.0025倍，向上保留两位小数
    :param buy_price:
    :return:
    """
    sell_price = buy_price * 1.0025
    sell_price = math.ceil(sell_price * 100) / 100
    return sell_price


def backtest_strategy_low_profit(data):
    """
    每个买入信号都入手，然后找到相应的交易日
    为data进行回测，买入条件为当有Buy_Signal时，卖出条件为大于买入价的0.02或者以开盘价卖出
    ，返回买入卖出的时间，价格，收益，持有时间
    :param data:
    :return:
    """

    results = []  # 存储回测结果
    name = data['名称'].iloc[0]
    symbol = data['代码'].iloc[0]
    total_profit = 0
    i = 0
    while i < len(data):
        if data['Buy_Signal'].iloc[i] == 1:
            buy_price = data['收盘'].iloc[i]
            buy_date = data['日期'].iloc[i]
            buy_index = i
            total_shares = 100  # 假设初始买入100股

            # 找到满足卖出条件的日期
            j = i + 1
            while j < len(data):
                if data['最高'].iloc[j] >= get_sell_price(buy_price):
                    break  # 找到了满足卖出条件的日期，跳出循环

                # 如果第二天未达到卖出价条件，再买入100股并重新计算买入成本
                additional_shares = 100
                total_shares += additional_shares
                new_buy_price = data['收盘'].iloc[j]  # 第二天的收盘价作为新的买入价
                buy_price = (buy_price * (
                        total_shares - additional_shares) + new_buy_price * additional_shares) / total_shares
                data.at[i, '数量'] = additional_shares  # 记录买入数量

                j += 1

            # 如果找到了满足卖出条件的日期
            if j < len(data):
                sell_price = get_sell_price(buy_price)
                if data['开盘'].iloc[j] > sell_price:
                    sell_price = data['开盘'].iloc[j]
            else:
                # 如果没有找到，强制在最后一天卖出
                j = len(data) - 1
                sell_price = data['收盘'].iloc[j]

            sell_date = data['日期'].iloc[j]
            profit = (sell_price - buy_price) * 100  # 每次卖出100股
            total_profit += profit
            total_cost = buy_price * 100
            days_held = j - buy_index
            results.append([name, symbol, buy_date, buy_price, sell_date, sell_price, profit, total_profit, total_cost,
                            days_held, i])

        i += 1

    results_df = pd.DataFrame(results,
                              columns=['名称', '代码', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit',
                                       'Total_Profit', 'total_cost',
                                       'Days Held', 'Buy_Index'])

    return results_df


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        traceback.print_exc()
        return {}


def parse_filename(file_path):
    # 提取文件名（不包括扩展名）
    base_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]

    # 以'_'分割文件名以获取名称和编号
    stock_name, stock_code = file_name_without_ext.split('_')

    return stock_name, stock_code

def load_data(file_path):
    data = pd.read_csv(file_path)
    name, code = parse_filename(file_path)
    if '时间' in data.columns:
        data = data.rename(columns={'时间': '日期'})
    data['日期'] = pd.to_datetime(data['日期'])
    data['名称'] = name
    data['代码'] = code
    data['数量'] = 0
    data.sort_values(by='日期', ascending=True, inplace=True)

    # 使用 pandas 查找并移除第一个日期，如果它与其它日期不连续
    date_diff = data['日期'].diff(-1).abs()
    filtered_diff = date_diff[date_diff > pd.Timedelta(days=30)]

    # 如果有大于30天的断层
    if not filtered_diff.empty:
        cutoff_index = filtered_diff.idxmax()
        if cutoff_index and cutoff_index != 0:
            data = data.loc[cutoff_index + 1:]  # 跳过第一个数据点
    # 如果没有大于30天的断层，保留所有数据

    data.reset_index(drop=True, inplace=True)

    data['Max_rate'] = data['名称'].str.contains('st', case=False).map({True: 5, False: 10})
    data['Buy_Signal'] = (data['涨跌幅'] < 0.95 * data['Max_rate'])
    return data


def optimized_filter_combination_list(combination_list, statistics, zero_combinations_set, trade_count_threshold=1000):
    """
    Optimized version of the filter_combination_list function.
    Filters out combinations with a trade count less than the specified threshold and not in the zero combinations set.
    :param combination_list: List of combinations to be filtered.
    :param statistics: Dictionary with trade statistics.
    :param zero_combinations_set: Set of combinations to be excluded.
    :param trade_count_threshold: Minimum trade count threshold.
    :return: Filtered list of combinations.
    """
    # Convert zero_combinations_set to a set of frozensets for faster checking
    zero_combinations_frozensets = {frozenset(zero_combination.split(':')) for zero_combination in
                                    zero_combinations_set}

    result_combination_list = []
    for combination in combination_list:
        combination_set = frozenset(combination)
        combination_key = ':'.join(combination)

        # Check if the combination is in the zero combinations set
        if any(combination_set >= zero_comb for zero_comb in zero_combinations_frozensets):
            continue

        # Check if the combination is in statistics and meets the trade count threshold
        if combination_key in statistics and statistics[combination_key]['trade_count'] >= trade_count_threshold:
            result_combination_list.append(combination)

    return result_combination_list


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


def gen_signal_optimized(data, combination):
    """
    优化版本的生成信号函数
    :param data: 输入的 DataFrame
    :param combination: 列名组合的列表
    :return: 带有 Buy_Signal 列的 DataFrame
    """
    # 预先计算涨跌幅条件
    buy_condition = data['涨跌幅'] < 0.95 * data['Max_rate']

    # 使用 reduce 函数结合所有的条件
    combined_condition = np.logical_and.reduce([data[col] for col in combination])

    # 应用所有条件生成 Buy_Signal
    data['Buy_Signal'] = buy_condition & combined_condition

    return data


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


def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


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
            # exist_combinations_set = set(result_df_dict.keys())
            # # 将final_combinations转换为集合,需要将元素按照':'拼接，然后过滤掉exist_combinations_set中已经存在的组合
            # final_combinations_set = {':'.join(combination) for combination in final_combinations}
            # final_combinations_set = final_combinations_set - exist_combinations_set
            # final_combinations = [combination.split(':') for combination in final_combinations_set]
            for key, value in result_df_dict.items():
                if value['trade_count'] == 0:
                    zero_combination.add(key)
            print(f"{file_name} zero_combination len: {len(zero_combination)}")
            zero_combinations_set = {frozenset(zero_combination.split(':')) for zero_combination in
                                     zero_combination}
            # 这里也需要加载对final_combinations的过滤
            for index, combination in enumerate(final_combinations, start=1):
                combination_key = ':'.join(combination)
                combination_set = frozenset(combination)
                if not is_skip:
                    if combination_key in result_df_dict:
                        print(f"combination {combination_key} in result_df_dict")
                        continue
                if not is_skip:
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
            print(full_name)
        finally:
            # 计时结束
            end_time = time.time()
            print(f"{full_name} 耗时：{end_time - start_time}秒")
            write_json(file_name, result_df_dict)
    except Exception as e:
        # 输出异常栈
        traceback.print_exc()


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


def filter_combination_list_optimized(combination_list, statistics, zero_combinations_set, trade_count_threshold=1000):
    zero_combinations_set = {frozenset(zero_combination.split(':')) for zero_combination in zero_combinations_set}
    result_combination_list = []
    for combination in combination_list:
        combination_key = ':'.join(combination)
        combination_set = set(combination)
        if combination_key not in statistics or statistics[combination_key]['trade_count'] >= trade_count_threshold:
            if not any(combination_set >= zero_comb for zero_comb in zero_combinations_set):
                result_combination_list.append(combination)
    return result_combination_list


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
        level += 1
        result_combination_list = ['振幅_大于_10_日均线_signal:最低_20日_大极值signal:最高_小于_20_日均线_signal'.split(':')]
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
            # gen_all_signal_processing(('../daily_data_exclude_new/ST国嘉_600646.txt', final_combinations, gen_signal_func, backtest_func))
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
            statistics = read_json('../back/statistics.json')
            exist_combinations_set = set(statistics.keys())
            zero_combinations_set = set()
            for key, value in statistics.items():
                if value['trade_count'] == 0:
                    zero_combinations_set.add(key)
            newest_indicators = filter_combination_list(full_combination_list, statistics, zero_combinations_set)


def find_st_periods(posts):
    periods = []
    start = None
    end = None

    # 去除posts中notice_type包含 '深交所' 或 '上交所' 的帖子,或者是None
    posts = [post for post in posts if
             post['notice_type'] is not None and '深交所' not in post['notice_type'] and '上交所' not in post['notice_type']]

    for i in range(len(posts) - 1):
        current_post = posts[i]
        next_post = posts[i + 1]

        # 检查当前和下一个帖子是否都包含'ST'
        if ('ST' in current_post['post_title'] or '终止上市风险提示' in current_post['notice_type']) and (
                'ST' in next_post['post_title'] or '终止上市风险提示' in next_post['notice_type']):
            current_time = datetime.strptime(current_post['post_publish_time'], '%Y-%m-%d %H:%M:%S')
            next_time = datetime.strptime(next_post['post_publish_time'], '%Y-%m-%d %H:%M:%S')

            if start is None:
                start = current_time  # 设置开始时间为下一个帖子的时间

            end = next_time  # 更新结束时间为当前帖子的时间
        else:
            # 如果当前或下一个帖子不包含'ST', 结束当前ST状态的时间段
            if start is not None and end is not None:
                # start保留到日
                start = datetime(start.year, start.month, start.day)
                # end保留到日
                end = datetime(end.year, end.month, end.day)
                periods.append((start, end))
                start = None
                end = None

    # 添加最后一个时间段
    if start is not None and end is not None:
        # start保留到日
        start = datetime(start.year, start.month, start.day)
        # end保留到日
        end = datetime(end.year, end.month, end.day)
        periods.append((start, end))

    return periods


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


def find_st_periods_strict(announcements):
    """
    Find periods when the stock was in ST status.

    :param announcements: List of announcements with 'post_title' and 'post_publish_time'.
    :return: List of tuples with the start and end dates of ST periods.
    """
    st_periods = []
    st_start = None

    for announcement in sorted(announcements, key=lambda x: x['post_publish_time']):
        title = announcement['post_title']
        date = datetime.strptime(announcement['post_publish_time'], '%Y-%m-%d %H:%M:%S')

        # Mark the start of an ST period
        if ('ST' in title or ('实施' in title and '风险警示' in title)) and not st_start:
            st_start = date

        # Mark the end of an ST period
        elif ('撤销' in title and st_start and '警示' in title) and '申请' not in title and '继续' not in title and '实施' not in title:
            st_end = date
            st_start = datetime(st_start.year, st_start.month, st_start.day)
            # end保留到日
            st_end = datetime(st_end.year, st_end.month, st_end.day)
            st_periods.append((st_start, st_end))
            st_start = None  # Reset for the next ST period
    # 添加最后一个时间段,end为当前时间
    if st_start:
        st_end = datetime.now()
        st_start = datetime(st_start.year, st_start.month, st_start.day)
        # end保留到日
        st_end = datetime(st_end.year, st_end.month, st_end.day)
        st_periods.append((st_start, st_end))

    return st_periods

def get_good_combinations():
    """
    获取表现好的组合
    :return:
    """
    statistics = read_json('../final_zuhe/statistics_target_key.json')
    statistics_filter = {k: v for k, v in statistics.items() if v['trade_count'] > 10 and (v['three_befor_year_rate'] > 0.2 or v['three_befor_year_count'] > 10)}
    good_statistics = {k: v for k, v in statistics_filter.items() if v['ratio'] <= 0.1 and v['three_befor_year_count_thread_ratio'] <= 0.1}
    good_statistics = dict(sorted(good_statistics.items(), key=lambda x: (-x[1]['ratio'], x[1]['trade_count']), reverse=True))
    # 将结果写入文件
    write_json('../final_zuhe/good_statistics.json', good_statistics)
    return good_statistics

def process_file(file_path, target_date, good_keys, good_statistics, gen_signal_func):
    # 为每个文件执行的处理逻辑
    stock_data = load_data(file_path)
    stock_data = gen_signal_func(stock_data)
    target_data = stock_data[stock_data['日期'] == pd.to_datetime(target_date)]

    if target_data.empty:
        return None
    satisfied_combinations = dict()
    for good_key in good_keys:
        signal_data = gen_signal(target_data, good_key.split(':'))
        if signal_data['Buy_Signal'].values[0]:
            satisfied_combinations[good_key] = good_statistics[good_key]
    if satisfied_combinations:
        return {'stock_name': os.path.basename(file_path).split('.')[0], 'satisfied_combinations': satisfied_combinations}
    return None

def sort_good_stocks(good_stocks):
    """
    Sorts a list of stocks based on the minimum value of 'ratio' and 'three_befor_year_count_thread_ratio'
    within each 'satisfied_combinations' of a stock.

    :param good_stocks: List of stocks with their respective satisfied combinations and statistics.
    :return: List of stocks sorted in ascending order based on the minimum value of the mentioned ratios.
    """
    for stock in good_stocks:
        # Extract all ratio values and three_befor_year_count_thread_ratio values from each satisfied_combination
        ratios = [values['ratio'] for values in stock['satisfied_combinations'].values()]
        three_befor_ratios = [values['three_befor_year_count_thread_ratio'] for values in stock['satisfied_combinations'].values()]

        # Calculate the minimum ratio for each stock
        stock['min_ratio'] = min(ratios + three_befor_ratios)

    # Sort the stocks based on the minimum ratio calculated
    sorted_stocks = sorted(good_stocks, key=lambda x: x['min_ratio'])

    return sorted_stocks

def get_target_date_good_stocks_mul(file_path, target_date, gen_signal_func):
    get_good_combinations()
    start_time = time.time()
    good_statistics = read_json('../final_zuhe/good_statistics.json')
    good_keys = list(good_statistics.keys())

    # 将 target_date 转换为 datetime 一次
    target_date = pd.to_datetime(target_date)

    # 使用 multiprocessing 处理文件
    pool = Pool(20)
    file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]
    results = pool.starmap(process_file, [(file, target_date, good_keys, good_statistics, gen_signal_func) for file in file_paths])
    pool.close()
    pool.join()

    # 过滤出非空结果
    good_stocks = [result for result in results if result is not None]

    # 将good_stocks进行排序
    good_stocks = sort_good_stocks(good_stocks)
    # 其他逻辑不变
    write_json('../final_zuhe/select_{}.json'.format(target_date.strftime('%Y-%m-%d')), good_stocks)
    print(good_stocks)
    end_time = time.time()
    print('get_target_date_good_stocks time cost: {}'.format(end_time - start_time))


def get_target_date_good_stocks(file_path, target_date, gen_signal_func=gen_full_all_basic_signal):
    """
    获取指定时间的好股票(好股票的含义为在指定时间满足好指标)
    :return:
    """
    # 开始计时
    start_time = time.time()
    good_statistics = read_json('../final_zuhe/good_statistics.json')
    good_keys = list(good_statistics.keys())
    good_stocks = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            full_path = os.path.join(root, file)
            stock_data = load_data(full_path)
            stock_data = gen_signal_func(stock_data)
            # 获取指定时间的数据
            target_date = pd.to_datetime(target_date)
            target_data = stock_data[stock_data['日期'] == target_date]

            if target_data.empty:
                continue
            satisfied_combinations = dict()
            for good_key in good_keys:
                signal_data = gen_signal(target_data, good_key.split(':'))
                # 判断signal_data的Buy_Signal是否为True
                if signal_data['Buy_Signal'].values[0]:
                    satisfied_combinations[good_key] = good_statistics[good_key]
                    print(file)
            if satisfied_combinations:
                # 将满足条件的股票加入到good_stocks中，注意要避免UnicodeDecodeError
                good_stocks.append({'stock_name': file.split('.')[0], 'satisfied_combinations': satisfied_combinations})
    # 将结果写入文件,文件名为slect_{target_date}.json
    write_json('../final_zuhe/select_{}.json'.format(target_date.strftime('%Y-%m-%d')), good_stocks)
    print(good_stocks)
    # 结束计时
    end_time = time.time()
    print('get_target_date_good_stocks time cost: {}'.format(end_time - start_time))

def test_back_all():
    """
    进行分层函数的性能测试
    :return:
    """
    should_write_data = True  # 新增变量来控制是否写入数据
    file_name = Path('../back/gen/zuhe') / f"浪潮软件.json"
    result_df_dict = read_json(file_name)
    final_combination = read_json('../back/gen/statistics_all.json').keys()
    # 将final_combination中的元素以':'分割
    final_combinations = [combination.split(':') for combination in final_combination]
    # 随机保留final_combinations前100个元素
    final_combinations = random.sample(final_combinations, 2000)
    start_time = time.time()
    try:
        zero_combination = set()  # Using a set for faster lookups
        full_name = '../daily_data_exclude_new_can_buy/浪潮软件_600756.txt'
        data = load_data(full_name)
        data = gen_full_all_basic_signal(data)
        file_name = Path('../back/gen/zuhe') / f"{data['名称'].iloc[0]}.json"
        recent_file_name = Path('../back/gen/single') / f"{data['名称'].iloc[0]}.json"
        # file_name = Path('../back/zuhe') / f"C夏厦.json"
        file_name.parent.mkdir(parents=True, exist_ok=True)

        # 一次性读取JSON
        result_df_dict = read_json(file_name)

        recent_result_df_dict = {}

        # 优化：过滤和处理逻辑提取为单独的函数
        final_combinations, zero_combination = filter_combinations(result_df_dict, final_combinations)

        # 处理每个组合
        for combination in final_combinations:
            combination_key = ':'.join(combination)
            signal_data = gen_signal(data, combination)

            # 如果Buy_Signal全为False，则不进行回测
            if not signal_data['Buy_Signal'].any():
                result_df_dict[combination_key] = create_empty_result()
                recent_result_df_dict[combination_key] = create_empty_result()
                continue

            results_df = backtest_strategy_low_profit(signal_data)
            processed_result = process_results_with_year(results_df, 1)

            if processed_result:
                result_df_dict[combination_key] = processed_result
                recent_result_df_dict[combination_key] = processed_result

    except Exception as e:
        traceback.print_exc()
    finally:
        # 写入文件
        write_json(file_name, result_df_dict)
        write_json(recent_file_name, recent_result_df_dict)
        end_time = time.time()
        print(
            f"{full_name} 耗时：{end_time - start_time}秒 data长度{data.shape[0]} zero_combination len: {len(zero_combination)} final_combinations len: {len(final_combinations)}")





if __name__ == '__main__':
    # data_1 = load_data('../daily_data_exclude_new/C润本_603193.txt')
    # data_2 = load_data_optimized('../daily_data_exclude_new/东方电子_000682.txt')
    # data_2 = gen_full_all_basic_signal(data_2)
    #
    # data_signal = gen_signal(data_2,"振幅_大于_10_固定区间_signal:最高_10日_大极值signal".split(':'))
    # data_signal_op = gen_signal_optimized(data_2, "振幅_大于_10_固定区间_signal:最高_10日_大极值signal".split(':'))
    # # 分别采用两种信号生成方式统计时间
    # print(timeit.timeit('gen_signal(data_2,"振幅_大于_10_固定区间_signal:最高_10日_大极值signal".split(":"))', globals=globals(), number=1000))
    # print(timeit.timeit('gen_signal_optimized(data_2,"振幅_大于_10_固定区间_signal:最高_10日_大极值signal".split(":"))', globals=globals(), number=1000))
    # sta = read_json('..\back\zuhe\C夏厦.json')

    # # 分别采用两种加载方式统计时间
    #
    # print(timeit.timeit('load_data_optimized("../daily_data_exclude_new/东方电子_000682.txt")', globals=globals(), number=100))
    # print(timeit.timeit('load_data("../daily_data_exclude_new/东方电子_000682.txt")', globals=globals(), number=100))
    # temp_dict = read_json(f'../back/combination_list_2.json')
    # result_combination_list = temp_dict['result_combination_list']
    # full_combination_list = temp_dict['full_combination_list']
    # newest_indicators = temp_dict['newest_indicators']

    # 获取目录下文件的所有key的交集

    # for root, ds, fs in os.walk('../back/zuhe'):
    #     for f in fs:
    #         fullname = os.path.join(root, f)
    #         data = read_json(fullname)
    #         # 获取data的所有key
    #         keys = list(data.keys())
    #         # 如果keys长度小于1000
    #         if len(keys) < 1000:
    #             print(fullname)
    #         # 获取所有文件的key的交集
    #         if 'all_keys' not in locals():
    #             all_keys = set(keys)
    #         else:
    #             all_keys = all_keys & set(keys)
    # statistics = read_json('../back/statistics_all.json')
    # back_all_stock('../daily_data_exclude_new_can_buy/', '../back/complex', gen_signal_func=mix, backtest_func=backtest_strategy_low_profit)

    # data = load_data('../daily_data_exclude_new/九阳股份_002242.txt')
    # data = gen_full_all_basic_signal(data)
    # print(data)

    # file_set = set()
    # new_file_set = set()
    # file_list = []
    # for root, ds, fs in os.walk('../daily_data_exclude_new_can_buy'):
    #     for f in fs:
    #         file_set.add(f.split('_')[0])
    #         file_list.append(f.split('_')[0])
    # #找到file_list中重复的元素
    # print([item for item, count in collections.Counter(file_list).items() if count > 1])
    #
    # for root, ds, fs in os.walk('../back/zuhe'):
    #     for f in fs:
    #         fullname = os.path.join(root, f)
    #         filename = f.split('.')[0]
    #         new_file_set.add(filename)
    # print(file_set - new_file_set)

    # get_target_date_good_stocks_mul('../daily_data_exclude_new_can_buy', '2023-12-08', gen_signal_func=gen_full_all_basic_signal)
    # good_data = sort_good_stocks(read_json('../final_zuhe/select_2023-12-07.json'))
    # print(good_data)


    # back_all_stock('../daily_data_exclude_new_can_buy/', '../back/complex', gen_signal_func=mix, backtest_func=backtest_strategy_low_profit)
    #
    # strategy('../daily_data_exclude_new_can_buy/第一医药_600833.txt', gen_signal_func=mix_back,backtest_func=backtest_strategy_low_profit)
    #
    # back_all_stock('../daily_data_exclude_new_can_buy/', '../back/complex', gen_signal_func=mix_back,
    #                backtest_func=backtest_strategy_low_profit)

    # while True:
    #     test_back_all()
    # statistics = read_json('../back/statistics_target_key.json')
    statistics = read_json('../back/gen/statistics_all.json') # 大小 113926
    # statistics = read_json('../final_zuhe/statistics_target_key.json')
    # statistics = read_json('../back/gen/statistics_target_key.json')
    # temp_data = read_json('../back/gen/zuhe/贵绳股份.json')
    # good_statistics = get_good_combinations()
    sublist_list = read_json('../back/gen/sublist.json') #大小 55044
    statistics = dict(sorted(statistics.items(), key=lambda x: (-x[1]['ratio'], x[1]['trade_count']), reverse=True))
    # sublist_list中的元素也是list，帮我对sublist_list进行去重
    # 将statistics中trade_count大于100的筛选出来，并且按照average_profit降序排序
    statistics_new = {k: v for k, v in statistics.items() if v['trade_count'] > 100} # 100交易次数以上 52765 最好数据 513次 ratio:0.0448
    statistics_new_1000 = {k: v for k, v in statistics.items() if v['trade_count'] > 1000}  # 1000交易次数以上 46649 最好数据 2776次 ratio:0.0764
    statistics_profit_temp = {k: v for k, v in statistics_new.items() if '实体_' not in k and '开盘_大于_20_固定区间' not in k and '收盘_大于_20_固定区间' not in k and '最高_大于_20_固定区间' not in k and '最低_大于_20_固定区间' not in k}
    statistics_profit = sorted(statistics_profit_temp.items(), key=lambda x: x[1]['average_profit'], reverse=True)

    statistics_average_days_held = sorted(statistics_new.items(), key=lambda x: x[1]['average_days_held'], reverse=False)
    statistics_1w_profit = sorted(statistics_new.items(), key=lambda x: x[1]['average_1w_profit'],
                                          reverse=True)
    print(len(statistics))
    print(len(sublist_list))

    # notice_list = read_json('../announcements/000682.json')
    # periods = find_st_periods(notice_list)
    # strict_periods = find_st_periods_strict(notice_list)
    # print(periods)
    # df = pd.read_csv('../daily_data_exclude_new_can_buy/ST同洲_002052.txt')
    # df['Max_rate'] = 10
    # df['日期'] = pd.to_datetime(df['日期'])
    # for end, start in periods:
    #     mask = (df['日期'] >= start) & (df['日期'] <= end)
    #     df.loc[mask, 'Max_rate'] = 5
    # print(df)

    # try:
    #     with open('../back/statistics_back.json', 'r', encoding='utf-8') as f:
    #         statistics = json.load(f)
    # except UnicodeDecodeError:
    #     with open('../back/statistics_back.json', 'r', encoding='gbk') as f:
    #         statistics = json.load(f)
    # with open('../back/statistics_1.json', 'w', encoding='utf-8') as f:
    #     json.dump(statistics, f, ensure_ascii=False)

    # statistics_op = read_json('../back/statistics_target_key.json')
    #
    # # 将new_statistics_10000中的元素按照trade_count进行排序
    # new_statistics_10000 = sorted(statistics_op.items(), key=lambda x: x[1]['trade_count'], reverse=False)
    # print(new_statistics_10000)
    # print(len(new_statistics_10000))
    # print(len(new_statistics_ratio))
    # print(55)

    # 找到trade_count为0的指标，将key保持在zero_combinations_set中
    # zero_combinations_set = set()
    # for key, value in statistics.items():
    #     if value['trade_count'] == 0:
    #         zero_combinations_set.add(key)
    # exist_combinations_set = set(statistics.keys())
    # newest_indicators = filter_combination_list(full_combination_list, statistics, zero_combinations_set)
    # newest_indicators_op = optimized_filter_combination_list(full_combination_list, statistics, zero_combinations_set)
    # back_layer_all('../daily_data_exclude_new', gen_signal_func=gen_full_all_basic_signal,backtest_func=backtest_strategy_low_profit)
