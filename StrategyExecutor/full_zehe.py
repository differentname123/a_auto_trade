# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023-11-24 15:30
:last_date:
    2023-11-24 15:30
:description:
    
"""
import logging

import concurrent

import multiprocessing
import os
import random

import akshare as ak
import re
import scipy
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
from scipy.stats import norm
from sympy.physics.quantum.identitysearch import scipy
import matplotlib

from InfoCollector.save_all import save_all_data_mul, get_price, fix_st, save_index_data
from StrategyExecutor.CommonRandomForestClassifier import load_rf_model, get_all_good_data_with_model_list
from StrategyExecutor.basic_daily_strategy import clear_other_clo

matplotlib.use('Agg')

from StrategyExecutor.daily_strategy import mix, gen_daily_buy_signal_26, gen_daily_buy_signal_27, \
    gen_daily_buy_signal_25, gen_daily_buy_signal_28, gen_true, gen_daily_buy_signal_29, gen_daily_buy_signal_30, \
    select_stocks, mix_back, gen_daily_buy_signal_31
from StrategyExecutor.strategy import back_all_stock, strategy
from StrategyExecutor.zuhe_daily_strategy import gen_full_all_basic_signal, filter_combinations, filter_combinations_op, \
    create_empty_result, process_results_with_year, gen_full_zhishu_basic_signal, process_results_with_every_year, \
    process_results_with_every_period, statistics_zuhe_gen_both_single_every_period, statistics_zuhe_good, \
    gen_26_zhibiao

pd.options.mode.chained_assignment = None  # 关闭SettingWithCopyWarning
import warnings
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')
MODEL_PATH = '../model/all_models'
def get_buy_price(close_price):
    """
    根据收盘价，返回买入价格
    买入价格为收盘价格的1.001倍，向上保留两位小数
    :param close_price:
    :return:
    """
    buy_price = close_price * 1.0025
    buy_price = math.ceil(buy_price * 100) / 100
    return buy_price

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
            high_list = []
            # 遍历i 后面的三个数据
            for xx in range(1, 4):
                j = i + xx
                if i + xx < len(data):
                    high_price = data['最高'].iloc[j]
                    if data['收盘'].iloc[j - 1] != 0:
                        high_price_ratio = 100 * (high_price - data['收盘'].iloc[j - 1]) / data['收盘'].iloc[j - 1]
                    else:
                        high_price_ratio = 0
                    high_list.append((data['涨跌幅'].iloc[j], high_price_ratio))
                else:
                    high_list.append((0, 0))

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
                data.at[i, '数量'] = total_shares  # 记录买入数量

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
            profit = (sell_price - buy_price) * total_shares  # 每次卖出100股
            total_profit += profit
            total_cost = buy_price * total_shares
            days_held = j - buy_index
            if profit < 0.1:
                days_held = 3
            results.append([name, symbol, buy_date, buy_date, buy_price, sell_date, sell_price, profit, total_profit, total_cost,
                            days_held, high_list, i])

        i += 1

    results_df = pd.DataFrame(results,
                              columns=['名称', '代码', '日期', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit',
                                       'Total_Profit', 'total_cost',
                                       'Days Held', 'high_list', 'Buy_Index'])

    return results_df

def backtest_strategy_low_profit_target_data_specia(data, date, price):
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
        # 进行时间判断
        if data['日期'].iloc[i] == date:
            high_list = []
            # 遍历i 后面的三个数据
            for xx in range(1, 4):
                j = i + xx
                if i + xx < len(data):
                    high_price = data['最高'].iloc[j]
                    high_price_ratio = 100 * (high_price - data['收盘'].iloc[j - 1]) / data['收盘'].iloc[j - 1]
                    high_list.append((data['涨跌幅'].iloc[j], high_price_ratio))
                else:
                    high_list.append((0, 0))

            buy_price = price
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
                data.at[i, '数量'] = total_shares  # 记录买入数量

                j += 1
                if j - i > 1:
                    break

            # 如果找到了满足卖出条件的日期
            if j < len(data):
                sell_price = get_sell_price(buy_price)
                if data['开盘'].iloc[j] > sell_price:
                    sell_price = data['开盘'].iloc[j]
                if data['最高'].iloc[j] < sell_price:
                    sell_price = data['收盘'].iloc[j]
            else:
                # 如果没有找到，强制在最后一天卖出
                j = len(data) - 1
                sell_price = data['收盘'].iloc[j]

            sell_date = data['日期'].iloc[j]
            profit = (sell_price - buy_price) * total_shares  # 每次卖出100股
            total_profit += profit
            total_cost = buy_price * total_shares
            days_held = j - buy_index
            results.append([name, symbol, buy_date, buy_price, sell_date, sell_price, profit, total_profit, total_cost,
                            days_held, high_list, i])

        i += 1

    results_df = pd.DataFrame(results,
                              columns=['名称', '代码', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit',
                                       'Total_Profit', 'total_cost',
                                       'Days Held', 'high_list', 'Buy_Index'])

    return results_df

def backtest_strategy_low_profit_target_data(data, date):
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
        # 进行时间判断
        if data['日期'].iloc[i] == date:
            high_list = []
            # 遍历i 后面的三个数据
            for xx in range(1, 4):
                j = i + xx
                if i + xx < len(data):
                    high_price = data['最高'].iloc[j]
                    high_price_ratio = 100 * (high_price - data['收盘'].iloc[j - 1]) / data['收盘'].iloc[j - 1]
                    high_list.append((data['涨跌幅'].iloc[j], high_price_ratio))
                else:
                    high_list.append((0, 0))

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
                data.at[i, '数量'] = total_shares  # 记录买入数量

                j += 1
                # if j - i > 1:
                #     break

            # 如果找到了满足卖出条件的日期
            if j < len(data):
                sell_price = get_sell_price(buy_price)
                if data['开盘'].iloc[j] > sell_price:
                    sell_price = data['开盘'].iloc[j]
                # if data['最高'].iloc[j] < sell_price:
                #     sell_price = data['收盘'].iloc[j]
            else:
                # 如果没有找到，强制在最后一天卖出
                j = len(data) - 1
                sell_price = data['收盘'].iloc[j]

            sell_date = data['日期'].iloc[j]
            profit = (sell_price - buy_price) * total_shares  # 每次卖出100股
            total_profit += profit
            total_cost = buy_price * total_shares
            days_held = j - buy_index
            results.append([name, symbol, buy_date, buy_price, sell_date, sell_price, profit, total_profit, total_cost,
                            days_held, high_list, i])

        i += 1

    results_df = pd.DataFrame(results,
                              columns=['名称', '代码', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit',
                                       'Total_Profit', 'total_cost',
                                       'Days Held', 'high_list', 'Buy_Index'])

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
    data = pd.read_csv(file_path, low_memory=False)
    name, code = parse_filename(file_path)
    if '时间' in data.columns:
        data = data.rename(columns={'时间': '日期'})
    data['日期'] = pd.to_datetime(data['日期'])
    data['名称'] = name
    data['代码'] = code
    data['数量'] = 0
    data.sort_values(by='日期', ascending=True, inplace=True)
    # 过滤掉收盘价小于等于0的数据
    data = data[data['收盘'] > 0]

    # 查找并移除第一个日期，如果与其他日期不连续超过30天
    date_diff = data['日期'].diff(-1).abs()
    filtered_diff = date_diff[date_diff > pd.Timedelta(days=30)]

    # 过滤时间大于2024年的数据
    data = data[data['日期'] > pd.Timestamp('2024-01-01')]
    data = data[data['日期'] > pd.Timestamp('2018-01-01')]

    # 如果有大于30天的断层
    if not filtered_diff.empty:
        cutoff_index = filtered_diff.idxmax()
        if cutoff_index and cutoff_index != 0:
            data = data.loc[cutoff_index + 1:]  # 跳过第一个数据点

    # 重置索引
    data.reset_index(drop=True, inplace=True)
    data['Buy_Signal'] = (data['涨跌幅'] < 0.95 * data['Max_rate'])
    return data

def load_full_data(file_path):
    """
    加载全部数据，不进行时间截断
    :param file_path:
    :return:
    """
    data = pd.read_csv(file_path)
    name, code = parse_filename(file_path)
    if '时间' in data.columns:
        data = data.rename(columns={'时间': '日期'})
    data['日期'] = pd.to_datetime(data['日期'])
    data['名称'] = name
    data['代码'] = code
    data['数量'] = 0
    data = data[data['日期'] > pd.Timestamp('2023-01-01')]
    data.sort_values(by='日期', ascending=True, inplace=True)
    # 重置索引
    data.reset_index(drop=True, inplace=True)
    data['Buy_Signal'] = (data['涨跌幅'] < 0.95 * data['Max_rate'])
    return data

def load_data_limit(file_path, limit=100):
    try:
        data = pd.read_csv(file_path)
        # 保留最近的limit条数据，考虑data长度小于limit的情况
        data = data[-limit:]
    except Exception:
        print(file_path)
        traceback.print_exc()
        return None
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

    # data['Max_rate'] = data['名称'].str.contains('st', case=False).map({True: 5, False: 10})
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
    data['Buy_Signal'] = ((data['涨跌幅'] >= -(data['Max_rate'] - 1.0 / data['收盘'])) & (data['涨跌幅'] <= (data['Max_rate'] - 1.0 / data['收盘'])) & (data['收盘'] >= 3) & (data['Max_rate'] > 0.1))
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
    st_end = None
    # 先将announcement['post_publish_time']变成时间格式然后announcements按照时间排序
    announcements = sorted(announcements, key=lambda x: datetime.strptime(x['post_publish_time'], '%Y-%m-%d %H:%M:%S'))

    for announcement in announcements:
        title = announcement['post_title']
        date = datetime.strptime(announcement['post_publish_time'], '%Y-%m-%d %H:%M:%S')
        # Mark the start of an ST period
        if ('ST' in title or ('风险警示' in title)) and not st_start and st_end != date:
            st_start = date
        # Mark the end of an ST period
        elif ('撤' in title and st_start) and '申请' not in title and '继续' not in title and '实施' not in title:
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

def get_good_statistic(date):
    bad_statistics = {}
    bad_output_file_path = '../final_zuhe/back/' + date + 'bad_back.json'
    statistics = read_json(bad_output_file_path)
    for k, v in statistics.items():
        bad_statistics.update(v['satisfied_combinations'])
    return bad_statistics

def judge_json(json_data):
    if 'all' in json_data.keys():
        if json_data['all']['trade_count'] > 1000:
            for k, v in json_data.items():
                if k != 'all' and '-' in k:
                    return False
                if v['ratio'] > 0.1:
                    return False
        else:
            return False
    else:
        return False
    return True

# def judge_json(json_data):
#     if 'all' not in json_data.keys():
#         if json_data['ratio'] <= 0.01 and json_data['size_of_result_df'] < 10:
#             return True
#         else:
#             return False
#     else:
#         return False
#     return False
def get_good_combinations():
    """
    获取表现好的组合
    :return:
    """
    # date = '2024-01-17'
    # statistics = read_json('../final_zuhe/statistics_target_key.json')
    statistics = read_json('../back/bad_gen/statistics_all.json')
    # statistics = read_json('../back/gen/statistics_all.json')
    # statistics = read_json('../final_zuhe/good_statistics.json')
    # bad_statistics = get_good_statistic(date)
    bad_statistics = {}
    statistics_all = {}
    # # 所有的指标都应该满足10次以上的交易
    statistics_new = {k: v for k, v in statistics.items() if '指数' not in k and judge_json(v)}  # 100交易次数以上 13859
    good_ratio_keys = {k: v for k, v in statistics_new.items()
                       if True
                       # or (v['all']['ratio'] <= 0.1)
                       # or (v['ratio'] <= 0.05 and (v['three_befor_year_count_thread_ratio'] <= 0.05) and v['three_befor_year_count'] > 0 and v['trade_count'] > 500)
                       # or (v['ratio'] == 0 and v['trade_count'] > 20)
                       # or (v['three_befor_year_count_thread_ratio'] == 0 and v['three_befor_year_count'] > 20)
                       # or ((v['ratio'] <= 0.07) and v['trade_count'] > 1000)
                       # or ((v['ratio'] <= 0.06) and v['trade_count'] > 1000)
                       }

    # good_1w_keys = {k: v for k, v in statistics.items()
    #                    if (v['average_1w_profit'] >= 100) and (v['trade_count'] >= 10)
    #                    }
    # statistics_all.update(good_1w_keys)

    # good_ratio_keys = {k: v for k, v in statistics_new.items()
    #                    if
    #                    (v['average_days_held'] <= (1 + v['ratio']) + 0.001) # 代表两天之内一定会卖出去
    #                    or (v['than_1_average_days_held'] >= 2.01 and v['ratio'] <= 0.05)
    #                    or (v['than_1_average_days_held'] >= 2.01 and v['ratio'] <= 0.06 and v['three_befor_year_count_thread_ratio'] <= 0.06)
    #                    }


    # statistics_new = {k: v for k, v in statistics.items() if v['trade_count'] > 10 and (v['three_befor_year_count'] >= 10)} # 100交易次数以上 13859
    # # statistics_new = {k: v for k, v in statistics_new.items() if v['three_befor_year_count_thread_ratio'] <= 0.10 and v['three_befor_year_rate'] >= 0.2}
    # good_ratio_keys = {k: v for k, v in statistics_new.items() if v['ratio'] <= 0.1 and v['1w_rate'] >= 100 and v['average_1w_profit'] >= 100 and v['three_befor_year_count_thread_ratio'] <= 0.1 }
    # # good_ratio_keys_day = {k: v for k, v in statistics_new.items() if v['than_1_average_days_held'] <= 3 or v["average_1w_profit"] >= 100}
    # # good_ratio_keys.update(good_ratio_keys_day)

    #
    #
    # good_fitness_keys = {k: v for k, v in statistics_new.items() if v['than_1_average_days_held'] <= 3.84 and v['average_1w_profit'] >= 100}
    # statistics_fitness = dict(sorted(good_fitness_keys.items(), key=lambda x: x[1]['than_1_average_days_held'], reverse=True))
    #
    #
    # good_1w_keys = {k: v for k, v in statistics_new.items() if v['than_1_average_days_held'] <= 2 or v["ratio"] <= 0.05}
    # statistics_1w = dict(sorted(good_1w_keys.items(), key=lambda x: x[1]['average_1w_profit'], reverse=True))
    #
    #
    # good_1w_rate_keys = {k: v for k, v in statistics_new.items() if (v['ratio'] <= 0.1 and v['three_befor_year_count_thread_ratio'] <= 0.05) or v['1w_rate'] >= 200}
    # statistics_1w_rate = dict(sorted(good_1w_rate_keys.items(), key=lambda x: x[1]['1w_rate'], reverse=True))
    #
    #
    # # 将所有的指标都写入文件,去重
    # statistics_all = dict()
    # # statistics_all.update(statistics_ratio)
    # statistics_all.update(statistics_fitness)
    # # statistics_all.update(statistics_1w)
    # # statistics_all.update(statistics_1w_rate)

    statistics_all.update(good_ratio_keys) # 761
    statistics_all = dict(sorted(statistics_all.items(), key=lambda x: (-x[1]['all']['ratio'], x[1]['all']['trade_count']), reverse=True))
    write_json('../final_zuhe/good_statistics.json', statistics_all)

def filter_combinations_good(zero_combinations_set, good_keys):
    """
    过滤已存在和无效的组合
    """
    final_combinations_set = good_keys
    return [comb.split(':') for comb in final_combinations_set if
            not any(frozenset(comb.split(':')) >= zc for zc in zero_combinations_set)], zero_combinations_set

def process_file_op(file_path, target_date, good_keys, good_statistics, gen_signal_func, index_data):
    # 为每个文件执行的处理逻辑
    # origin_data = load_data_limit('../daily_data_exclude_new_can_buy/五方光电_002962.txt', 200)
    origin_data = load_data_limit(file_path, 200)
    stock_data = origin_data.copy()
    if stock_data is None:
        return None
    stock_data = gen_signal_func(stock_data)
    target_data = stock_data[stock_data['日期'] == pd.to_datetime(target_date)]
    # 将index_data中的数据合并到target_data中
    target_data = pd.merge(target_data, index_data, on='日期', how='left')
    # 获取target_data中值为False的列名
    false_columns = target_data.columns[(target_data == False).any()]
    # 将false_columns转换为frozenset
    false_columns = [frozenset([col]) for col in false_columns]
    final_combinations, zero_combination = filter_combinations_good(false_columns, good_keys)

    if target_data.empty:
        return None
    satisfied_combinations = dict()
    # for good_key in final_combinations:
    #     good_key_str = ':'.join(good_key)
    #     signal_data = gen_signal(target_data, good_key)
    #     if signal_data['Buy_Signal'].values[0]:
    #         satisfied_combinations[good_key_str] = good_statistics[good_key_str]

    stock_data = origin_data.copy()
    good_key = "超级短线_26"
    signal_data = gen_daily_buy_signal_26(stock_data)
    target_data = signal_data[signal_data['日期'] == pd.to_datetime(target_date)]
    if target_data['Buy_Signal'].values[0]:
        satisfied_combinations[good_key] = {'ratio': 0.0721, 'three_befor_year_count_thread_ratio': 0.0755,
                                            'three_befor_year_rate': 0.244, 'than_1_average_days_held': 4.4,
                                            'average_1w_profit': 56.929, 'average_days_held': 1.2556,
                                            'three_befor_year_count': 6117, 'trade_count': 25057, '1w_rate': 0.0}

    if satisfied_combinations:
        return {'stock_name': os.path.basename(file_path).split('.')[0],
                'satisfied_combinations': satisfied_combinations,
                '收盘': target_data['收盘'].values[0], '涨跌幅': target_data['涨跌幅'].values[0],
                'Max_rate': int(target_data['Max_rate'].values[0])}
    return None

def process_file(file_path, target_date, good_keys, good_statistics, gen_signal_func):
    # 为每个文件执行的处理逻辑
    stock_data = load_data(file_path)
    # stock_data = load_data('../daily_data_exclude_new_can_buy/易普力_002096.txt')
    if stock_data is None:
        return None
    stock_data = gen_signal_func(stock_data)
    target_data = stock_data[stock_data['日期'] == pd.to_datetime(target_date)]

    if target_data.empty:
        return None
    satisfied_combinations = dict()
    for good_key in good_keys:
        signal_data = gen_signal(target_data, good_key.split(':'))
        if signal_data['Buy_Signal'].values[0]:
            satisfied_combinations[good_key] = good_statistics[good_key]

    good_key = "超级短线_26"
    signal_data = gen_daily_buy_signal_26(stock_data)
    target_data = signal_data[signal_data['日期'] == pd.to_datetime(target_date)]
    if target_data['Buy_Signal'].values[0]:
        satisfied_combinations[good_key] = {'ratio': 0.0721, 'three_befor_year_count_thread_ratio': 0.0755,
                                            'three_befor_year_rate': 0.244, 'than_1_average_days_held': 4.4,
                                            'average_1w_profit': 56.929, 'average_days_held': 1.2556,
                                            'three_befor_year_count': 6117, 'trade_count': 25057, '1w_rate': 0.0}

    if satisfied_combinations:
        return {'stock_name': os.path.basename(file_path).split('.')[0],
                'satisfied_combinations': satisfied_combinations,
                '收盘': target_data['收盘'].values[0], '涨跌幅': target_data['涨跌幅'].values[0],
                'Max_rate': int(target_data['Max_rate'].values[0])}
    return None


def sort_good_stocks_op(good_stocks):
    """
    Sorts a list of stocks based on a custom score calculated from
    'three_befor_year_rate', 'three_befor_year_count_thread_ratio', and 'ratio'.

    :param good_stocks: List of stocks with their respective satisfied combinations and statistics.
    :return: List of stocks sorted in ascending order based on the calculated score.
    """
    for stock in good_stocks:
        scores = []
        for values in stock['satisfied_combinations'].values():
            if 'all' in values:
                ratio = values['all']['ratio']
            else:
                ratio = values['ratio']

            # Calculate the custom score for each combination
            score = ratio
            scores.append(score)

        # Find the minimum score for each stock
        stock['min_score'] = min(scores)

    # Sort the stocks based on their minimum score
    sorted_stocks = sorted(good_stocks, key=lambda x: x['min_score'])

    return sorted_stocks


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
        three_befor_ratios = [values['three_befor_year_count_thread_ratio'] for values in
                              stock['satisfied_combinations'].values()]

        # Calculate the minimum ratio for each stock
        stock['min_ratio'] = min(ratios + three_befor_ratios)

    # Sort the stocks based on the minimum ratio calculated
    sorted_stocks = sorted(good_stocks, key=lambda x: x['min_ratio'])

    return sorted_stocks


def get_newest_stock():
    """
    获取最新时间的选股
    :return:
    """
    # 开始计时
    start_time = time.time()
    # get_good_combinations()
    # 获取当前时间，保留到日
    target_date = datetime.now()
    good_statistics = read_json('../final_zuhe/good_statistics.json')
    good_keys = list(good_statistics.keys())
    save_all_data_mul()
    target_date = datetime(target_date.year, target_date.month, target_date.day)
    index_data = save_index_data()
    # 只保留大于2023的index_data
    index_data = index_data[index_data['日期'] > pd.to_datetime('20230101')]
    index_data = gen_full_zhishu_basic_signal(index_data, True)
    target_data = index_data[index_data['日期'] == pd.to_datetime(target_date)]
    # 获取target_data中值为False的列名
    false_columns = target_data.columns[(target_data == False).any()]
    # 将false_columns转换为frozenset
    false_columns = [frozenset([col]) for col in false_columns]
    final_combinations, zero_combination = filter_combinations_good(false_columns, good_keys)
    # 将final_combinations每个元素以:连接
    final_combinations = [':'.join(combination) for combination in final_combinations]
    # 过滤出key在final_combinations中的good_statistics
    final_statistics = {key: good_statistics[key] for key in final_combinations}
    print(str(target_date) + '的组合数量为：' + str(len(final_combinations)))
    get_target_date_good_stocks_mul_op('../daily_data_exclude_new_can_buy', target_date,
                                    gen_full_all_basic_signal, final_combinations, final_statistics, index_data)
    # 结束计时
    end_time = time.time()
    print('耗时：', end_time - start_time)

def get_target_date_good_stocks_mul(file_path, target_date, gen_signal_func):
    # 开始计时
    get_good_combinations()
    start_time = time.time()
    good_statistics = read_json('../final_zuhe/good_statistics.json')
    good_keys = list(good_statistics.keys())

    # 将 target_date 转换为 datetime 一次
    target_date = pd.to_datetime(target_date)

    # 使用 multiprocessing 处理文件
    pool = Pool(multiprocessing.cpu_count())
    file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]
    results = pool.starmap(process_file,
                           [(file, target_date, good_keys, good_statistics, gen_signal_func) for file in file_paths])
    pool.close()
    pool.join()

    # 过滤出非空结果
    good_stocks = [result for result in results if result is not None]
    # 将good_stocks进行排序
    good_stocks = sort_good_stocks_op(good_stocks)
    # 其他逻辑不变

    print(good_stocks)
    total_price = 0
    final_good_stocks = []
    for stock_info in good_stocks:
        price = stock_info['收盘']
        if stock_info['涨跌幅'] >= -0.95 * stock_info['Max_rate'] and price >= 1:
            total_price += price
            final_good_stocks.append(stock_info)

    print('总价值：', total_price)
    end_time = time.time()
    print('get_target_date_good_stocks time cost: {}'.format(end_time - start_time))
    write_json('../final_zuhe/select/select_{}.json'.format(target_date.strftime('%Y-%m-%d')), final_good_stocks)
    # 结束计时
    end_time = time.time()
    print('耗时：', end_time - start_time)

def get_target_date_good_stocks_mul_op(file_path, target_date, gen_signal_func, good_keys, good_statistics, index_data):

    start_time = time.time()


    # 将 target_date 转换为 datetime 一次
    target_date = pd.to_datetime(target_date)

    # 使用 multiprocessing 处理文件
    pool = Pool(multiprocessing.cpu_count())
    file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]
    results = pool.starmap(process_file_op,
                           [(file, target_date, good_keys, good_statistics, gen_signal_func, index_data) for file in file_paths])
    pool.close()
    pool.join()

    # 过滤出非空结果
    good_stocks = [result for result in results if result is not None]
    # 将good_stocks进行排序
    good_stocks = sort_good_stocks_op(good_stocks)
    # 其他逻辑不变

    # print(good_stocks)
    total_price = 0
    final_good_stocks = []
    total_count = 0
    for stock_info in good_stocks:
        price = stock_info['收盘']
        if stock_info['涨跌幅'] >= -(stock_info['Max_rate'] - 1.0 / price) and price >= 3:
            total_price += price
            final_good_stocks.append(stock_info)
            total_count += 1

    print('总数量:{} 总价值：{}w'.format(total_count, round(total_price / 100, 2)))
    end_time = time.time()
    print('get_target_date_good_stocks time cost: {}'.format(end_time - start_time))
    write_json('../final_zuhe/select/select_{}.json'.format(target_date.strftime('%Y-%m-%d')), final_good_stocks)
    # 结束计时
    end_time = time.time()
    print('耗时：', end_time - start_time)


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
    write_json('../final_zuhe/select/select_{}.json'.format(target_date.strftime('%Y-%m-%d')), good_stocks)
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


def adjust_price_data(sub_data, index, temp_close, previous_close):
    """ 调整价格相关数据 """
    updated_high = max(sub_data.at[index, '最高'], temp_close)
    updated_low = min(sub_data.at[index, '最低'], temp_close)
    updated_amplitude = ((updated_high - updated_low) / previous_close) * 100
    updated_change_rate = ((temp_close - previous_close) / previous_close) * 100

    sub_data.at[index, '收盘'] = temp_close
    sub_data.at[index, '最高'] = updated_high
    sub_data.at[index, '最低'] = updated_low
    sub_data.at[index, '振幅'] = updated_amplitude
    sub_data.at[index, '涨跌幅'] = updated_change_rate

    return updated_change_rate


def find_threshold_close(data, index, step, max_rate, gen_signal_func, search_direction):
    """ 寻找阈值收盘价 """
    original_close = data.at[index, '收盘']
    previous_close = data.at[index - 1, '收盘'] if index > 0 else None
    temp_close = original_close
    threshold_close = None

    while 0 < temp_close <= original_close + max_rate * previous_close / 100:
        sub_data = data[max(0, index - 100):index + 1].copy()
        updated_change_rate = adjust_price_data(sub_data, index, temp_close, previous_close)

        if abs(updated_change_rate) > max_rate:
            break

        if not gen_signal_func(sub_data).loc[sub_data['日期'] == sub_data.at[index, '日期'], 'Buy_Signal'].iloc[0]:
            threshold_close = temp_close
            break

        temp_close += step * search_direction

    return threshold_close

def find_threshold_close_target_date(data, index, step, max_rate, gen_signal_func, search_direction, now_status):
    """ 寻找阈值收盘价 """
    original_close = data.at[index, '收盘']
    previous_close = data.at[index - 1, '收盘'] if index > 0 else None
    temp_close = original_close
    threshold_close = None
    data['Buy_Signal'] = True
    if previous_close is None:
        return threshold_close
    try:
        while 0 < temp_close <= original_close + max_rate * previous_close / 100:
            sub_data = data[max(0, index - 100):index + 1].copy()
            updated_change_rate = adjust_price_data(sub_data, index, temp_close, previous_close)

            if abs(updated_change_rate) > max_rate:
                break

            if now_status != gen_signal_func(sub_data).loc[sub_data['日期'] == sub_data.at[index, '日期'], 'Buy_Signal'].iloc[0]:
                threshold_close = temp_close
                break

            temp_close += step * search_direction
    except Exception as e:
        traceback.print_exc()
        print(data)

    return threshold_close


def get_threshold_close(data, gen_signal_func=gen_daily_buy_signal_26, step=0.01):
    """
    获取data生成买入信号的收盘价阈值
    :param data: DataFrame, 包含股票数据
    :param gen_signal_func: 用于生成买入信号的函数
    :param step: 调整步长，默认为0.01
    :return: buy_data DataFrame，包含阈值收盘价
    """
    signal_data = gen_signal_func(data)
    buy_data = signal_data[signal_data['Buy_Signal'] == True]

    for index, row in buy_data.iterrows():
        current_row_index = data.index[data['日期'] == row['日期']].tolist()[0]

        # 检查是否有前一天的数据
        if current_row_index == 0:
            continue

        # 向上和向下寻找阈值
        threshold_close_up = find_threshold_close(data, current_row_index, step, row['Max_rate'], gen_signal_func, 1)
        threshold_close_down = find_threshold_close(data, current_row_index, -step, row['Max_rate'], gen_signal_func, 1)

        # 记录阈值收盘价
        buy_data.at[index, 'threshold_close_up'] = threshold_close_up
        buy_data.at[index, 'threshold_close_down'] = threshold_close_down

    return buy_data

def get_threshold_close_target_date(date, data, gen_signal_func=gen_daily_buy_signal_26, step=0.01):
    """
    获取data生成买入信号的收盘价阈值
    :param data: DataFrame, 包含股票数据
    :param gen_signal_func: 用于生成买入信号的函数
    :param step: 调整步长，默认为0.01
    :return: buy_data DataFrame，包含阈值收盘价
    """
    data['Buy_Signal'] = True
    signal_data = gen_signal_func(data)
    target_data = signal_data[signal_data['日期'] == date]



    if target_data.empty:
        return target_data
    index = target_data.index[0]
    now_status = target_data['Buy_Signal'].values[0]
    if now_status == True:
        target_data.at[index, 'threshold_close_up'] = target_data['收盘'].values[0]
        target_data.at[index, 'threshold_close_down'] = target_data['收盘'].values[0]
    # 向上和向下寻找阈值
    Max_rate = target_data['Max_rate'].values[0]
    # 将Max_rate转换为int
    Max_rate = int(Max_rate)
    threshold_close_up = find_threshold_close_target_date(data, index, step, Max_rate, gen_signal_func, 1, now_status)
    threshold_close_down = find_threshold_close_target_date(data, index, -step, Max_rate, gen_signal_func, 1, now_status)

    # 记录阈值收盘价
    target_data.at[index, 'threshold_close_up'] = threshold_close_up
    target_data.at[index, 'threshold_close_down'] = threshold_close_down

    return target_data


def parse_row(row):
    """
    解析文件中的每一行，将字符串格式的数据转换为元组列表。

    :param row: 文件中的一行数据。
    :return: 解析后的数据，为一个包含元组的列表。
    """
    # 使用正则表达式找出所有的元组
    tuples = re.findall(r'\[?\((.*?)\)\]?', row)
    parsed_data = []
    for t in tuples:
        # 将字符串分割成两部分，并转换为浮点数
        closing, highest = t.split(',')
        parsed_data.append((float(closing.strip()), float(highest.strip())))
    return parsed_data


def get_best_threshold(file_path, num_days):
    """
    穷举法，在指定的data获取最好的阈值组合，data为每天可能的表现，是个二维数组
    :param file_path: 文件路径
    :param num_days: 考虑的天数，决定循环的深度
    :return:
    """
    # 读取文件并将每一行数据进行解析
    parsed_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                parsed_lines.append(parse_row(line.strip()))

    # 定义新的卖出阈值范围进行测试，从-10%到10%
    new_thresholds_range = np.arange(0, 10.1, 0.1)

    # 初始化最佳利润和阈值组合
    best_profit_realistic = float('-inf')
    best_thresholds_realistic = None

    # 生成所有可能的阈值组合
    for thresholds in itertools.product(new_thresholds_range, repeat=num_days):
        total_profit_realistic = 0
        jian_count = 0
        fail_rate = 1
        for change_rate_list in parsed_lines:
            # 保留change_rate_list的前num_days个元素
            change_rate_list = change_rate_list[:num_days]
            profit_realistic = simulate_strategy_profitable(change_rate_list, list(thresholds))
            if profit_realistic < 0:
                jian_count += 1
            total_profit_realistic += profit_realistic

        if total_profit_realistic > best_profit_realistic:
            best_profit_realistic = total_profit_realistic
            best_thresholds_realistic = thresholds
            print('fail_rate: ', jian_count / len(parsed_lines))
            print('thresholds: ', thresholds)
            print('total_profit_realistic_rate: ', total_profit_realistic / len(parsed_lines))

    print('best_profit_realistic: ', best_profit_realistic)
    return best_thresholds_realistic


def fill_all_df(all_df, file_path, code_list, gen_signal_func=mix):
    """
    将阈值收盘价填充到all_df中
    :param all_df:
    :param file_path:
    :param code_list:
    :param gen_signal_func:
    :return:
    """
    fullname_map = {}
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            code = f.split('_')[1].split('.')[0]
            fullname = os.path.join(root, f)
            fullname_map[code] = fullname
    for code in code_list:
        data = load_data(fullname_map[code])
        buy_data = get_threshold_close(data, gen_signal_func)
        # 将buy_data和all_df匹配，条件为 代码 和 日期 相等，将buy_data中的threshold_close_up和threshold_close_down的值赋给all_df
        for index, row in buy_data.iterrows():
            all_df.loc[(all_df['代码'] == code) & (all_df['Buy Date'] == row['日期']), 'threshold_close_down'] = row[
                'threshold_close_down']
            all_df.loc[(all_df['代码'] == code) & (all_df['Buy Date'] == row['日期']), 'threshold_close_up'] = row[
                'threshold_close_up']
    return all_df


def count_min_profit_rate(file_path, all_df_file_path, gen_signal_func=mix):
    """
    计算最低的收益率，及后续的售出策略
    :param all_df_file_path:
    :return:
    """

    thread_day = 1
    # 加载all_df，但是不要把000001变成1
    all_df = pd.read_csv(all_df_file_path, dtype={'代码': str})
    # 截取all_df的后100个元素
    # all_df = all_df.tail(100)
    # 将all_df['Buy Date']转换为datetime类型
    all_df['Buy Date'] = pd.to_datetime(all_df['Buy Date'])
    success_data = all_df[all_df['Days Held'] == thread_day]
    # 获取success_data中的所有high_list这一列
    high_list = success_data['high_list'].values
    code_list = list(success_data['代码'].values)
    code_list = list(set(code_list))

    # # 将阈值收盘价填充到all_df中
    # all_df = fill_all_df(all_df, file_path, code_list, gen_signal_func)

    # 计算相对于收盘价的最优阈值
    # 1.简单进行排序
    high_list = [eval(high_list_item) for high_list_item in high_list]
    high_list = sorted(high_list, key=lambda x: x[thread_day - 1][1])
    print(high_list)

    # 穷尽得到指定天数内的最优阈值搭配
    # 将high_list写入'../back/complex/high_list.csv'
    # with open('../back/complex/high_list.csv', 'w') as f:
    #     for high_list_item in high_list:
    #         f.write(str(high_list_item) + '\n')
    # print(high_list)
    # get_best_threshold('../back/complex/high_list.csv', 2)
    # 将all_df写入'../back/complex/all_df_full.csv'
    all_df.to_csv('../back/complex/all_df_full.csv', index=False)


def calculate_normal_distribution(data):
    """
    计算数据的正态分布。

    Args:
      data: 数据列表，每个元素都是浮点数。

    Returns:
      均值、标准差、正态分布概率密度函数。
    """

    # 计算均值和标准差
    mean = np.mean(data)
    std = np.std(data)

    # 生成正态分布概率密度函数
    pdf = scipy.stats.norm(mean, std).pdf

    return mean, std, pdf


def simulate_strategy_profitable(data, sell_thresholds):
    """
    根据设定的卖出阈值来卖出股票的交易策略，并记录股票数量来计算利润。

    :param data: 包含每天（收盘价涨跌幅，最高价涨跌幅）的元组列表。
    :param sell_thresholds: 第1天和第2天的卖出阈值列表。
    :return: 策略的总利润。
    """
    initial_stock_price = 100  # 初始股价
    current_stock_price = initial_stock_price  # 当前股价
    total_cost = initial_stock_price  # 总成本
    stock_count = 1  # 初始股票数量
    total_profit = 0  # 总利润
    max_hold_day = len(sell_thresholds)  # 最大持有天数
    for i, day_data in enumerate(data):
        closing_change, highest_change = day_data

        # 检查当天的最高价是否达到卖出阈值
        if stock_count != 0 and i <= max_hold_day and current_stock_price * (
                1 + highest_change / 100) >= current_stock_price * (1 + sell_thresholds[i] / 100):
            sell_price = current_stock_price * (1 + sell_thresholds[i] / 100)  # 计算卖出价格
            profit = stock_count * (sell_price - total_cost / stock_count)  # 计算利润
            total_profit += profit  # 累加利润
            return total_profit

        # 更新当前股价为收盘价
        current_stock_price *= (1 + closing_change / 100)

        # 如果当天未卖出，股票数量不为0，且不是最后一天，则以当天收盘价买入股票
        if stock_count != 0 and i <= max_hold_day:
            total_cost += current_stock_price  # 更新总成本
            stock_count += 1  # 股票数量增加

    # 如果最后一天结束时还持有股票，以最后一天的收盘价卖出
    if stock_count > 0:
        total_profit += stock_count * (current_stock_price - total_cost / stock_count)

    return total_profit


def compute_more_than_one_day_held(file_path):
    """
    计算超过一天的平均持有天数
    :return:
    """
    # 先将file_path备份一份
    shutil.copyfile(file_path, file_path + '.bak')
    good_statistics = dict(read_json(file_path))
    for key, value in good_statistics.items():
        than_1_average_days_held = 0
        if value['ratio'] > 0:
            than_1_average_days_held = (value['ratio'] - 1 + value['average_days_held']) / value['ratio']
        value['than_1_average_days_held'] = than_1_average_days_held
    write_json(file_path, good_statistics)


def compute_1w_rate_day_held(file_path):
    """
    计算1w利润和超过一天的比例
    :return:
    """
    # 先将file_path备份一份
    shutil.copyfile(file_path, file_path + '.bak')
    good_statistics = dict(read_json(file_path))
    for k, v in good_statistics.items():
        if v['than_1_average_days_held'] != 0 and v['ratio'] != 0:
            v['1w_rate'] = v['average_1w_profit'] / v['than_1_average_days_held'] / v['ratio']
        else:
            v['1w_rate'] = v['average_1w_profit']
    statistics_1w = dict(sorted(good_statistics.items(), key=lambda x: x[1]['1w_rate'], reverse=True))
    write_json(file_path, statistics_1w)


def _fitness(statistic):
    """
    单个统计信息,适应度得分计算
    :param statistic:
    :return:
    """
    trade_count_threshold = 10
    # 适应度函数：计算该组合的得分
    if 'average_1w_profit' not in statistic:
        statistic["average_1w_profit"] = 0
    if statistic["trade_count"] == 0:
        return -10000
    trade_count_score = math.log(statistic["trade_count"])
    total_fitness = trade_count_score
    if statistic["three_befor_year_count"] >= trade_count_threshold or statistic["three_befor_year_rate"] >= 0.2:
        if statistic["ratio"] > 0:
            total_fitness = total_fitness / statistic["ratio"]
        else:
            total_fitness = total_fitness * 400
            total_fitness += 400
        total_fitness -= statistic['average_days_held']
        total_fitness += statistic['average_1w_profit'] / 4
        total_fitness -= statistic["ratio"] * 10 * statistic['than_1_average_days_held']

    return total_fitness


def compute_all_round_value(file_path):
    """
    计算综合得分
    :param file_path:
    :return:
    """
    # 先将file_path备份一份
    shutil.copyfile(file_path, file_path + '.bak')
    statistics = dict(read_json(file_path))
    for key, value in statistics.items():
        value['fitness'] = _fitness(value)
    # 按照fitness排序
    statistics_average_days_held = dict(sorted(statistics.items(), key=lambda x: x[1]['fitness'],
                                               reverse=True))
    write_json(file_path, statistics_average_days_held)


def filter_good_zuhe():
    """
    过滤出好的指标，并且全部再跑一次
    :return:
    """
    # compute_more_than_one_day_held('../back/gen/statistics_all.json')
    statistics = read_json('../final_zuhe/statistics_target_key.json')
    # 所有的指标都应该满足10次以上的交易
    statistics_new = {k: v for k, v in statistics.items() if
                      v['trade_count'] > 10 and (v['three_befor_year_count'] >= 10)}  # 100交易次数以上 13859
    # statistics_new = {k: v for k, v in statistics_new.items() if v['three_befor_year_count_thread_ratio'] <= 0.10 and v['three_befor_year_rate'] >= 0.2}
    good_ratio_keys = {k: v for k, v in statistics_new.items() if
                       v['ratio'] <= 0.1 and v['1w_rate'] >= 100 and v['average_1w_profit'] >= 100 and v[
                           'three_befor_year_count_thread_ratio'] <= 0.1}
    # good_ratio_keys_day = {k: v for k, v in statistics_new.items() if v['than_1_average_days_held'] <= 3 or v["average_1w_profit"] >= 100}
    # good_ratio_keys.update(good_ratio_keys_day)
    statistics_ratio = dict(sorted(good_ratio_keys.items(), key=lambda x: x[1]['ratio'], reverse=False))

    good_fitness_keys = {k: v for k, v in statistics_new.items() if
                         v['than_1_average_days_held'] <= 3.84 and v['average_1w_profit'] >= 100}
    statistics_fitness = dict(
        sorted(good_fitness_keys.items(), key=lambda x: x[1]['than_1_average_days_held'], reverse=True))

    good_1w_keys = {k: v for k, v in statistics_new.items() if v['than_1_average_days_held'] <= 2 or v["ratio"] <= 0.05}
    statistics_1w = dict(sorted(good_1w_keys.items(), key=lambda x: x[1]['average_1w_profit'], reverse=True))

    good_1w_rate_keys = {k: v for k, v in statistics_new.items() if
                         (v['ratio'] <= 0.1 and v['three_befor_year_count_thread_ratio'] <= 0.05) or v[
                             '1w_rate'] >= 200}
    statistics_1w_rate = dict(sorted(good_1w_rate_keys.items(), key=lambda x: x[1]['1w_rate'], reverse=True))

    # 将所有的指标都写入文件,去重
    statistics_all = dict()
    statistics_all.update(statistics_ratio)
    statistics_all.update(statistics_fitness)
    statistics_all.update(statistics_1w)
    statistics_all.update(statistics_1w_rate)
    write_json('../final_zuhe/good_statistics.json', statistics_all)


def temp():
    statistics = read_json('../back/gen/statistics_all_good_1w_not_in_fitness.json')
    for k, v in statistics.items():
        if v['than_1_average_days_held'] != 0:
            v['1w_rate'] = v['average_1w_profit'] / v['than_1_average_days_held']
        else:
            v['1w_rate'] = v['average_1w_profit']
    statistics_1w = dict(sorted(statistics.items(), key=lambda x: x[1]['1w_rate'], reverse=True))
    write_json('../back/gen/statistics_all_good_1w_rate.json', statistics_1w)


def back_select(file_path):
    """
    回测选中的文件
    :param file_path:
    :return:
    """
    all_df_list = []
    data = read_json(file_path)
    # 解析出时间
    date = file_path.split('/')[-1].split('.')[0].split('_')[-1]
    output_file_path = '../final_zuhe/back/' + date + 'back.csv'
    bad_output_file_path = '../final_zuhe/back/' + date + 'bad_back.json'
    bad_dict = dict()
    bad_count = 0
    date = datetime.strptime(date, '%Y-%m-%d')
    for stock_info in data:
        data_file_path = '../daily_data_exclude_new_can_buy/' + stock_info['stock_name'] + '.txt'
        daily_data = load_data(data_file_path)
        # 将date转换成datetime
        result_df = backtest_strategy_low_profit_target_data(daily_data, date)
        # 如果result_df中的Days Held大于1或者Profit小于0那么就将其添加到bad_dict中
        if result_df['Days Held'].values[0] > 1 or result_df['Profit'].values[0] < 0:
            bad_dict[stock_info['stock_name']] = stock_info
            bad_count += 1
        # 将result_df添加到all_df中
        all_df_list.append(result_df)
        # print(result_df)
    # 将bad_dict写入文件
    write_json(bad_output_file_path, bad_dict)
    if len(all_df_list) > 0:
        all_df = pd.concat(all_df_list, ignore_index=True)
        trade_count = all_df.shape[0]
        total_cost = all_df['total_cost'].sum()
        total_profit = all_df['Profit'].sum()
        average_1w_profit = total_profit * 10000 / total_cost
        all_df['this_pici_average_1w_profit'] = average_1w_profit
        ratio = bad_count / trade_count
        all_df['this_pici_ratio'] = ratio

        all_df['grouth_rate'] = 100 * all_df['Profit'] / all_df['total_cost']

        all_df.to_csv(output_file_path, index=False)
        print("date:{}, trade_count:{},bad_count:{},bad_ratio:{}, 1w_profit:{}".format(date, trade_count, bad_count, ratio, average_1w_profit))
        return all_df
    else:
        return None

def back_select_target_date(file_path):
    """
    回测选中的文件
    :param file_path:
    :return:
    """
    all_df_list = []
    data = pd.read_csv(file_path, dtype={'代码': str})
    # 解析出时间
    date = file_path.split('/')[-1].split('.')[0].split('_')[0]
    output_file_path = '../final_zuhe/back/' + date + 'target_back.csv'
    bad_output_file_path = '../final_zuhe/back/' + date + 'target_bad_back.json'
    bad_dict = dict()
    bad_count = 0
    date = datetime.strptime(date, '%Y-%m-%d')
    # 遍历data
    for index, row in data.iterrows():
        stock_name = "{}_{}".format(row['名称'], row['代码'])
        price = row['price']
        date = row['日期']
        # 将date转换成datetime,格式为'%Y-%m-%d'
        date = date.split(' ')[0]
        data_file_path = '../daily_data_exclude_new_can_buy/' + stock_name + '.txt'
        daily_data = load_data(data_file_path)
        daily_data = daily_data[daily_data['日期'] >= pd.Timestamp(date)]
        daily_data.reset_index(drop=True, inplace=True)

        result_df = backtest_strategy_low_profit_target_data_specia(daily_data, pd.Timestamp(date), price)
        # 如果result_df中的Days Held大于1或者Profit小于0那么就将其添加到bad_dict中
        if result_df['Days Held'].values[0] > 1 or result_df['Profit'].values[0] < 0:
            bad_dict[stock_name] = str(row)
            bad_count += 1
        # 将result_df添加到all_df中
        all_df_list.append(result_df)
        # print(result_df)
    # 将bad_dict写入文件
    write_json(bad_output_file_path, bad_dict)
    if len(all_df_list) > 0:
        all_df = pd.concat(all_df_list, ignore_index=True)
        trade_count = all_df.shape[0]
        total_cost = all_df['total_cost'].sum()
        total_profit = all_df['Profit'].sum()
        average_1w_profit = total_profit * 10000 / total_cost
        all_df['this_pici_average_1w_profit'] = average_1w_profit
        ratio = bad_count / trade_count
        all_df['this_pici_ratio'] = ratio

        all_df['grouth_rate'] = 100 * all_df['Profit'] / all_df['total_cost']

        all_df.to_csv(output_file_path, index=False)
        print("date:{}, trade_count:{},bad_count:{},bad_ratio:{}, 1w_profit:{}".format(date, trade_count, bad_count, ratio, average_1w_profit))
        return all_df
    else:
        return None

def back_range_select(start_time='2023-12-04', end_time='2023-12-17'):
    """
    回测一段时间内选中的文件
    :param start_time:
    :param end_time:
    :return:
    """
    # 生成时间在start_time和end_time之间的所有时间
    date_list = []
    start_time = datetime.strptime(start_time, '%Y-%m-%d')
    end_time = datetime.strptime(end_time, '%Y-%m-%d')
    all_back_result = []
    while start_time <= end_time:
        date_list.append(start_time)
        start_time += timedelta(days=1)
    for date in date_list:
        get_target_date_good_stocks_mul('../daily_data_exclude_new_can_buy', date,
                                        gen_signal_func=gen_full_all_basic_signal)
        date = date.strftime('%Y-%m-%d')
        file_path = '../final_zuhe/select/select_' + date + '.json'
        back_result = back_select(file_path)
        if back_result is not None:
            all_back_result.append(back_result)
    if len(all_back_result) > 0:
        all_df = pd.concat(all_back_result, ignore_index=True)
        all_df.to_csv('../final_zuhe/back/back.csv', index=False)
        return all_df

def back_range_select_real_time(start_time='2023-12-04', end_time='2023-12-17'):
    """
    回测一段时间内选中的文件
    :param start_time:
    :param end_time:
    :return:
    """

    out_put_file_path = '../final_zuhe/back/' + start_time + '_' + end_time + 'real_time_back.csv'
    start_time = datetime.strptime(start_time, '%Y-%m-%d')
    end_time = datetime.strptime(end_time, '%Y-%m-%d')
    date_list = []
    basic_data = load_data('../daily_data_exclude_new_can_buy/东方电子_000682.txt')
    # basic_data是dataFrame格式数据,获取basic_data中的所有日期
    all_date_list = basic_data['日期'].tolist()

    all_back_result = []
    while start_time <= end_time:
        # 如果start_time不在date_list中那么就跳过
        if start_time not in all_date_list:
            start_time += timedelta(days=1)
            continue
        date_list.append(start_time)
        start_time += timedelta(days=1)
    for date in date_list:
        buy_data = get_target_thread_min(date)
        if buy_data is not None:
            all_back_result.append(buy_data)
    if len(all_back_result) > 0:
        # 合并所有的back_result
        all_df = pd.concat(all_back_result, ignore_index=True)
        trade_count = all_df.shape[0]
        total_cost = all_df['total_cost'].sum()
        total_profit = all_df['Profit'].sum()
        average_1w_profit = total_profit * 10000 / total_cost
        all_df['this_pici_average_1w_profit'] = average_1w_profit

        result_df = all_df[all_df['Days Held'] > 1]
        result_df_size = result_df.shape[0]
        ratio = result_df_size / trade_count
        all_df['this_pici_ratio'] = ratio
        all_df.to_csv(out_put_file_path, index=False)
        return all_df

def back_range_RF_real_time(start_time='2023-12-04', end_time='2023-12-17'):
    """
    回测一段时间内选中的文件
    :param start_time:
    :param end_time:
    :return:
    """
    all_selected_samples_list = []
    # 生成时间在start_time和end_time之间的所有时间
    date_list = []
    out_put_file_path = '../final_zuhe/back/' + start_time + '_' + end_time + 'rf.csv'
    start_time = datetime.strptime(start_time, '%Y-%m-%d')
    end_time = datetime.strptime(end_time, '%Y-%m-%d')
    # 开始计时
    # get_good_combinations()
    good_statistics = read_json('../final_zuhe/good_statistics.json')
    good_keys = list(good_statistics.keys())
    index_data = save_index_data()
    # 只保留大于2023的index_data
    index_data = index_data[index_data['日期'] > pd.to_datetime('20230101')]
    basic_data = load_data('../daily_data_exclude_new_can_buy/东方电子_000682.txt')
    # basic_data是dataFrame格式数据,获取basic_data中的所有日期
    all_date_list = basic_data['日期'].tolist()

    all_rf_model_list = load_rf_model(MODEL_PATH)
    file_path = '../daily_data_exclude_new_can_buy/新坐标_603040.txt'
    while start_time <= end_time:
        # 如果start_time不在date_list中那么就跳过
        if start_time not in all_date_list:
            start_time += timedelta(days=1)
            continue
        date_list.append(start_time)
        start_time += timedelta(days=1)
    for date in date_list:
        all_selected_samples = get_RF_real_time_price(file_path, date, all_rf_model_list)
        # 判断all_selected_samples是否为空，是否为[]，如果是则跳过
        if all_selected_samples is not None and len(all_selected_samples) > 0:
            all_selected_samples_list.append(all_selected_samples)
    if len(all_selected_samples_list) > 0:
        pd.concat(all_selected_samples_list, ignore_index=True).to_csv(out_put_file_path, index=False)


def back_range_select_op(start_time='2023-12-04', end_time='2023-12-17'):
    """
    回测一段时间内选中的文件
    :param start_time:
    :param end_time:
    :return:
    """
    # 生成时间在start_time和end_time之间的所有时间
    date_list = []
    out_put_file_path = '../final_zuhe/back/' + start_time + '_' + end_time + 'back.csv'
    start_time = datetime.strptime(start_time, '%Y-%m-%d')
    end_time = datetime.strptime(end_time, '%Y-%m-%d')
    # 开始计时
    # get_good_combinations()
    good_statistics = read_json('../final_zuhe/good_statistics.json')
    good_keys = list(good_statistics.keys())
    index_data = save_index_data()
    # 只保留大于2023的index_data
    index_data = index_data[index_data['日期'] > pd.to_datetime('20230101')]
    index_data = gen_full_zhishu_basic_signal(index_data, True)
    basic_data = load_data('../daily_data_exclude_new_can_buy/东方电子_000682.txt')
    # basic_data是dataFrame格式数据,获取basic_data中的所有日期
    all_date_list = basic_data['日期'].tolist()

    all_back_result = []
    while start_time <= end_time:
        # 如果start_time不在date_list中那么就跳过
        if start_time not in all_date_list:
            start_time += timedelta(days=1)
            continue
        date_list.append(start_time)
        start_time += timedelta(days=1)
    for date in date_list:
        target_data = index_data[index_data['日期'] == pd.to_datetime(date)]
        # 获取target_data中值为False的列名
        false_columns = target_data.columns[(target_data == False).any()]
        # 将false_columns转换为frozenset
        false_columns = [frozenset([col]) for col in false_columns]
        final_combinations, zero_combination = filter_combinations_good(false_columns, good_keys)
        # 将final_combinations每个元素以:连接
        final_combinations = [':'.join(combination) for combination in final_combinations]
        # 过滤出key在final_combinations中的good_statistics
        final_statistics = {key: good_statistics[key] for key in final_combinations}
        print(str(date) + '的组合数量为：' + str(len(final_combinations)))
        get_target_date_good_stocks_mul_op('../daily_data_exclude_new_can_buy', date,
                                        gen_full_all_basic_signal, final_combinations, final_statistics, index_data)
        date = date.strftime('%Y-%m-%d')
        file_path = '../final_zuhe/select/select_' + date + '.json'
        back_result = back_select(file_path)
        if back_result is not None:
            all_back_result.append(back_result)
    if len(all_back_result) > 0:
        all_df = pd.concat(all_back_result, ignore_index=True)
        trade_count = all_df.shape[0]
        total_cost = all_df['total_cost'].sum()
        total_profit = all_df['Profit'].sum()
        average_1w_profit = total_profit * 10000 / total_cost
        all_df['this_pici_average_1w_profit'] = average_1w_profit

        result_df = all_df[all_df['Days Held'] > 1]
        result_df_size = result_df.shape[0]
        ratio = result_df_size / trade_count
        all_df['this_pici_ratio'] = ratio
        all_df.to_csv(out_put_file_path, index=False)
        return all_df

def save_and_analyse_stock_data_real_time_RF_thread(stock_data_list, exclude_code, target_date, need_skip=False):
    """
    拉取并且分析股票数据
    :param stock_data:
    :param exclude_code:
    :return:
    """
    all_result_list = []
    for stock_data in stock_data_list:
        code = stock_data['代码']
        name = stock_data['名称'].replace('*', '')
        if code not in exclude_code or need_skip:
            # code = '603985'
            # 开始计时
            start = time.time()
            price_data = get_price(code, '20230101', '20291021', period='daily')
            price_data['code'] = code
            # price_data不为空才保存
            if not price_data.empty:
                origin_data = fix_st(price_data, '../announcements/{}.json'.format(code))
                origin_data['代码'] = code
                origin_data['名称'] = name
                stock_data = origin_data.copy()

                if stock_data is None:
                    return None
                result_df = get_RF_real_time_price_thread(stock_data, target_date)
                if result_df is not None and not result_df.empty:
                    all_result_list.append(result_df)

            end = time.time()
            print('name: {}, code: {}, 耗时: {}'.format(name, code, end - start))
    all_result_df = pd.concat(all_result_list, ignore_index=True)
    return all_result_df

def get_good_price_RF():
    """
    扫描../final_zuhe/real_time下面所有数据，并加载模型进行预测
    :return:
    """
    file_list = []
    for root, dirs, files in os.walk('../final_zuhe/real_time'):
        for file in files:
            file_list.append(os.path.join(root, file))
    # 删除所有文件
    for file in file_list:
        os.remove(file)
    target_date = datetime.now().strftime('%Y-%m-%d')
    out_put_path = '../final_zuhe/select/{}real_time_good_price.txt'.format(target_date)
    start = time.time()
    all_rf_model_list = load_rf_model(MODEL_PATH)
    print('加载模型耗时：{}'.format(time.time() - start))
    while True:
        file_list = []
        # 获取../final_zuhe/real_time下面所有文件，保存到file_list中
        for root, dirs, files in os.walk('../final_zuhe/real_time'):
            for file in files:
                file_list.append(os.path.join(root, file))
        # 如果file_list为空，那么就休眠一段时间
        if len(file_list) == 0:
            print('当前时间：{} 没有新文件待处理'.format(datetime.now()))
            time.sleep(60)
            continue
        print('当前时间：{} 有{}个新文件待处理'.format(datetime.now(), len(file_list)))

        # 读取并合并所有文件
        all_data = pd.concat([pd.read_csv(file, dtype={'code': str}) for file in file_list])
        print('合并后的数据量：{}'.format(all_data.shape[0]))

        try:
            all_selected_samples = get_all_good_data_with_model_list(all_data, all_rf_model_list)
            if all_selected_samples is not None and not all_selected_samples.empty:
                new_selected_samples = all_selected_samples.groupby(['日期', '名称', '收盘', 'code']).size().reset_index(
                    name='选中数量')
                # 将计算得到的数量合并回原始DataFrame，以便我们可以基于数量筛选
                selected_samples_with_count = pd.merge(all_selected_samples, new_selected_samples,
                                                       on=['日期', '名称', '收盘', 'code'])

                # 筛选出数量大于等于2的条目
                all_selected_samples = selected_samples_with_count[selected_samples_with_count['选中数量'] >= 2]
                # 处理合并后的DataFrame
                grouped = all_selected_samples.groupby('code').agg(max_close=('收盘', 'max'), min_close=('收盘', 'min'), current_price=('current_price', 'min'))

                # 将结果保存到out_put_path
                with open(out_put_path, 'a') as f:
                    for code, row in grouped.iterrows():
                        f.write('{}, {}, {}, {}\n'.format(code, row['min_close'], row['max_close'], row['current_price']))
        except Exception as e:
            print('处理文件失败：{}'.format(e))

        # 删除原始文件
        for file in file_list:
            os.remove(file)

        print('文件处理完成')


def save_and_analyse_stock_data_real_time_RF(stock_data, exclude_code, target_date, all_rf_model_list, output_file_path, need_skip=False):
    """
    拉取并且分析股票数据
    :param stock_data:
    :param exclude_code:
    :return:
    """

    code = stock_data['代码']
    name = stock_data['名称'].replace('*', '')
    if code not in exclude_code or need_skip:
        # code = '603985'
        # 开始计时
        start = time.time()
        price_data = get_price(code, '20230101', '20291021', period='daily')
        price_data['code'] = code
        # price_data不为空才保存
        if not price_data.empty:
            origin_data = fix_st(price_data, '../announcements/{}.json'.format(code))
            stock_data = origin_data.copy()

            if stock_data is None:
                return None
            print('name: {}, code: {}下载完毕'.format(name, code))
            all_selected_samples = get_RF_real_time_price(stock_data, target_date, all_rf_model_list)
            # 如果all_selected_samples不空， 获取all_selected_samples中 收盘 的 最大和最小值
            if all_selected_samples and not all_selected_samples.empty:
                max_close = all_selected_samples['收盘'].max()
                min_close = all_selected_samples['收盘'].min()
                # 将code,max_close,min_close增量写入output_file_path
                with open(output_file_path, 'a') as f:
                    f.write(code + ',' + str(max_close) + ',' + str(min_close) + '\n')
                print('name: {}, code: {}分析完毕'.format(name, code))
        end = time.time()
        print('name: {}, code: {}, time: {}'.format(name, code, end - start))


def save_and_analyse_stock_data_real_time(stock_data, exclude_code, target_date, good_keys, good_statistics, output_file_path, index_data, need_skip=False, gen_signal_func=gen_full_all_basic_signal):
    """
    拉取并且分析股票数据
    :param stock_data:
    :param exclude_code:
    :return:
    """

    code = stock_data['代码']
    name = stock_data['名称'].replace('*', '')
    if code not in exclude_code or need_skip:
        # code = '603985'
        # 开始计时
        start = time.time()
        price_data = get_price(code, '20230101', '20291021', period='daily')
        price_data['code'] = code
        # price_data不为空才保存
        if not price_data.empty:
            origin_data = fix_st(price_data, '../announcements/{}.json'.format(code))
            stock_data = origin_data.copy()

            if stock_data is None:
                return None
            buy_data = get_threshold_close_target_date(target_date, stock_data)
            current_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
            if not buy_data.empty:
                # if buy_data.iloc[0]['Buy_Signal']:
                #     price = buy_data.iloc[0]['收盘']
                #     with open(output_file_path, 'a') as f:
                #         f.write(code + ',' + str(price) + '\n')
                #     # 获取当前时间,精确到分钟
                #     print('time:{} data:{}'.format(current_time, buy_data))
                if (not np.isnan(buy_data.iloc[0]['threshold_close_down'])):
                    price = buy_data.iloc[0]['threshold_close_down']
                    with open(output_file_path, 'a') as f:
                        f.write(code + ',' + str(price) + '\n')
                    print('time:{} data:{}'.format(current_time, buy_data))
        end = time.time()
        # print("{}耗时：{}".format(name, end - start))

def save_and_analyse_stock_data(stock_data, exclude_code, target_date, good_keys, good_statistics, output_file_path, index_data, need_skip=False, gen_signal_func=gen_full_all_basic_signal):
    """
    拉取并且分析股票数据
    :param stock_data:
    :param exclude_code:
    :return:
    """

    code = stock_data['代码']
    name = stock_data['名称'].replace('*', '')
    if code not in exclude_code or need_skip:
        # 开始计时
        start = time.time()
        price_data = get_price(code, '20230101', '20291021', period='daily')

        # price_data不为空才保存
        if not price_data.empty:
            origin_data = fix_st(price_data, '../announcements/{}.json'.format(code))
            stock_data = origin_data.copy()

            if stock_data is None:
                return None
            stock_data = gen_signal_func(stock_data)
            target_data = stock_data[stock_data['日期'] == pd.to_datetime(target_date)]
            # 将target_data和index_data合并，按照日期作为key
            target_data = pd.merge(target_data, index_data, on='日期', how='left')

            # 获取target_data中值为False的列名
            false_columns = target_data.columns[(target_data == False).any()]
            # 将false_columns转换为frozenset
            false_columns = [frozenset([col]) for col in false_columns]
            final_combinations, zero_combination = filter_combinations_good(false_columns, good_keys)

            if target_data.empty:
                return None
            price = target_data['收盘'].values[0]
            if target_data['涨跌幅'].values[0] <= -(target_data['Max_rate'].values[0] - 1.0 / price) or price < 3:
                return None
            satisfied_combinations = dict()
            # for good_key in final_combinations:
            #     good_key_str = ':'.join(good_key)
            #     signal_data = gen_signal(target_data, good_key)
            #     if signal_data['Buy_Signal'].values[0]:
            #         satisfied_combinations[good_key_str] = good_statistics[good_key_str]
            #         end = time.time()
            #         # 将code增量写入output_file_path，注意这个文件可能会被多个线程同时写入
            #         with open(output_file_path, 'a') as f:
            #             f.write(code + ',' + str(get_buy_price(price)) + '\n')
            #         print("{}耗时：{}".format(name, end - start))

            stock_data = origin_data.copy()
            stock_data['Buy_Signal'] = True
            stock_data = gen_daily_buy_signal_26(stock_data)
            signal_data = stock_data[stock_data['日期'] == pd.to_datetime(target_date)]
            if signal_data['Buy_Signal'].values[0]:
                end = time.time()
                # 将code增量写入output_file_path，注意这个文件可能会被多个线程同时写入
                with open(output_file_path, 'a') as f:
                    f.write(code + ',' + str(get_buy_price(price)) + '\n')
                print("{}耗时：{}".format(name, end - start))

def save_and_analyse_all_data_RF_real_time_thread(target_date):
    """
    多线程拉取最新数据并且分析出结果
    :return:
    """
    start = time.time()
    stock_data_df = ak.stock_zh_a_spot_em()
    exclude_code_set = set(ak.stock_kc_a_spot_em()['代码'].tolist())
    exclude_code_set.update(ak.stock_cy_a_spot_em()['代码'].tolist())
    need_code_set = {code for code in stock_data_df['代码'] if
                     code.startswith(('000', '002', '003', '001', '600', '601', '603', '605'))}
    new_exclude_code_set = set(stock_data_df['代码']) - need_code_set
    new_exclude_code_set.update(exclude_code_set)
    output_file_path = '../final_zuhe/select/select_RF_{}_real_time.csv'.format(target_date)
    output_result_path = '../final_zuhe/select/select_RF_result_{}_real_time.csv'.format(target_date)

    # 筛选出满足条件的股票数据
    filtered_stock_data_df = stock_data_df[~stock_data_df['代码'].isin(new_exclude_code_set)]

    # 保留filtered_stock_data_df的后100个数据
    filtered_stock_data_df = filtered_stock_data_df

    # 将满足条件的DataFrame分成100个子列表
    stock_data_lists = np.array_split(filtered_stock_data_df, 96)

    # 准备多进程执行的参数列表
    args = [(stock_data_list.to_dict('records'), new_exclude_code_set, target_date) for stock_data_list in
            stock_data_lists]

    # 使用多进程执行
    with Pool(processes=multiprocessing.cpu_count()) as pool:  # 根据你的机器性能调整进程数
        results = pool.starmap(save_and_analyse_stock_data_real_time_RF_thread, args)

    # 假设results是DataFrame的列表，我们将其合并
    all_result_df = pd.concat(results, ignore_index=True)

    # 保存合并后的DataFrame到文件
    all_result_df.to_csv(output_file_path, index=False)

    end = time.time()
    print(f'处理完成，耗时：{end - start}秒')

def save_and_analyse_all_data_RF_real_time(target_date):
    """
    多线程拉取最新数据并且分析出结果
    :return:
    """
    # 开始计时
    start = time.time()
    stock_data_df = ak.stock_zh_a_spot_em()
    all_code_set = set(stock_data_df['代码'].tolist())
    output_file_path = '../final_zuhe/select/select_RF_{}_real_time.txt'.format(target_date)
    all_rf_model_list = load_rf_model(MODEL_PATH)
    print('加载模型耗时：{}'.format(time.time() - start))

    exclude_code_set = set(ak.stock_kc_a_spot_em()['代码'].tolist())
    exclude_code_set.update(ak.stock_cy_a_spot_em()['代码'].tolist())

    need_code_set = {code for code in all_code_set if code.startswith(('000', '002', '003', '001', '600', '601', '603', '605'))}
    new_exclude_code_set = all_code_set - need_code_set
    new_exclude_code_set.update(exclude_code_set)


    # save_and_analyse_stock_data_real_time_RF(stock_data_df.iloc[0], new_exclude_code_set, target_date, all_rf_model_list,output_file_path, need_skip=True)
    for _, stock_data in stock_data_df.iterrows():
        save_and_analyse_stock_data_real_time_RF(stock_data, new_exclude_code_set, target_date, all_rf_model_list, output_file_path, need_skip=True)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    #     futures = [executor.submit(save_and_analyse_stock_data_real_time_RF, stock_data, new_exclude_code_set, target_date, all_rf_model_list, output_file_path) for _, stock_data in stock_data_df.iterrows()]
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             future.result()
    #         except Exception as e:
    #             logging.error(f"Error occurred: {e}")
    end = time.time()
    print("总耗时：{}".format(end - start))
    # exist_codes = []
    # total_count = 0
    # total_price = 0
    # if os.path.exists(output_file_path):
    #     with open(output_file_path, 'r') as lines:
    #         for line in lines:
    #             try:
    #                 stock_no, price = line.strip().split(',')
    #                 price = float(price)
    #                 if stock_no not in exist_codes:
    #                     total_price += price
    #                     total_count += 1
    #             except Exception as e:
    #                 print(e)
    #     print("总数：{}，总价：{}".format(total_count, total_price))

def save_and_analyse_all_data_mul_real_time(target_date):
    """
    多线程拉取最新数据并且分析出结果
    :return:
    """
    # 开始计时
    start = time.time()
    good_statistics = read_json('../final_zuhe/good_statistics.json')
    good_keys = list(good_statistics.keys())
    stock_data_df = ak.stock_zh_a_spot_em()
    all_code_set = set(stock_data_df['代码'].tolist())
    output_file_path = '../final_zuhe/select/select_{}_real_time.txt'.format(target_date)
    # 如果output_file_path存在，先删除
    # if os.path.exists(output_file_path):
    #     os.remove(output_file_path)

    exclude_code_set = set(ak.stock_kc_a_spot_em()['代码'].tolist())
    exclude_code_set.update(ak.stock_cy_a_spot_em()['代码'].tolist())

    need_code_set = {code for code in all_code_set if code.startswith(('000', '002', '003', '001', '600', '601', '603', '605'))}
    new_exclude_code_set = all_code_set - need_code_set
    new_exclude_code_set.update(exclude_code_set)
    index_data = save_index_data()
    # 只保留index_data中的日期大于2023年的数据
    index_data = index_data[index_data['日期'] > pd.to_datetime('20230101')]
    # index_data是dataFrame类型的数据,如果index_data中不存在日期为target_date的数据，新增一行数据
    if index_data[index_data['日期'] == pd.to_datetime(target_date)].empty:
        new_row = pd.DataFrame({'日期': [pd.to_datetime(target_date)]})
        index_data = pd.concat([index_data, new_row], ignore_index=True)

    index_data = gen_full_zhishu_basic_signal(index_data, True)

    # save_and_analyse_stock_data_real_time(stock_data_df.iloc[0], new_exclude_code_set, target_date, good_keys, good_statistics, output_file_path, index_data, need_skip=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(save_and_analyse_stock_data_real_time, stock_data, new_exclude_code_set, target_date, good_keys, good_statistics, output_file_path, index_data) for _, stock_data in stock_data_df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error occurred: {e}")
    end = time.time()
    print("总耗时：{}".format(end - start))
    exist_codes = []
    total_count = 0
    total_price = 0
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as lines:
            for line in lines:
                try:
                    stock_no, price = line.strip().split(',')
                    price = float(price)
                    if stock_no not in exist_codes:
                        total_price += price
                        total_count += 1
                except Exception as e:
                    print(e)
        print("总数：{}，总价：{}".format(total_count, total_price))

def save_and_analyse_all_data_mul(target_date):
    """
    多线程拉取最新数据并且分析出结果
    :return:
    """
    # 开始计时
    start = time.time()
    good_statistics = read_json('../final_zuhe/good_statistics.json')
    good_keys = list(good_statistics.keys())
    stock_data_df = ak.stock_zh_a_spot_em()
    all_code_set = set(stock_data_df['代码'].tolist())
    output_file_path = '../final_zuhe/select/select_{}.txt'.format(target_date)
    # 如果output_file_path存在，先删除
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    exclude_code_set = set(ak.stock_kc_a_spot_em()['代码'].tolist())
    exclude_code_set.update(ak.stock_cy_a_spot_em()['代码'].tolist())

    need_code_set = {code for code in all_code_set if code.startswith(('000', '002', '003', '001', '600', '601', '603', '605'))}
    new_exclude_code_set = all_code_set - need_code_set
    new_exclude_code_set.update(exclude_code_set)
    index_data = save_index_data()
    # 只保留index_data中的日期大于2023年的数据
    index_data = index_data[index_data['日期'] > pd.to_datetime('20230101')]
    # index_data是dataFrame类型的数据,如果index_data中不存在日期为target_date的数据，新增一行数据
    if index_data[index_data['日期'] == pd.to_datetime(target_date)].empty:
        new_row = pd.DataFrame({'日期': [pd.to_datetime(target_date)]})
        index_data = pd.concat([index_data, new_row], ignore_index=True)

    index_data = gen_full_zhishu_basic_signal(index_data, True)

    # save_and_analyse_stock_data(stock_data_df.iloc[0], new_exclude_code_set, target_date, good_keys, good_statistics, output_file_path, index_data, need_skip=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(save_and_analyse_stock_data, stock_data, new_exclude_code_set, target_date, good_keys, good_statistics, output_file_path, index_data) for _, stock_data in stock_data_df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error occurred: {e}")
    end = time.time()
    print("总耗时：{}".format(end - start))
    exist_codes = []
    total_count = 0
    total_price = 0
    with open(output_file_path, 'r') as lines:
        for line in lines:
            try:
                stock_no, price = line.strip().split(',')
                price = float(price)
                if stock_no not in exist_codes:
                    total_price += price
                    total_count += 1
            except Exception as e:
                print(e)
    print("总数：{}，总价：{}".format(total_count, total_price))

def process_file_all(fullname, out_put_file_path, new_index_data_df):
    origin_data_df = load_full_data(fullname)
    origin_data_df = gen_full_all_basic_signal(origin_data_df)
    origin_data_df['Buy_Signal'] = True
    back_result_df = backtest_strategy_low_profit(origin_data_df)
    origin_data_df = origin_data_df.merge(back_result_df, on=['日期', '名称'], how='left')
    # 将new_index_data_df中的数据合并到origin_data_df中，以日期为key
    # origin_data_df = origin_data_df.merge(new_index_data_df, on=['日期'], how='left')
    output_filename = os.path.join(out_put_file_path, os.path.basename(fullname))
    # 找出origin_data_df中全为
    false_columns = origin_data_df.columns[(origin_data_df == False).all()]
    false_columns_output_filename = os.path.join('{}_false'.format(out_put_file_path), '{}false_columns.txt'.format(os.path.basename(fullname)))
    # origin_data_df = clear_other_clo(origin_data_df)
    # 过滤掉origin_data_df['涨跌幅']大于(origin_data_df['Max_rate'] 1/origin_data_df['收盘'])的数据
    origin_data_df = origin_data_df[origin_data_df['涨跌幅'] <= (origin_data_df['Max_rate'] - 1.0/ origin_data_df['收盘'])]
    #将false_columns写入文件
    with open(false_columns_output_filename, 'w') as f:
        f.write('\n'.join(false_columns.tolist()))
    origin_data_df['Buy Date'] = pd.to_datetime(origin_data_df['Buy Date'])
    origin_data_df.to_csv(output_filename, index=False)

def process_file_zhibiao(fullname, out_put_file_path):
    origin_data_df = load_full_data(fullname)
    origin_data_df['Buy_Signal'] = True
    back_result_df = backtest_strategy_low_profit(origin_data_df)
    origin_data_df = origin_data_df.merge(back_result_df, on=['日期'], how='left')
    output_filename = os.path.join(out_put_file_path, os.path.basename(fullname))
    origin_data_df.to_csv(output_filename, index=False)

def gen_all_back():
    """
    为所有数据生成回测结果并写入文件，这样后续就不用每次进行回测了，大大节约时间
    :return:
    """
    save_all_data_mul()
    file_path = '../daily_data_exclude_new_can_buy/'
    out_put_file_path = '../daily_data_exclude_new_can_buy_with_back'
    index_data = save_index_data()
    new_index_data = gen_full_zhishu_basic_signal(index_data, True)
    with Pool() as pool:
        for root, ds, fs in os.walk(file_path):
            for f in fs:
                fullname = os.path.join(root, f)
                # process_file_all(fullname, out_put_file_path, new_index_data)
                pool.apply_async(process_file_all, args=(fullname, out_put_file_path, new_index_data))
        pool.close()
        pool.join()

def gen_all_zhibiao():
    """
    为所有数据生成所有指标（天，周，月，上证）并写入文件，这样后续就不用每次新生成，大大节约时间
    :return:
    """
    file_path = '../daily_data_exclude_new_can_buy_with_back/'
    out_put_file_path = '../daily_data_exclude_new_can_buy_with_all_zhibiao'
    with Pool() as pool:
        for root, ds, fs in os.walk(file_path):
            for f in fs:
                fullname = os.path.join(root, f)
                pool.apply_async(process_file_zhibiao, args=(fullname, out_put_file_path))

        pool.close()
        pool.join()



def process_file_target_date(fullname, date, gen_daily_buy_signal_26):
    """
    处理单个文件
    """
    data = load_data(fullname)
    buy_data = get_threshold_close_target_date(date, data, gen_daily_buy_signal_26)
    if not buy_data.empty:
        if (not np.isnan(buy_data.iloc[0]['threshold_close_up'])) or (not np.isnan(buy_data.iloc[0]['threshold_close_down'])):
            print(buy_data)
            return buy_data
    return None

def process_file_target_date_min(file_path, date, gen_daily_buy_signal_26):
    """
    处理单个文件
    """
    start_time = time.time()
    buy_data_list = []
    fullname, min_fullname = file_path
    data = load_data(fullname)
    min_data = load_data(min_fullname)
    # 将min_data中的日期转换为datetime类型
    min_data['日期'] = pd.to_datetime(min_data['日期'])
    # 找到min_data中日期大于date且小于date+1的数据
    min_data = min_data[(min_data['日期'] > pd.to_datetime(date)) & (min_data['日期'] < pd.to_datetime(date) + timedelta(days=1))]
    target_data = data[data['日期'] == date]
    # # 将min_data按照日期逆序排列
    # min_data = min_data.sort_values(by='日期', ascending=False)
    # 获取target_data的index
    target_data_index = target_data.index
    # 遍历min_data
    for index, row in min_data.iterrows():
        # 将row的开盘,收盘,最高,最低,成交量,成交额赋值给data中target_data_index的行
        data.loc[target_data_index, '开盘'] = row['开盘']
        data.loc[target_data_index, '收盘'] = row['收盘']
        data.loc[target_data_index, '最高'] = row['最高']
        data.loc[target_data_index, '最低'] = row['最低']
        data.loc[target_data_index, '成交量'] = row['成交量']
        data.loc[target_data_index, '成交额'] = row['成交额']
        # 计算step , step = 开盘/1000 ,保留小数点后两位,向上取整
        step = math.ceil(row['开盘'] / 1000 * 100) / 100
        buy_data = get_threshold_close_target_date(date, data, gen_daily_buy_signal_26, step=step)
        if not buy_data.empty:
            buy_data.at[buy_data.index[0], '日期'] = row['日期']
            if buy_data.iloc[0]['Buy_Signal']:
                # 复制一份buy_data
                buy_data = buy_data.copy()
                buy_data.at[buy_data.index[0], 'price'] = row['收盘']
                print(buy_data)
                buy_data_list.append(buy_data)
                if (not np.isnan(buy_data.iloc[0]['threshold_close_down']) and buy_data.iloc[0]['threshold_close_down'] > row['后续最低']):
                    # 复制一份buy_data
                    buy_data = buy_data.copy()
                    buy_data.at[buy_data.index[0], 'price'] = buy_data.iloc[0]['threshold_close_down']
                    print(buy_data)
                    buy_data_list.append(buy_data)
            else:
                if (not np.isnan(buy_data.iloc[0]['threshold_close_down']) and buy_data.iloc[0]['threshold_close_down'] > row['后续最低']):
                    # 复制一份buy_data
                    buy_data = buy_data.copy()
                    buy_data.at[buy_data.index[0], 'price'] = buy_data.iloc[0]['threshold_close_down']
                    print(buy_data)
                    buy_data_list.append(buy_data)
                if (not np.isnan(buy_data.iloc[0]['threshold_close_up']) and buy_data.iloc[0]['收盘'] > row['后续最低']):
                    # 复制一份buy_data
                    buy_data = buy_data.copy()
                    buy_data.at[buy_data.index[0], 'price'] = row['收盘']
                    print(buy_data)
                    buy_data_list.append(buy_data)
    # 如果buy_data_list不为空,则将buy_data_list中的数据合并
    if buy_data_list:
        return pd.concat(buy_data_list)
    end_time = time.time()
    print('处理{}耗时:{}'.format(fullname, end_time - start_time))
    return None

def get_target_thread(date='2024-01-22'):
    """
    获取指定时间满足好指标的每个股票的阈值
    :return:
    """
    file_path = '../daily_data_exclude_new_can_buy/'
    all_files = []

    # 获取所有文件路径
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            fullname = os.path.join(root, f)
            all_files.append(fullname)

    # 创建进程池
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    # 处理每个文件
    results = [pool.apply_async(process_file_target_date, args=(f, date, gen_daily_buy_signal_26)) for f in all_files]

    # 收集结果
    all_data = [res.get() for res in results if res.get() is not None]

    # 关闭进程池
    pool.close()
    pool.join()
    # 将结果写入文件
    output_filename = os.path.join('../final_zuhe/select/', '{}_target_thread.csv'.format(date))
    pd.concat(all_data).to_csv(output_filename, index=False)
    back_select_target_date(output_filename)
    return all_data

def contains_english_letter(s):
    # 使用正则表达式检查字符串中是否包含英文字母
    return bool(re.search(r'[a-zA-Z]', s))

def get_target_thread_min(date='2024-01-22'):
    """
    获取指定时间满足好指标的每个股票的阈值
    :return:
    """
    file_path = '../daily_data_exclude_new_can_buy/'
    min_file_path = '../min_data_exclude_new_can_buy/'
    all_files = []
    start_time = time.time()
    # 获取所有文件路径
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            fullname = os.path.join(root, f)
            # 如果f中包含字母，则跳过
            if contains_english_letter(f.split('.')[0]):
                continue
            min_fullname = os.path.join(min_file_path, f)
            if os.path.exists(min_fullname):
                all_files.append([fullname, min_fullname])

    # 创建进程池
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    # 处理每个文件
    results = [pool.apply_async(process_file_target_date_min, args=(f, date, gen_daily_buy_signal_26)) for f in all_files]

    # 收集结果
    all_data = [res.get() for res in results if res.get() is not None]

    # 关闭进程池
    pool.close()
    pool.join()
    if all_data is not None:
        all_df = pd.concat(all_data)
    else:
        return None
    # 将结果写入文件
    # 将date转换成str
    date = str(date)
    output_filename = os.path.join('../final_zuhe/select/', '{}_target_thread.csv'.format(date.split(' ')[0]))
    all_df.to_csv(output_filename, index=False)
    total_count = len(all_df)
    total_price = all_df['收盘'].sum()
    print('总数量:{} 总价值：{}w'.format(total_count, round(total_price / 100, 2)))
    end_time = time.time()
    print('get_target_date_good_stocks time cost: {}'.format(end_time - start_time))
    back_select_target_date(output_filename)
    return all_df

def get_target_date(date='2024-01-17'):
    """
    获取指定时间满足好指标的每个股票的阈值
    :return:
    """
    all_data = []
    file_path = '../daily_data_exclude_new_can_buy/'
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            fullname = os.path.join(root, f)
            data = load_data(fullname)
            buy_data = get_threshold_close_target_date(date, data, gen_daily_buy_signal_26)
            if not buy_data.empty:
                # 如果buy_data的第一行数据的threshold_close_up不为Nan，说明满足条件
                if (not np.isnan(buy_data.iloc[0]['threshold_close_up'])) or (not np.isnan(buy_data.iloc[0]['threshold_close_down'])):
                    all_data.append(buy_data)
                    print(buy_data)
    return all_data


def load_file_chunk(file_chunk):
    """
    加载文件块的数据
    """
    chunk_data = [load_data(fname) for fname in file_chunk]
    return pd.concat(chunk_data)

def get_all_data_perfomance():
    """
    加载所有的数据，并且获取每一天的表现情况
    """
    start_time = time.time()
    file_path = '../daily_data_exclude_new_can_buy_with_back'

    # 获取所有文件名
    all_files = [os.path.join(root, f) for root, ds, fs in os.walk(file_path) for f in fs]

    # 使用多进程分块加载数据
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)
    file_chunks = [all_files[i::cpu_count] for i in range(cpu_count)]
    chunk_dfs = pool.map(load_file_chunk, file_chunks)
    pool.close()
    pool.join()

    # 合并数据
    all_data_df = pd.concat(chunk_dfs)

    # 确保 'Buy Date' 列是 datetime 类型
    all_data_df['Buy Date'] = pd.to_datetime(all_data_df['Buy Date'], errors='coerce')

    # 检查转换是否成功
    if all_data_df['Buy Date'].dtype != '<M8[ns]':
        raise ValueError("Buy Date column could not be converted to datetime")

    merge_time = time.time()
    print('合并耗时：', merge_time - start_time)

    # 按照'Buy Date'进行分组
    grouped_data = all_data_df.groupby(all_data_df['Buy Date'].dt.date)

    # 分析每组数据
    result_dict = {}
    for date, group in grouped_data:
        date_str = date.strftime('%Y-%m-%d')  # 将date对象转换为字符串
        result_dict[date_str] = process_results_with_every_period(group, threshold_day=1)

    # 将结果写入文件
    output_filename = os.path.join('../final_zuhe/select/', 'all_data_perfomance.json')
    write_json(output_filename, result_dict)

    return result_dict

def load_all_data():
    """
    加载所有数据
    """
    start_time = time.time()
    file_path = '../daily_data_exclude_new_can_buy_with_back'

    # 获取所有文件名
    all_files = [os.path.join(root, f) for root, ds, fs in os.walk(file_path) for f in fs]

    # 使用多进程分块加载数据
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)
    file_chunks = [all_files[i::cpu_count] for i in range(cpu_count)]
    chunk_dfs = pool.map(load_file_chunk, file_chunks)
    pool.close()
    pool.join()

    # 合并数据
    all_data_df = pd.concat(chunk_dfs)
    merge_time = time.time()
    print('合并耗时：', merge_time - start_time)

    # 转换日期格式
    all_data_df['Buy Date'] = pd.to_datetime(all_data_df['Buy Date'])

    # 将数据均匀分成100份并写入文件
    num_splits = 1
    folder_path = '../daily_all_2024'
    os.makedirs(folder_path, exist_ok=True)  # 创建文件夹（如果不存在）
    split_size = math.ceil(len(all_data_df) / num_splits)

    for i in range(num_splits):
        split_data = all_data_df.iloc[i * split_size:(i + 1) * split_size]
        split_data.to_csv(os.path.join(folder_path, f'{i + 1}.txt'), index=False)

    end_time = time.time()
    print('总耗时：', end_time - start_time)

def load_all_data_performance(ratio_threshold=0.0):
    """
    加载所有数据，并根据给定的比率阈值筛选数据
    """
    start_time = time.time()
    file_path = '../daily_data_exclude_new_can_buy_with_back'
    output_filename = os.path.join('../final_zuhe/select/', 'all_data_perfomance.json')
    data = read_json(output_filename)  # 假设read_json是一个已定义的函数，用于读取JSON数据
    bad_data_list = []
    for key, value in data.items():
        if value['all']['ratio'] >= ratio_threshold:
            bad_data_list.append(key)

    # 获取所有文件名
    all_files = [os.path.join(root, f) for root, ds, fs in os.walk(file_path) for f in fs]

    # 使用多进程分块加载数据
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)
    file_chunks = [all_files[i::cpu_count] for i in range(cpu_count)]
    chunk_dfs = pool.map(load_file_chunk, file_chunks)  # 假设load_file_chunk是一个已定义的函数，用于加载文件块
    pool.close()
    pool.join()

    # 合并数据
    all_data_df = pd.concat(chunk_dfs)
    merge_time = time.time()
    print('合并耗时：', merge_time - start_time)

    bad_data_df = all_data_df
    print('bad_data_df长度：', len(bad_data_df))
    print('all_data_df长度：', len(all_data_df))
    all_data_df = bad_data_df
    all_data_df['Buy Date'] = pd.to_datetime(all_data_df['Buy Date'])

    # 将数据均匀分成100份并写入文件
    num_splits = 1  # 假设你想分成100份，根据你的描述进行调整
    folder_path = f'../train_data/daily_all_{num_splits}_bad_{ratio_threshold}'
    os.makedirs(folder_path, exist_ok=True)  # 创建文件夹（如果不存在）
    split_size = math.ceil(len(all_data_df) / num_splits)

    for i in range(num_splits):
        split_data = all_data_df.iloc[i * split_size:(i + 1) * split_size]
        split_data.to_csv(os.path.join(folder_path, f'{i + 1}.txt'), index=False)

    end_time = time.time()
    print('总耗时：', end_time - start_time)

def get_common_line():
    file_path = '../model/selected_features.txt'
    file_path1 = '../model/selected_features_0.5.txt'
    file_set = set()
    file_1_set = set()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            file_set.add(line.strip())
    with open(file_path1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            file_1_set.add(line.strip())
    # 输出两个文件的数量和交集数量
    print('file_set长度：', len(file_set))
    print('file_1_set长度：', len(file_1_set))
    print('交集长度：', len(file_set & file_1_set))
    return file_set & file_1_set

def get_RF_real_time_price_thread(file_path, target_date):
    """
    获取能够满足模型的价格数据
    :return:
    """
    # 如果file_path是DataFrame类型，则直接使用
    if isinstance(file_path, pd.DataFrame):
        origin_data = file_path
    else:
        origin_data = pd.read_csv(file_path, low_memory=False)
        name, code = parse_filename(file_path)
        origin_data['名称'] = name
        origin_data['代码'] = code
    name = origin_data.iloc[0]['名称']
    code = origin_data.iloc[0]['代码']
    # 生成output文件名，由名称和代码组成
    output_file_name = f'../final_zuhe/real_time/{name}_{code}_real_time_price.csv'
    result_list = []
    # 将origin_data按照日期转换为时间序列
    origin_data['日期'] = pd.to_datetime(origin_data['日期'])
    temp_origin_data = origin_data[(origin_data['日期'] < pd.to_datetime(target_date))]
    target_data = origin_data[(origin_data['日期'] == pd.to_datetime(target_date))]
    # 截取temp_origin_data的后100条数据
    temp_origin_data = temp_origin_data.iloc[-100:]
    if target_data.empty:
        return None
    # 获取temp_origin_data的最后一行数据的日期
    last_price = temp_origin_data.iloc[-1]['收盘']
    current_price = target_data.iloc[0]['收盘']
    max_rate = temp_origin_data.iloc[-1]['Max_rate']
    highest_price = round(last_price * (100 + max_rate) / 100, 2)
    lowest_price = round(last_price * (100 - max_rate) / 100, 2)
    # current_highest_price = round(current_price * (100 + max_rate / 5) / 100, 2)
    # current_lowest_price = round(current_price * (100 - max_rate / 5) / 100, 2)
    current_highest_price = current_price + 0.02
    current_lowest_price = current_price - 0.02
    highest_price = min(highest_price, current_highest_price)
    lowest_price = max(lowest_price, current_lowest_price)
    # 计算步长step，step = (highest_price - lowest_price) / 1000，向上取整保留两位小数
    step = math.ceil((highest_price - lowest_price) / 10) / 100
    if step < 0.01:
        step = 0.01


    # 遍历last_price到highest_price和lowest_price之间的数据，步长最小为0.01，并且数据个数最多为1000个
    price_list = np.arange(lowest_price, highest_price, step)
    for price in price_list:
        temp_origin_data_copy = temp_origin_data.copy()
        temp_target_data = target_data.copy()
        temp_target_data['收盘'] = price
        if (price > temp_target_data['最高']).any():
            temp_target_data['最高'] = price
        if (price < temp_target_data['最低']).any():
            temp_target_data['最低'] = price
        temp_target_data['涨跌幅'] = round((price - last_price) / last_price * 100, 2)
        temp_target_data['振幅'] = round((temp_target_data['最高'] - temp_target_data['最低']) / last_price * 100, 2)
        # 将temp_target_data增加到temp_origin_data_copy中,temp_target_data是DataFrame类型的
        temp_origin_data_copy = pd.concat([temp_origin_data_copy, temp_target_data], axis=0)
        full_temp_origin_data_copy = gen_full_all_basic_signal(temp_origin_data_copy)
        result = full_temp_origin_data_copy[(full_temp_origin_data_copy['日期'] == pd.to_datetime(target_date))]
        result['current_price'] = current_price
        result_list.append(result)
    if len(result_list) == 0:
        return None
    result_df = pd.concat(result_list, axis=0).reset_index(drop=True)
    # 将result_df写入文件
    result_df.to_csv(output_file_name, index=False)

    return result_df

def get_RF_real_time_price(file_path, target_date, all_rf_model_list):
    """
    获取能够满足模型的价格数据
    :return:
    """
    # 如果file_path是DataFrame类型，则直接使用
    if isinstance(file_path, pd.DataFrame):
        origin_data = file_path
    else:
        origin_data = pd.read_csv(file_path, low_memory=False)
        name, code = parse_filename(file_path)
        origin_data['名称'] = name
        origin_data['代码'] = code
    name = origin_data.iloc[0]['名称']
    code = origin_data.iloc[0]['代码']
    # 生成output文件名，由名称和代码组成
    output_file_name = f'../final_zuhe/real_time/{name}_{code}_real_time_price.csv'
    result_list = []
    # 将origin_data按照日期转换为时间序列
    origin_data['日期'] = pd.to_datetime(origin_data['日期'])
    temp_origin_data = origin_data[(origin_data['日期'] < pd.to_datetime(target_date))]
    target_data = origin_data[(origin_data['日期'] == pd.to_datetime(target_date))]
    # 截取temp_origin_data的后100条数据
    temp_origin_data = temp_origin_data.iloc[-100:]
    if target_data.empty:
        return None
    # 获取temp_origin_data的最后一行数据的日期
    last_price = temp_origin_data.iloc[-1]['收盘']
    max_rate = temp_origin_data.iloc[-1]['Max_rate']
    highest_price = round(last_price * (100 + max_rate) / 100, 2)
    lowest_price = round(last_price * (100 - max_rate) / 100, 2)
    # 计算步长step，step = (highest_price - lowest_price) / 1000，向上取整保留两位小数
    step = math.ceil((highest_price - lowest_price) / 10) / 100
    if step < 0.01:
        step = 0.01


    # 遍历last_price到highest_price和lowest_price之间的数据，步长最小为0.01，并且数据个数最多为1000个
    price_list = np.arange(lowest_price, highest_price, step)
    for price in price_list:
        temp_origin_data_copy = temp_origin_data.copy()
        temp_target_data = target_data.copy()
        temp_target_data['收盘'] = price
        if (price > temp_target_data['最高']).any():
            temp_target_data['最高'] = price
        if (price < temp_target_data['最低']).any():
            temp_target_data['最低'] = price
        temp_target_data['涨跌幅'] = round((price - last_price) / last_price * 100, 2)
        temp_target_data['振幅'] = round((temp_target_data['最高'] - temp_target_data['最低']) / last_price * 100, 2)
        # 将temp_target_data增加到temp_origin_data_copy中,temp_target_data是DataFrame类型的
        temp_origin_data_copy = pd.concat([temp_origin_data_copy, temp_target_data], axis=0)
        full_temp_origin_data_copy = gen_full_all_basic_signal(temp_origin_data_copy)
        result = full_temp_origin_data_copy[(full_temp_origin_data_copy['日期'] == pd.to_datetime(target_date))]
        result_list.append(result)
    if len(result_list) == 0:
        return None
    result_df = pd.concat(result_list, axis=0).reset_index(drop=True)
    # 将result_df写入文件
    result_df.to_csv(output_file_name, index=False)
    all_selected_samples = get_all_good_data_with_model_list(result_df, all_rf_model_list)
    return all_selected_samples

def save_and_analyse_all_data_mul_real_time_RF(target_date):
    process = multiprocessing.Process(target=save_and_analyse_all_data_RF_real_time_thread, args=(target_date,))
    process.start()
    get_good_price_RF()

if __name__ == '__main__':
    # file_path = '../final_zuhe/statistics_target_key.json'
    # # file_path = '../back/gen/statistics_all.json'
    # compute_more_than_one_day_held(file_path)
    # compute_all_round_value(file_path)
    # compute_1w_rate_day_held(file_path)
    # filter_good_zuhe()


    # get_good_combinations()
    # save_and_analyse_all_data_mul('2024-01-22')
    # save_and_analyse_all_data_mul_real_time('2024-02-05')
    # get_newest_stock()
    # back_range_select_op(start_time='2023-10-01', end_time='2023-12-01')
    # back_range_select_op(start_time='2024-01-12', end_time='2024-01-19')
    # back_range_select_op(start_time='2024-01-17', end_time='2024-01-19')
    # back_range_select_op(start_time='2024-01-22', end_time='2024-01-22')
    # print(good_data)

    # load_all_data()
    # get_all_data_perfomance()
    # gen_all_back()
    # load_all_data_performance()
    data = pd.read_csv('../min_data_exclude_new_can_buy/恒基达鑫_002492.txt')
    print(data)
    # save_and_analyse_all_data_mul_real_time_RF('2024-02-23')

    # common_data = get_common_line()
    # print(common_data)

    #     for k, v in value.items():
    #         v['ratio'] = round(v['size_of_result_df'] / v['trade_count'], 4)
    #         v['average_1w_profit'] = round(v['total_profit'] / v['total_cost'] * 10000, 4)
    # write_json(output_filename, data)

    # statistics_zuhe_gen_both_single_every_period('../back/gen/single')

    # data = process_file_target_date_min(['../daily_data_exclude_new_can_buy/中际联合_605305.txt', '../min_data_exclude_new_can_buy/中际联合_605305.txt'], '2024-01-17', gen_daily_buy_signal_26)
    # print(data)
    # back_select_target_date('../final_zuhe/select/2024-01-03_target_thread.csv')
    # date = '2024-01-26'
    # buy_data = get_target_thread(date)
    # buy_data = get_target_thread_min(date)
    back_range_select_real_time(start_time='2024-02-20', end_time='2024-02-20')
    # print(buy_data)
    # back_range_RF_real_time(start_time='2024-01-03', end_time='2024-02-05')


    # statistics_zuhe_gen_both_single_every_period('../back/gen/single')


    # 读取../back/complex/all_df.csv

    # all_df = pd.read_csv('../back/complex/all_df.csv')
    # # 获取all_df中所有的日期,并且去重
    # all_date = all_df['日期'].unique()
    # # 获取每个日期对应的数据的数量
    # all_date_count = all_df.groupby('日期').size()
    # print(all_date_count)
    # data = pd.read_csv('../daily_data_exclude_new_can_buy_with_back/东方电子_000682.txt', low_memory=False)


    # count_min_profit_rate('../daily_data_exclude_new_can_buy', '../back/complex/all_df.csv', gen_signal_func=mix)
    # back_all_stock('../daily_data_exclude_new_can_buy/', '../back/complex', gen_signal_func=gen_daily_buy_signal_31, backtest_func=backtest_strategy_low_profit)
    #
    # strategy('../1_min_data_exclude_new_can_buy/蓝天燃气_605368.txt', gen_signal_func=gen_daily_buy_signal_26, backtest_func=backtest_strategy_low_profit)

    # # statistics = read_json('../back/statistics_target_key.json')
    # statistics = read_json('../back/gen/statistics_all.json') # 大小 2126501
    # # statistics = read_json('../final_zuhe/statistics_target_key.json')
    # # statistics = read_json('../back/gen/statistics_target_key.json')
    # # temp_data = read_json('../back/gen/zuhe/贵绳股份.json')
    # # temp_data = read_json('../final_zuhe/zuhe/贵绳股份.json')
    # # good_statistics = get_good_combinations()
    # # sublist_list = read_json('../back/gen/sublist.json') #大小 55044
    # statistics = dict(sorted(statistics.items(), key=lambda x: (-x[1]['ratio'], x[1]['trade_count']), reverse=True))
    # # sublist_list中的元素也是list，帮我对sublist_list进行去重
    # # 将statistics中trade_count大于100的筛选出来，并且按照average_profit降序排序
    # # statistics = {k: v for k, v in statistics.items() if v['three_befor_year_count'] > 1} # 100交易次数以上 150999 最好数据 111次 ratio:0.036
    # statistics_new = {k: v for k, v in statistics.items() if v['trade_count'] > 100} # 100交易次数以上 1432237 最好数据 105次 ratio:0
    # statistics_new_1000 = {k: v for k, v in statistics.items() if v['trade_count'] > 1000}  # 1000交易次数以上 1280612 最好数据 1174次 ratio:0.0494
    # statistics_three_befor_year_count = {k: v for k, v in statistics.items() if v['three_befor_year_count'] > 1000}
    # statistics_three_befor_year_rate = dict(sorted(statistics_three_befor_year_count.items(), key=lambda x: x[1]['three_befor_year_count_thread_ratio'], reverse=False))
    # statistics_profit_temp = {k: v for k, v in statistics_new.items() if '实体_' not in k and '开盘_大于_20_固定区间' not in k and '收盘_大于_20_固定区间' not in k and '最高_大于_20_固定区间' not in k and '最低_大于_20_固定区间' not in k}
    # statistics_profit = sorted(statistics_profit_temp.items(), key=lambda x: x[1]['average_profit'], reverse=True)
    #
    # statistics_average_days_held = sorted(statistics_new.items(), key=lambda x: x[1]['average_days_held'], reverse=False)
    # statistics_1w_profit = sorted(statistics_new.items(), key=lambda x: x[1]['average_1w_profit'], reverse=True)
    # print(len(statistics))

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
