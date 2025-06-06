# -*- coding: utf-8 -*-
""":authors:
    zhuxiaohu
:create_date:
    2024-03-01 23:30
:last_date:
    2024-03-01 23:30
:description:
    主要进行更加全面的指标计算
    指标的类别分为:1.每只股票自己分析出来的特征(收盘 开盘等数据的比较加上和一段时间比较将时间序列压缩到每一行) 2.大盘分析出来的特征（上证指数 + 深证指数 + 所有股票整体的分析(比如上涨比例 平均涨跌幅...)） 3.自己相对于大盘分析出来的特征（比如自己跌幅相对于大盘平均跌幅的比例）
    
"""
import json
import multiprocessing
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import akshare as ak

from InfoCollector.save_all import save_all_data_mul
from StrategyExecutor.common import load_data, load_file_chunk, write_json, read_json, load_data_filter, \
    load_file_chunk_filter, downcast_dtypes, low_memory_load
from itertools import combinations
import talib
import warnings
warnings.filterwarnings('ignore', message='.*Warning.*')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
def calculate_change_percentages(data, periods, key_list):
    """
    计算指定列在给定周期的涨跌幅，并将涨跌幅乘以100，保留两位小数。
    无法计算的值（例如，因为NaN）将被设置为0。

    参数:
    - data: DataFrame，包含市场数据。
    - periods: list of int，包含要计算涨跌幅的周期。
    - key_list: list of str，包含需要计算涨跌幅的列名。

    返回值:
    - 无，函数将直接修改传入的DataFrame。
    """
    for key in key_list:
        for period in periods:
            change_col_name = f'{key}_涨跌幅_{period}_归一化_信号'
            data[change_col_name] = (data[key].diff(period) / data[key].shift(period) * 100).round(2).fillna(0)


def calculate_relative_amplitude(data, key_list):
    """
    对于key_list中的键进行两两组合，计算它们的差值，并将差值分别除以每个键的前一天的值。
    结果列名格式为key1_key2_diff_相较于_key3_振幅，其中key3是被除数。

    参数:
    - data: DataFrame，包含市场数据。
    - key_list: list of str，包含需要进行操作的列名。

    返回值:
    - 无，函数将直接修改传入的DataFrame。
    """
    for key1, key2 in combinations(key_list, 2):
        # 计算两个键的差值
        diff = data[key1] - data[key2]
        # 对每个键的前一天的值进行操作
        for key in key_list:
            col_name = f'{key1}_{key2}_diff_相较于_{key}_振幅_归一化_信号'
            data[col_name] = (diff / data[key].shift(1) * 100).round(2)
            # 替换NaN值为0
            data[col_name].fillna(0, inplace=True)


def calculate_window_relative_amplitude(data, windows=[1,3,5,10], prefix=''):
    """
    对于windows中的每个窗口大小，计算该窗口内的最高和最低价格相对于窗口开始的振幅。

    参数:
    - data: DataFrame，包含市场数据。
    - windows: list of int，表示想要计算振幅的时间窗口大小。

    返回值:
    - 无，函数将直接修改传入的DataFrame。
    """
    for window in windows:
        high_col = f'窗口_{window}_天_{prefix}最高'
        low_col = f'窗口_{window}_天_{prefix}最低'
        amplitude_col = f'{window}_天_{prefix}振幅_归一化_信号'

        # 计算窗口内最高和最低价
        data[high_col] = data[prefix + '最高'].rolling(window=window, min_periods=1).max()
        data[low_col] = data[prefix + '最低'].rolling(window=window, min_periods=1).min()

        # 计算相对振幅
        data[amplitude_col] = ((data[high_col] - data[low_col]) / data[prefix + '收盘'].shift(window - 1) * 100).round(2)

        # 替换NaN值为0
        data[amplitude_col].fillna(0, inplace=True)

        # 可以选择删除临时列
        data.drop(columns=[high_col, low_col], inplace=True)

def calculate_technical_indicators(data, prefix=''):
    """
    计算股票市场数据的常用技术分析指标。

    参数:
    - data: DataFrame，包含市场数据的最高价、最低价、收盘价和成交量。

    返回值:
    - 无，函数将直接在传入的DataFrame中添加新列，每列代表一个技术指标。
    """
    # 计算随机振荡器（Stochastic Oscillator）
    data[prefix + 'Stochastic_Oscillator_K_归一化_信号'], data[prefix + 'Stochastic_Oscillator_D_归一化_信号'] = talib.STOCH(
        data[prefix + '最高'], data[prefix + '最低'], data[prefix + '收盘'],
        fastk_period=14, slowk_period=3, slowd_period=3
    )

    # 计算威廉姆斯%R指标，反映市场超买或超卖情况
    data[prefix + 'Williams_R_归一化_信号'] = talib.WILLR(data[prefix + '最高'], data[prefix + '最低'], data[prefix + '收盘'], timeperiod=14)

    data[prefix + 'Bollinger_Upper_股价'], data[prefix + 'Bollinger_Middle_股价'], data[prefix + 'Bollinger_Lower_股价'] = talib.BBANDS(data[prefix + '收盘'],
                                                                                                      timeperiod=20)

    # 可以考虑添加的其他重要技术指标
    # 计算相对强弱指数（Relative Strength Index，RSI），评估价格走势的速度和变化，以识别超买或超卖条件
    data[prefix + 'RSI_归一化_信号'] = talib.RSI(data[prefix + '收盘'], timeperiod=14)

    # 计算移动平均收敛散度（Moving Average Convergence Divergence，MACD）
    data[prefix + 'MACD'], data[prefix + 'MACD_signal'], data[prefix + 'MACD_hist'] = talib.MACD(
        data[prefix + '收盘'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    data[prefix + 'BAR_归一化_信号'] = (data[prefix + 'MACD'] - data[prefix + 'MACD_signal']) * 2

def calculate_trend_changes(data, key_list):
    """
    计算指定列的连续上涨或下跌天数，并标记趋势由涨转跌及由跌转涨的点。

    参数:
    - data: DataFrame，包含市场数据。
    - key_list: list of str，包含需要计算连续上涨或下跌天数和趋势变化的列名。

    返回值:
    - 修改原始DataFrame，为每个指定列添加四个新列，表示连续上涨天数、连续下跌天数及趋势转变点。
    """
    new_columns = {}  # 初始化一个新的字典来收集所有新列

    for key in key_list:
        daily_change = data[key].diff()
        rise_fall_signal = np.sign(daily_change)
        direction_change = rise_fall_signal.diff().ne(0)
        groups = direction_change.cumsum()
        consecutive_counts = rise_fall_signal.groupby(groups).cumsum().abs()

        # 直接在 new_columns 字典中存储连续上涨天数和连续下跌天数的计算结果
        new_columns[f'{key}_连续上涨天数_信号'] = np.where(rise_fall_signal > 0, consecutive_counts, 0)
        new_columns[f'{key}_连续下跌天数_信号'] = np.where(rise_fall_signal < 0, consecutive_counts, 0)

    # 将字典转换为 DataFrame
    new_columns_df = pd.DataFrame(new_columns, index=data.index)

    # 计算由跌转涨和由涨转跌的信号，并加入到 new_columns_df 中
    for key in key_list:
        new_columns_df[f'{key}_由跌转涨_信号'] = (
            (new_columns_df[f'{key}_连续上涨天数_信号'] > 0) &
            (new_columns_df[f'{key}_连续下跌天数_信号'].shift(1) == 1)
        ).astype(int)
        new_columns_df[f'{key}_由涨转跌_信号'] = (
            (new_columns_df[f'{key}_连续下跌天数_信号'] > 0) &
            (new_columns_df[f'{key}_连续上涨天数_信号'].shift(1) == 1)
        ).astype(int)

    # 合并原始 DataFrame 和新列 DataFrame
    data = pd.concat([data, new_columns_df], axis=1)

    return data


def calculate_rolling_stats_with_max_min_flags_optimized(data, key_list, windows=[3, 5, 10]):
    """
    计算指定列的3天、5天、10天滑动平均值，以及3天、5天、10天内的极大值和极小值。
    同时标记是否_信号为滑动窗口内的最大值或最小值，并优化性能以避免DataFrame碎片化。

    参数:
    - data: DataFrame，包含市场数据。
    - key_list: list of str，包含需要计算滑动统计数据的列名。

    返回值:
    - 修改后的DataFrame，包含新添加的统计量及其标记列。
    """
    new_columns = {}  # 用于收集新列的字典

    for key in key_list:
        for window in windows:
            # 计算滑动平均值、最大值、最小值
            avg_col_name = f'{key}_{window}日均值'
            new_columns[avg_col_name] = data[key].rolling(window=window).mean()
            max_values = data[key].rolling(window=window).max()
            min_values = data[key].rolling(window=window).min()

            # 标记是否为滑动窗口内的最大值或最小值
            new_columns[f'是否_信号_{key}_{window}日最大值'] = (data[key] == max_values).astype(int)
            new_columns[f'是否_信号_{key}_{window}日最小值'] = (data[key] == min_values).astype(int)

            # 标记是否大于或小于滑动窗口内的平均值
            new_columns[f'是否_信号_大于{key}_{window}日均值'] = (data[key] > new_columns[avg_col_name]).astype(int)
            new_columns[f'是否_信号_小于{key}_{window}日均值'] = (data[key] < new_columns[avg_col_name]).astype(int)

    # 一次性将所有新列添加到原DataFrame中
    data = pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)

    return data



def calculate_ratio_and_dynamic_frequency(data, key_list, ratio_windows=[5, 10], frequency_windows=[]):
    """
    计算指定列在不同滑动窗口内的值分布占比_信号，并动态计算频率。

    参数:
    - data: DataFrame，包含市场数据。
    - key_list: list of str，需要计算统计量的列名。
    - ratio_windows: list of int，定义计算占比_信号的滑动窗口的大小。
    - frequency_windows: list of int，定义计算频率的滑动窗口的大小。

    返回值:
    - 无，函数将直接在传入的DataFrame中添加新列，每列代表一个统计量。
    """
    new_columns = {}  # 用于收集新列的字典

    for key in key_list:
        # 计算占比_信号
        for window in ratio_windows:
            min_col = data[key].rolling(window=window).min()
            max_col = data[key].rolling(window=window).max()
            ratio_name = f'{key}_{window}日占比_信号'
            new_columns[ratio_name] = (data[key] - min_col) / (max_col - min_col)
            new_columns[ratio_name].fillna(0, inplace=True)  # 处理NaN和除以0的情况

        for window in frequency_windows:
            ratio_name = f'{key}_{window}日频率_占比_信号'
            temp_series = pd.Series(index=data.index, dtype=float)  # 初始化临时Series

            for idx in range(window - 1, len(data)):
                window_slice = data[key][idx - window + 1:idx + 1]
                current_value = window_slice.iloc[-1]
                min_val = window_slice.min()
                max_val = window_slice.max()

                if max_val != min_val:  # 避免除以零
                    ratio = (current_value - min_val) / (max_val - min_val)
                else:
                    ratio = 1.0  # 如果窗口期内所有值相同，则占比_信号设为1

                temp_series.at[data.index[idx]] = ratio

            new_columns[ratio_name] = temp_series  # 将计算好的占比_信号Series添加到字典中

    # 使用pd.concat一次性将所有新列添加到原DataFrame中
    data = pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)

    return data  # 确保函数返回更新后的DataFrame


def mark_holiday_eve_and_post_v3(data, date_column, next_saturday=False):
    """
    标记节前和节后的第一天、第二天和第三天，考虑到连续工作日小于3天的情况。

    参数:
    - data: DataFrame，包含日期字段。
    - date_column: str，日期字段的列名。

    返回值:
    - 修改原DataFrame，添加节前和节后标记列。
    """
    # 确保日期列是日期时间类型
    data[date_column] = pd.to_datetime(data[date_column])

    # 对日期进行排序
    data.sort_values(by=date_column, inplace=True)

    # 计算相邻日期之间的差距
    data['date_diff'] = data[date_column].diff().dt.days

    # 初始化节前和节后标记列
    for i in range(-3, 4):
        if i != 0:
            data[f'是否_信号_节假日_{i}天'] = False

    # 查找节假日
    holiday_indices = data.index[data['date_diff'] > 1].tolist()

    # 遍历每个节假日，标记节前和节后日期
    for idx in holiday_indices:
        # 标记节前日期
        for i in range(1, 4):
            if (idx - i) in data.index:
                data.at[idx - i, f'是否_信号_节假日_{i}天'] = True

        # 查找连续节假日的结束点
        next_day = data[date_column].loc[idx]
        if next_day in data[date_column].values:
            # 标记节后日期
            for i in range(1, 4):
                next_idx = idx + i - 1
                if next_idx <= data.index[-1]:
                    data.at[next_idx, f'是否_信号_节假日_{-i}天'] = True

    last_date = data[date_column].max()
    if next_saturday:
        # 将next_saturday转换为时间格式
        next_saturday = pd.to_datetime(next_saturday)
    else:
        next_saturday = last_date + pd.Timedelta(days=(5 - last_date.weekday()) % 7)
    days_diff = (next_saturday - last_date).days

    # 如果最近的星期六距离最后一天在3天之内，将其标记
    if days_diff <= 3:
        last_date_index = data.index[data[date_column] == last_date].tolist()  # Find the indices of the last date
        for i in range(days_diff, 4):
            idx = last_date_index[0] - i + days_diff
            if idx in data.index:
                data.at[idx, f'是否_信号_节假日_{i}天'] = True


    data.drop(columns=['date_diff'], inplace=True)


def calculate_crossovers(data, key_list):
    """
    计算key_list中每对键的两两组合是否发生了上穿或下穿。

    参数:
    - data: DataFrame，包含市场数据。
    - key_list: list of str，包含需要计算上穿或下穿的列名。

    返回值:
    - 修改后的DataFrame，包含新添加的上穿或下穿状态列。
    """
    new_columns = {}  # 用于收集新列的字典

    for key1, key2 in combinations(key_list, 2):
        # 计算前一天的差异
        prev_diff = data[key1].shift(1) - data[key2].shift(1)
        # 计算当天的差异
        current_diff = data[key1] - data[key2]

        # 上穿：前一天key1 < key2，今天key1 > key2
        crossover_up = (prev_diff < 0) & (current_diff > 0)
        # 下穿：前一天key1 > key2，今天key1 < key2
        crossover_down = (prev_diff > 0) & (current_diff < 0)

        # 添加上穿和下穿标记列到字典
        new_columns[f'是否_信号_{key1}_上穿_{key2}'] = crossover_up.astype(int)
        new_columns[f'是否_信号_{key1}_下穿_{key2}'] = crossover_down.astype(int)

    # 一次性将所有新列添加到原DataFrame中
    data = pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)

    return data


def filter_data_containing_all_keys(key_list, data_list):
    """
    返回data_list中包含key_list中所有关键字的数据项。

    参数:
    - key_list: 一个字符串列表，表示必须全部包含的关键字列表。
    - data_list: 一个字符串列表，表示待筛选的数据列表。

    返回值:
    - 包含所有关键字的数据项列表。
    """
    filtered_data = []
    for data in data_list:
        # 检查data中是否_信号包含key_list中的所有key
        if all(key in data for key in key_list):
            filtered_data.append(data)
    return filtered_data

def get_calculate_crossovers(data, key_list, windows=[3, 5, 10]):
    name_list = [f'{key}_{window}日均值' for window in windows for key in key_list]
    name_list.extend(key_list)
    return calculate_crossovers(data, name_list)

def sort_keys_by_max_min_diff(data, key_list):
    """
    计算每个key的最大最小值差值，并按降序排列这些key及其差值。

    参数:
    - data: DataFrame，包含市场数据。
    - key_list: list of str，包含需要计算差值的列名。

    返回值:
    - 一个按差值降序排列的包含(key, 差值)元组的列表。
    """
    diff_list = []

    for key in key_list:
        try:
            # 确保列是数值类型
            if pd.api.types.is_numeric_dtype(data[key]):
                max_val = data[key].max()
                min_val = data[key].min()
                diff = max_val - min_val
                diff_list.append((key, diff))
            else:
                print(f"列 '{key}' 不是数值类型，已跳过。")
        except Exception as e:
            print(f"计算列 '{key}' 的最大最小值差值时出现错误：{e}")

    sorted_diff_list = sorted(diff_list, key=lambda x: x[1], reverse=True)
    sorted_keys_with_diff = [f'{key} ({diff})' for key, diff in sorted_diff_list]

    return sorted_keys_with_diff

def calculate_future_high_prices(data):
    """
    计算后续1，2，3日的最高价。

    参数:
    - data: DataFrame，包含市场数据，其中应包含一个名为'最高价'的列。

    返回值:
    - 修改后的DataFrame，包含新增的后续1，2，3日最高价列。
    """
    # 计算后续1，2，3日的最高价
    for days in [1, 2, 3]:
        # 使用shift()将数据向上移动指定的天数，然后使用rolling()和max()计算指定窗口内的最大值
        data[f'后续{days}日最高价'] = data['最高'].shift(-days).rolling(window=days, min_periods=1).max()

    return data

def get_data_feature(data, next_saturday=False):
    """
    Generate features for a given dataset. The dataset includes open, close, high, low prices,
    volume, date, ticker symbol, and turnover rate. This function adds technical indicators,
    calculates price changes, marks pre and post-holiday effects, and more.

    :param data: DataFrame containing the stock data with columns for opening, closing, highest,
                 lowest prices, volume, date, code, and turnover rate.
    :return: DataFrame with added features or None if the input data is too short.
    """
    start_time = time.time()
    min_required_length = 26  # Minimum required length of the data to proceed

    # Check if data length is less than the minimum required length
    if len(data) < min_required_length:
        print(f"{data['名称'].iloc[0]}数据长度小于{min_required_length}，跳过")
        return None

    # Mark the days before and after holidays
    mark_holiday_eve_and_post_v3(data, '日期', next_saturday=next_saturday)

    # Calculate change percentages for close, open, high, and low prices over various periods
    calculate_change_percentages(data, periods=[3, 5, 10], key_list=['收盘'])

    # Filter columns for change percentages
    change_percentage_columns = [column for column in data.columns if '涨跌幅' in column]

    # Calculate crossover signals based on change percentages
    data = calculate_crossovers(data, change_percentage_columns)

    # Calculate technical indicators
    calculate_technical_indicators(data)

    # Calculate the relative amplitude for specified price types
    calculate_window_relative_amplitude(data)

    # Filter columns excluding specific ones and those not containing signals
    columns_for_analysis = [column for column in data.columns
                            if (column not in ['日期', '代码', 'code','current_price', '名称', '数量', 'Max_rate', 'Buy_Signal']
                            and '是否_信号' not in column and '相较于' not in column and '后续' not in column )]

    # Calculate ratios and dynamic frequencies for selected columns
    data = calculate_ratio_and_dynamic_frequency(data, columns_for_analysis)

    # Calculate continuous rise or fall in selected columns and mark trend reversal points
    data = calculate_trend_changes(data, columns_for_analysis)

    # Calculate rolling statistics (mean, max, min) for selected columns and mark extremes
    data = calculate_rolling_stats_with_max_min_flags_optimized(data, columns_for_analysis, windows=[3, 5, 10])

    # Calculate crossovers for specific price types and Bollinger Bands over various windows
    data = get_calculate_crossovers(data, key_list=['收盘', 'Bollinger_Upper_股价', 'Bollinger_Lower_股价'], windows=[3, 5, 10])

    # Replace NaN values with 0
    data.fillna(0, inplace=True)

    # Discard the initial rows that are less than the minimum required length
    data = data.iloc[min_required_length:]

    # Logging the time taken to compute features
    # print(f"名称{data['名称'].iloc[0]} 计算特征耗时 {time.time() - start_time}")

    delete_column = [column for column in data.columns if '信号' not in column and '均值' in column]
    # 将delete_column从data中删除
    data.drop(columns=delete_column, inplace=True)


    # sort_keys_by_max_min_diff(data,[column for column in data.columns if '信号' in column])

    return data

def get_index_data_feature(data):
    """
    计算大盘数据的特征。这些特征包括上证指数、深证指数、所有股票的平均涨跌幅、上涨比例等。
    """

    # Mark the days before and after holidays
    mark_holiday_eve_and_post_v3(data, '指数日期')

    # Calculate change percentages for close, open, high, and low prices over various periods
    calculate_change_percentages(data, periods=[1, 3, 5], key_list=['深证指数收盘', '上证指数收盘'])

    # Filter columns for change percentages
    change_percentage_columns = [column for column in data.columns if '涨跌幅_1_' in column or '涨跌幅_3_' in column]

    # Calculate crossover signals based on change percentages
    data = calculate_crossovers(data, change_percentage_columns)

    # Calculate technical indicators
    calculate_technical_indicators(data, prefix='深证指数')
    calculate_technical_indicators(data, prefix='上证指数')

    # Calculate the relative amplitude for specified price types
    calculate_window_relative_amplitude(data, windows=[1,3,5], prefix='上证指数')
    calculate_window_relative_amplitude(data, windows=[1,3,5], prefix='深证指数')

    # Filter columns excluding specific ones and those not containing signals
    columns_for_analysis = [column for column in data.columns if (column in ['深证指数开盘', '深证指数收盘', '深证指数最高', '深证指数最低', '深证指数成交额', '上证指数开盘', '上证指数收盘', '上证指数最高', '上证指数最低', '上证指数成交额','深证指数收盘_涨跌幅_1_归一化_信号', '深证指数收盘_涨跌幅_3_归一化_信号', '深证指数收盘_涨跌幅_5_归一化_信号', '深证指数收盘_涨跌幅_10_归一化_信号', '上证指数收盘_涨跌幅_1_归一化_信号', '上证指数收盘_涨跌幅_3_归一化_信号', '上证指数收盘_涨跌幅_5_归一化_信号', '上证指数收盘_涨跌幅_10_归一化_信号'
                                                                             , '上证指数Bollinger_Upper_股价', '上证指数Bollinger_Lower_股价','深证指数Bollinger_Upper_股价', '深证指数Bollinger_Lower_股价','深证指数RSI_归一化_信号','深证指数BAR_归一化_信号', '上证指数RSI_归一化_信号','上证指数BAR_归一化_信号','1_天_上证指数振幅_归一化_信号', '3_天_上证指数振幅_归一化_信号', '5_天_上证指数振幅_归一化_信号', '10_天_上证指数振幅_归一化_信号', '1_天_深证指数振幅_归一化_信号', '3_天_深证指数振幅_归一化_信号', '5_天_深证指数振幅_归一化_信号', '10_天_深证指数振幅_归一化_信号'])]
    columns_for_analysis = [column for column in columns_for_analysis if '最' not in column and '5' not in column and '10' not in column]
    # Calculate ratios and dynamic frequencies for selected columns
    data = calculate_ratio_and_dynamic_frequency(data, columns_for_analysis)

    # Calculate continuous rise or fall in selected columns and mark trend reversal points
    data = calculate_trend_changes(data, columns_for_analysis)
    data = calculate_rolling_stats_with_max_min_flags_optimized(data, columns_for_analysis, windows=[3, 5])

    # Calculate crossovers for specific price types and Bollinger Bands over various windows
    data = get_calculate_crossovers(data, key_list=['深证指数收盘', '深证指数Bollinger_Upper_股价', '深证指数Bollinger_Lower_股价'], windows=[3, 5])
    data = get_calculate_crossovers(data, key_list=['上证指数收盘', '上证指数Bollinger_Upper_股价', '上证指数Bollinger_Lower_股价'], windows=[3, 5])

    # Replace NaN values with 0
    data.fillna(0, inplace=True)


    # Logging the time taken to compute features
    # print(f"名称{data['名称'].iloc[0]} 计算特征耗时 {time.time() - start_time}")

    delete_column = [column for column in data.columns if '信号' not in column and '均值' in column]
    # 将delete_column从data中删除
    data.drop(columns=delete_column, inplace=True)


    # sort_keys_by_max_min_diff(data,[column for column in data.columns if '信号' in column])

    return data


def get_all_data_performance():
    """
    加载所有数据，并获取每天的表现情况。
    此函数首先遍历指定目录下的所有文件，然后使用多进程并行加载这些文件的数据。
    加载后，数据被合并并按日期分组，以分析每天的数据表现，并将结果保存到JSON文件中。
    """
    file_path = '../train_data/2024_data_2024.csv'

    all_data_df = low_memory_load(file_path)
    # 将日期列转换为日期时间格式，并处理格式错误
    all_data_df['日期'] = pd.to_datetime(all_data_df['日期'], errors='coerce')

    # 按日期分组
    grouped_data = all_data_df.groupby(all_data_df['日期'].dt.date)

    # 分析每组（每天）的数据表现
    performance_results = {}
    pre_date_str = ''
    next_profit = 0
    for date, group in grouped_data:
        date_str = date.strftime('%Y-%m-%d')  # 格式化日期为字符串
        performance = analyze_data_performance(group)
        performance_results[date_str] = performance
        # 如果pre_date_str在grouped_data中
        if pre_date_str in performance_results.keys():
            # 计算前一天的上涨比例
            rise_rate = performance_results[date_str]['上涨比例']
            performance_results[pre_date_str]['下一天上涨比例'] = rise_rate
        performance_results[date_str]['后续0日1成功率'] = next_profit

        pre_date_str = date_str
        next_profit = performance_results[date_str]['后续1日1成功率']
    performance_results[date_str]['下一天上涨比例'] = 0


    # 保存分析结果到JSON文件
    results_file_path = '../final_zuhe/other/2024_all_data_performance.json'
    write_json(results_file_path, performance_results)
    print(f'所有数据的表现分析已保存到 {results_file_path}')

def analyze_data_performance(data, min_profit_list=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]):
    """
    分析给定数据的表现情况。3天内的成功率
    :param data:
    :return:
    """
    result = {}
    for min_profit in min_profit_list:
        for days in [1, 2, 3]:
            key_name = '后续{}日最高价利润率'.format(days)
            success_rate = (data[key_name]).ge(min_profit).mean()
            result[f'后续{days}日{min_profit}成功率'] = success_rate
            # 增加上涨个数比例
    rise_rate = (data['涨跌幅'] > 0).mean()
    result['上涨比例'] = rise_rate
    return result

def generate_features_for_file(file_path, save_path):
    """
    读取指定路径的文件，生成其特征，并将特征数据保存到给定的保存路径。

    :param file_path: str, 待处理文件的完整路径。该文件是生成特征的数据源。
    :param save_path: str, 特征数据保存的目标目录路径。生成的文件将以源文件名保存在此路径下。
    :return: None
    """
    data_list = load_data_filter(file_path, start_date='2023-01-01', end_date='2025-05-01')
    new_data_list = []
    for data in data_list:
        new_data = get_data_feature(data)
        new_data_list.append(new_data)
    if len(new_data_list) != 0:
        new_data = pd.concat(new_data_list)
        if new_data is not None:
            new_data.to_csv(os.path.join(save_path, os.path.basename(file_path)), index=False)

def generate_features_for_all_files(source_path, save_path):
    """
    遍历指定路径下的所有文件，为每个文件生成特征并保存。

    该函数首先调用一个函数来保存所有必要的数据（此处省略了该调用的具体实现细节），然后使用多进程方式并行处理每个文件，
    为每个文件生成特征并将结果保存到指定的目录中。

    :param source_path: str, 包含待处理文件的源目录路径。该函数将递归遍历此路径下的所有文件。
    :param save_path: str, 生成的特征文件的保存目录路径。每个文件的特征将保存为一个新文件，位置在此路径下。
    :return: None
    """
    # 删除source_path和save_path下的所有文件
    # for root, _, files in os.walk(save_path):
    #     for file in files:
    #         os.remove(os.path.join(root, file))
    # for root, _, files in os.walk(source_path):
    #     for file in files:
    #         os.remove(os.path.join(root, file))

    save_all_data_mul()
    # with Pool() as pool:
    #     for root, _, files in os.walk(source_path):
    #         for file in files:
    #             full_path = os.path.join(root, file)
    #             pool.apply_async(generate_features_for_file, args=(full_path, save_path))
    #     pool.close()
    #     pool.join()

def load_and_merge_data(batch):
    start_time = time.time()
    # 使用多进程并行加载当前批次的数据
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=10)
    file_chunks = [batch[i::cpu_count] for i in range(cpu_count)]
    chunk_dfs = pool.map(load_file_chunk_filter, file_chunks)
    pool.close()
    pool.join()
    try:
    # 合并当前批次加载的数据
        all_data_df = pd.concat(chunk_dfs)
        merge_time = time.time()
        print(f'当前批次数据合并耗时：{merge_time - start_time:.2f}秒')
        return all_data_df
    except Exception as e:
        print(f'合并失败，错误为{e}')
    return None


def split_dataframe(data_df, split_num):
    """
    将 DataFrame 中的数据按行分割成多个列表,并返回这些列表。

    参数:
    data_df (pandas.DataFrame): 输入的 DataFrame
    split_num (int): 需要分割的列表数量

    返回:
    list: 包含分割后的多个列表
    """
    start = time.time()
    # 按照日期分组并排序
    grouped = data_df.sort_values(by=['日期', '后续1日最高价利润率', '后续2日最高价利润率', '涨跌幅']).groupby('日期')

    # 创建 split_num 个空列表
    split_lists = [[] for _ in range(split_num)]

    # 遍历分组后的数据
    for _, group_df in grouped:
        group_values = group_df.values.tolist()
        for i, row in enumerate(group_values):
            split_lists[i % split_num].append(row)

    # 将每个列表合并为 DataFrame
    split_df_lists = [pd.DataFrame(split_list, columns=data_df.columns) for split_list in split_lists]
    print(f'分割数据耗时：{time.time() - start:.2f}秒')
    return split_df_lists

def merge_splited_data():
    """
    合并分割后的数据，之前数据因为大小关系被分成了两份
    :return:
    """
    file_path = '../train_data'
    # 获取file_path目录下所有的文件夹
    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]

def load_bad_data():
    """
    all_data_performance.json
    :param ratio:
    :param profit:
    :param day:
    :return:
    """
    start_time = time.time()
    # all_data_file_path = '../train_data/profit_1_day_1_ratio_0.25/good_0.25_data_batch_count.csv'
    all_data_file_path = '../train_data/all_data.csv'
    if not os.path.exists(all_data_file_path):
        print('开始加载所有数据')
        # 生成相应的数据
        file_path = '../feature_data_exclude_new_can_buy'
        # 获取目录下所有文件的完整路径
        all_files = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]
        all_data_df = load_and_merge_data(all_files)
        all_data_df.to_csv(all_data_file_path, index=False)
    else:
        print('开始加载已有的所有数据')
        all_data_df = low_memory_load(all_data_file_path)
    all_data_df['日期'] = pd.to_datetime(all_data_df['日期'])
    all_data_df_2024 = all_data_df[all_data_df['日期'] >= pd.Timestamp('2024-01-01')]
    all_data_df_2024.to_csv('../train_data/2024_data_2024.csv', index=False)
    return

    # del all_data_df_2024
    all_data_df = downcast_dtypes(all_data_df)
    all_data_df = all_data_df[all_data_df['日期'] >= pd.Timestamp('2021-01-01')]
    all_data_df = all_data_df[all_data_df['日期'] < pd.Timestamp('2024-01-01')]

    # all_data_df = all_data_df[all_data_df['日期'] < pd.Timestamp('2021-01-01')]
    # return
    print(f'加载所有数据耗时：{time.time() - start_time:.2f}秒')
    # time.sleep(10)
    new_data = data_filter(all_data_df)
    print(f'加载所有数据耗时：{time.time() - start_time:.2f}秒 过滤前数据量为{len(all_data_df)} 过滤后数据量为{len(new_data)}')
    all_data_df = new_data
    del new_data

    results_file_path = '../final_zuhe/other/all_data_performance.json'
    performance_results = read_json(results_file_path)

    ratio_list = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]
    profit_list = [1]
    day_list = [1, 2]

    for profit in profit_list:
        for day in day_list:
            for ratio in ratio_list:
                key_name = f'后续{day}日{profit}成功率'
                bad_ratio_day = [pd.to_datetime(date).date() for date, result in performance_results.items() if
                                 result[key_name] <= ratio]
                key = f'profit_{profit}_day_{day}_ratio_{ratio}'
                out_put_path = f'../train_data/{key}'
                print(f'开始加载bad_{out_put_path}的数据')
                file_name = f'bad_{ratio}_data_batch_count_2023.csv'
                good_file_name = f'good_{ratio}_data_batch_count_2023.csv'
                bad_file_out_put_path = os.path.join(out_put_path, good_file_name)
                file_out_put_path = os.path.join(out_put_path, file_name)

                # 检查目录是否存在，不存在则创建
                os.makedirs(out_put_path, exist_ok=True)

                bad_data = all_data_df[all_data_df['日期'].dt.date.isin(bad_ratio_day)]
                print(f'bad_{out_put_path}的数据量为{len(bad_data)}')
                # if not os.path.exists(file_out_put_path) or len(bad_data) > 5000000:
                #     bad_data.to_csv(file_out_put_path, index=False)

                if len(bad_data) <= 5000000:
                    for idx, temp_df in enumerate(split_dataframe(bad_data, 3)):
                        temp_file_path = f'../train_data/{key}/bad_{idx}_2023.csv'
                        if not os.path.exists(temp_file_path):
                            temp_df.to_csv(temp_file_path, index=False)
                            all_10_list = split_dataframe(temp_df, 10)
                            train_df = pd.concat(all_10_list[:7], ignore_index=True)
                            test_df = pd.concat(all_10_list[7:], ignore_index=True)
                            train_file_path = f'../train_data/{key}/bad_{idx}_train_2023.csv'
                            test_file_path = f'../train_data/{key}/bad_{idx}_test_2023.csv'
                            if not os.path.exists(train_file_path):
                                train_df.to_csv(train_file_path, index=False)
                            if not os.path.exists(test_file_path):
                                test_df.to_csv(test_file_path, index=False)
                            print(f'bad_{idx}_2023.csv的数据量为{len(temp_df)}')
                            del temp_df, train_df, test_df
                del bad_data

                good_data = all_data_df[~all_data_df['日期'].dt.date.isin(bad_ratio_day)]
                print(f'good_{out_put_path}的数据量为{len(good_data)}')
                # if not os.path.exists(bad_file_out_put_path) or len(good_data) > 5000000:
                #     good_data.to_csv(bad_file_out_put_path, index=False)
                if len(good_data) <= 5000000:
                    for idx, temp_df in enumerate(split_dataframe(good_data, 3)):
                        temp_file_path = f'../train_data/{key}/good_{idx}_2023.csv'
                        if not os.path.exists(temp_file_path):
                            temp_df.to_csv(temp_file_path, index=False)
                            all_10_list = split_dataframe(temp_df, 10)
                            train_df = pd.concat(all_10_list[:7], ignore_index=True)
                            test_df = pd.concat(all_10_list[7:], ignore_index=True)
                            train_file_path = f'../train_data/{key}/good_{idx}_train_2023.csv'
                            test_file_path = f'../train_data/{key}/good_{idx}_test_2023.csv'
                            if not os.path.exists(train_file_path):
                                train_df.to_csv(train_file_path, index=False)
                            if not os.path.exists(test_file_path):
                                test_df.to_csv(test_file_path, index=False)
                            print(f'good_{idx}_2023.csv的数据量为{len(temp_df)}')
                            del temp_df, train_df, test_df
                del good_data


def data_filter(data):
    """
    过滤数据，过滤掉涨跌幅大于0，且开盘、收盘、低价、高价之间最大相差小于0.01的数据。

    :param data: pandas DataFrame，包含市场数据，预期有涨跌幅、开盘、收盘、最低价、最高价等列。
    :return: 过滤后的pandas DataFrame。
    """
    # 检查必要的列是否存在
    necessary_columns = ['涨跌幅', '开盘', '收盘', '最低', '最高']
    for column in necessary_columns:
        if column not in data.columns:
            raise ValueError(f"缺少必要的列：{column}")

    # 计算开盘、收盘、最低价、最高价之间的最大差异
    price_range = data[['开盘', '收盘', '最低', '最高']].max(axis=1) - data[['开盘', '收盘', '最低', '最高']].min(axis=1)

    # 应用过滤条件
    filtered_data = data[(data['涨跌幅'] <= 0) | (price_range >= 0.01)]
    filtered_data = filtered_data[(filtered_data['涨跌幅'] <= (filtered_data['Max_rate'] - 0.01 / filtered_data['收盘'].shift(1) * 100))]

    return filtered_data

def data_filter_for_single(data):
    """
    针对单个股票数据进行过滤，发现异常数据，直接删除后面一个月的数据

    :param data: pandas DataFrame，包含市场数据，预期有涨跌幅、开盘、收盘、最低价、最高价等列。
    :return: 过滤后的pandas DataFrame。
    """
    # 检查必要的列是否存在
    necessary_columns = ['涨跌幅', '开盘', '收盘', '最低', '最高']
    for column in necessary_columns:
        if column not in data.columns:
            raise ValueError(f"缺少必要的列：{column}")

    # 计算开盘、收盘、最低价、最高价之间的最大差异
    price_range = data[['开盘', '收盘', '最低', '最高']].max(axis=1) - data[['开盘', '收盘', '最低', '最高']].min(axis=1)

    # 应用过滤条件
    filtered_data = data[(data['涨跌幅'] <= 0) | (price_range >= 0.01)]
    filtered_data = filtered_data[(filtered_data['涨跌幅'] <= (filtered_data['Max_rate'] - 0.01 / filtered_data['收盘'].shift(1) * 100))]

    return filtered_data

def is_precision_always_increasing(json_list, max_count=2, min_cha=-0.1):
    # 首先检查列表是否为空或只有一个元素
    if len(json_list) < 2:
        return True
    # if json_list[-1]['true_stocks_set'] != []:
    #     return True
    # else:
    #     return False
    # 遍历列表，比较每个元素的precision值
    for i in range(len(json_list) - 1):
        # 获取当前元素和下一个元素的precision值
        current_precision = json_list[i].get('precision', None)
        next_precision = json_list[i + 1].get('precision', None)

        # 检查precision值是否存在
        if current_precision is None or next_precision is None:
            raise ValueError("Some items are missing the 'precision' attribute")

        # 如果当前的precision值不小于下一个，返回False
        if (current_precision + min_cha) > next_precision or current_precision <= 0.5:
            max_count -= 1
            # print(f"Precision is not always increasing: {json_list[i].get('abs_threshold', None)}")
            if max_count == 0:
                return False

    # 所有比较都显示precision是递增的
    return True

def test():
    # # 读取'../model/other/good_model_list.txt'
    # with open('../model/other/good_model_list.txt', 'r') as f:
    #     good_model_list = f.readlines()
    # good_model_list_befor = [model.strip() for model in good_model_list]
    # # 读取'../model/other/bad_model_list.txt'
    # with open('../model/other/bad_model_list.txt', 'r') as f:
    #     bad_model_list = f.readlines()
    # bad_model_list_befor = [model.strip() for model in bad_model_list]
    #
    # with open('../model/other/good_model_list_last.txt', 'r') as f:
    #     good_model_list = f.readlines()
    # good_model_list_last = [model.strip() for model in good_model_list]
    # # 读取'../model/other/bad_model_list.txt'
    # with open('../model/other/bad_model_list_last.txt', 'r') as f:
    #     bad_model_list = f.readlines()
    # bad_model_list_last = [model.strip() for model in bad_model_list]
    #
    # # 找到good_model_list_last比good_model_list多的数据
    # last_more = set(good_model_list_last) - set(good_model_list_befor)
    # last_less = set(good_model_list_befor) - set(good_model_list_last)

    good_model_list = []
    bad_model_list = []
    file_path = '../model/reports'
    # 遍历file_path下的所有json文件
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.json'):
                json_file = os.path.join(root, file)
                data = read_json(json_file)
                for model_name, model_data in data.items():
                    for test_data, test_result in model_data.items():
                        if 'bad_0_test.csv' not in test_data:
                            continue
                        detail_report = test_result['tree_0_abs_1']
                        is_precision = is_precision_always_increasing(detail_report)
                        if is_precision:
                            good_model_list.append(model_name)
                        else:
                            bad_model_list.append(model_name)
    print(f'good_model_list:{len(good_model_list)}')
    print(f'bad_model_list:{len(bad_model_list)}')
    # 将good_model_list和bad_model_list写入文件，每个模型名占用一行
    with open('../model/other/good_model_list.txt', 'w') as f:
        for model in good_model_list:
            f.write(model + '\n')
    with open('../model/other/bad_model_list.txt', 'w') as f:
        for model in bad_model_list:
            f.write(model + '\n')

def update_2024_data():
    """
    更新2024年的数据
    :return:
    """
    file_path = '../daily_data_exclude_new_can_buy'
    out_path = '../feature_data_exclude_new_can_buy'
    generate_features_for_all_files(file_path, out_path)
    print('开始加载所有数据')
    # 生成相应的数据
    file_path = '../feature_data_exclude_new_can_buy'
    # 获取目录下所有文件的完整路径
    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]
    all_data_df = load_and_merge_data(all_files)
    all_data_df['日期'] = pd.to_datetime(all_data_df['日期'])
    all_data_df_2024 = all_data_df[all_data_df['日期'] >= pd.Timestamp('2024-01-01')]
    key_signal_columns = [column for column in all_data_df_2024.columns if '后续' in column]
    key_signal_columns.append('涨跌幅')
    key_signal_columns.extend(['日期', '代码'])
    final_data = all_data_df_2024[key_signal_columns]
    # 将data写入'../final_zuhe/other/2024_data_2024_simple.json'
    final_data.to_csv('../final_zuhe/other/2024_data_2024_simple.csv', index=False)
    all_data_df_2024.to_csv('../train_data/2024_data_2024.csv', index=False)
    get_all_data_performance()

    # compare_origin_selected_samples = low_memory_load('../train_data/2024_data.csv')
    # compare_origin_selected_samples = compare_origin_selected_samples[
    #     compare_origin_selected_samples.columns.drop(list(compare_origin_selected_samples.filter(regex='信号')))]
    # compare_origin_selected_samples1 = low_memory_load('../train_data/2024_data_2024.csv')
    # compare_origin_selected_samples1 = compare_origin_selected_samples1[
    #     compare_origin_selected_samples1.columns.drop(list(compare_origin_selected_samples1.filter(regex='信号')))]
    # # 合并compare_origin_selected_samples和compare_origin_selected_samples1，删除重复的数据
    # compare_origin_selected_samples = pd.concat([compare_origin_selected_samples, compare_origin_selected_samples1], ignore_index=True)
    # compare_origin_selected_samples.to_csv('../train_data/full_2024_compare.csv', index=False)

def update_2024_data_simple():
    """
    更新2024年的数据
    :return:
    """
    start_time = time.time()
    # save_path = '../daily_data_exclude_new_can_buy'
    # # 删除source_path和save_path下的所有文件
    # for root, _, files in os.walk(save_path):
    #     for file in files:
    #         os.remove(os.path.join(root, file))
    # for root, _, files in os.walk(save_path):
    #     for file in files:
    #         os.remove(os.path.join(root, file))

    # save_all_data_mul()

    stock_data_df = ak.stock_zh_a_spot_em()
    exclude_code_set = set(ak.stock_kc_a_spot_em()['代码'].tolist())
    exclude_code_set.update(ak.stock_cy_a_spot_em()['代码'].tolist())
    need_code_set = {code for code in stock_data_df['代码'] if
                     code.startswith(('000', '002', '003', '001', '600', '601', '603', '605'))}
    new_exclude_code_set = set(stock_data_df['代码']) - need_code_set
    new_exclude_code_set.update(exclude_code_set)

    # 筛选出满足条件的股票数据
    filtered_stock_data_df = stock_data_df[~stock_data_df['代码'].isin(new_exclude_code_set)]
    print('筛选后的数据量：{}'.format(filtered_stock_data_df.shape[0]))

    # 获取filtered_stock_data_df的所有代码
    code_list = filtered_stock_data_df['代码'].tolist()


    print(f'开始加载所有数据 耗时：{time.time() - start_time:.2f}秒')
    # 生成相应的数据
    file_path = '../daily_data_exclude_new_can_buy'
    # 获取目录下所有文件的完整路径
    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]
    # 过滤all_files，只保留名称包含code_list的文件
    all_files = [file for file in all_files if any(code in file for code in code_list)]
    print(f'all_files的数量为{len(all_files)}')

    all_data_df = load_and_merge_data(all_files)
    print(f'加载所有数据 耗时：{time.time() - start_time:.2f}秒')
    all_data_df['日期'] = pd.to_datetime(all_data_df['日期'])
    all_data_df_2024 = all_data_df[all_data_df['日期'] >= pd.Timestamp('2025-01-01')]
    key_signal_columns = [column for column in all_data_df_2024.columns if '后续' in column]
    key_signal_columns.append('涨跌幅')
    key_signal_columns.extend(['日期', '代码'])
    key_signal_columns.extend(['换手率'])
    final_data = all_data_df_2024[key_signal_columns]
    # 将data写入'../final_zuhe/other/2024_data_2024_simple.json'
    final_data.to_csv('../final_zuhe/other/2024_data_2024_simple.csv', index=False)
    all_data_df_2024.to_csv('../train_data/2024_data_2024.csv', index=False)
    get_all_data_performance()
    print(f'加载所有数据 耗时：{time.time() - start_time:.2f}秒')

    # compare_origin_selected_samples = low_memory_load('../train_data/2024_data.csv')
    # compare_origin_selected_samples = compare_origin_selected_samples[
    #     compare_origin_selected_samples.columns.drop(list(compare_origin_selected_samples.filter(regex='信号')))]
    # compare_origin_selected_samples1 = low_memory_load('../train_data/2024_data_2024.csv')
    # compare_origin_selected_samples1 = compare_origin_selected_samples1[
    #     compare_origin_selected_samples1.columns.drop(list(compare_origin_selected_samples1.filter(regex='信号')))]
    # # 合并compare_origin_selected_samples和compare_origin_selected_samples1，删除重复的数据
    # compare_origin_selected_samples = pd.concat([compare_origin_selected_samples, compare_origin_selected_samples1], ignore_index=True)
    # compare_origin_selected_samples.to_csv('../train_data/full_2024_compare.csv', index=False)

if __name__ == '__main__':
    update_2024_data_simple()
    # get_all_data_performance()

    # test()

    # for root, _, files in os.walk('../daily_data_exclude_new_can_buy'):
    #     for file in files:
    #         full_path = os.path.join(root, file)
    # full_path = '../daily_data_exclude_new_can_buy/北讯退_002359.txt'
    # data = load_data_filter(full_path, end_date='2024-03-01')

    # file_path = '../daily_data_exclude_new_can_buy'
    # # 获取目录下所有文件的完整路径
    # all_files = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]
    # # all_files = all_files[:100]
    # all_data_df = load_and_merge_data(all_files)
    # print(all_data_df)

    # file_path = '../daily_data_exclude_new_can_buy'
    # out_path = '../feature_data_exclude_new_can_buy'
    # generate_features_for_all_files(file_path, out_path)
    # load_bad_data()

    # file_path = '../feature_data_exclude_new_can_buy/ST实达_600734.txt'
    # file_path = '../train_data/2024_data.csv'
    # data = pd.read_csv(file_path)
    # split_dataframe(data, 3)
    # new_data = data_filter(data)
    # print(new_data)
    # # 找到data比new_data多的数据
    # print(data[~data.isin(new_data).all(1)])

    # file_path = '../daily_data_exclude_new_can_buy/东方电子_000682.txt'
    # data = load_data(file_path)
    # # 除去data最后一行
    # # data = data[:-1]
    # new_data = get_data_feature(data, '2024-01-01')
    # print(new_data)

    # data = pd.read_csv('../train_data/index_data.csv')
    # get_index_data_feature(data)

    # file_path = '../train_data/profit_1_day_1'
    #
    # # 获取目录下所有文件的完整路径
    # all_files = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]
    # data_list = []
    # for file in all_files:
    #
    #     data = pd.read_csv(file)
    #     data_list.append(data)
    #     print(f'文件{file}的数据量为{len(data)}')
    # all_df = pd.concat(data_list)
    # all_df.to_csv('../train_data/profit_1_day_1/bad_0.5_data_batch_count.csv', index=False)