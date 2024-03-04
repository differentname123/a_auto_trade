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

from InfoCollector.save_all import save_all_data_mul
from StrategyExecutor.common import load_data, load_file_chunk, write_json, read_json
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


def calculate_technical_indicators(data):
    """
    计算股票市场数据的常用技术分析指标。

    参数:
    - data: DataFrame，包含市场数据的最高价、最低价、收盘价和成交量。

    返回值:
    - 无，函数将直接在传入的DataFrame中添加新列，每列代表一个技术指标。
    """
    # 计算随机振荡器（Stochastic Oscillator）
    data['Stochastic_Oscillator_K_归一化_信号'], data['Stochastic_Oscillator_D_归一化_信号'] = talib.STOCH(
        data['最高'], data['最低'], data['收盘'],
        fastk_period=14, slowk_period=3, slowd_period=3
    )

    # 计算威廉姆斯%R指标，反映市场超买或超卖情况
    data['Williams_R_归一化_信号'] = talib.WILLR(data['最高'], data['最低'], data['收盘'], timeperiod=14)

    data['Bollinger_Upper_股价'], data['Bollinger_Middle_股价'], data['Bollinger_Lower_股价'] = talib.BBANDS(data['收盘'],
                                                                                              timeperiod=20)

    # 可以考虑添加的其他重要技术指标
    # 计算相对强弱指数（Relative Strength Index，RSI），评估价格走势的速度和变化，以识别超买或超卖条件
    data['RSI_归一化_信号'] = talib.RSI(data['收盘'], timeperiod=14)

    # 计算移动平均收敛散度（Moving Average Convergence Divergence，MACD）
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(
        data['收盘'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    data['BAR_归一化_信号'] = (data['MACD'] - data['MACD_signal']) * 2

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
    new_columns_df = pd.DataFrame(new_columns)

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
            max_col_name = f'{key}_{window}日最大值'
            min_col_name = f'{key}_{window}日最小值'
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



def calculate_ratio_and_dynamic_frequency(data, key_list, ratio_windows=[5, 10, 20], frequency_windows=[30, 60, 120]):
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


def mark_holiday_eve_and_post_v3(data, date_column):
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

def get_data_feature(data):
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
    mark_holiday_eve_and_post_v3(data, '日期')

    # Calculate change percentages for close, open, high, and low prices over various periods
    calculate_change_percentages(data, periods=[1, 3, 5, 10], key_list=['收盘', '开盘', '最高', '最低'])

    # Filter columns for change percentages
    change_percentage_columns = [column for column in data.columns if '_涨跌幅_' in column]

    # Calculate crossover signals based on change percentages
    data = calculate_crossovers(data, change_percentage_columns)

    # Calculate technical indicators
    calculate_technical_indicators(data)

    # Calculate the relative amplitude for specified price types
    calculate_relative_amplitude(data, key_list=['收盘', '开盘', '最高', '最低'])

    # Filter columns excluding specific ones and those not containing signals
    columns_for_analysis = [column for column in data.columns
                            if column not in ['日期', '代码', '名称', '数量', 'Max_rate', 'Buy_Signal']
                            and '是否_信号' not in column]

    # Calculate ratios and dynamic frequencies for selected columns
    data = calculate_ratio_and_dynamic_frequency(data, columns_for_analysis)

    # Calculate continuous rise or fall in selected columns and mark trend reversal points
    data = calculate_trend_changes(data, columns_for_analysis)

    # Calculate rolling statistics (mean, max, min) for selected columns and mark extremes
    data = calculate_rolling_stats_with_max_min_flags_optimized(data, columns_for_analysis, windows=[3, 5, 10])

    # Calculate crossovers for specific price types and Bollinger Bands over various windows
    data = get_calculate_crossovers(data, key_list=['收盘', '开盘', '最高', '最低',
                                                       'Bollinger_Upper_股价', 'Bollinger_Middle_股价',
                                                       'Bollinger_Lower_股价'], windows=[3, 5, 10])

    # Replace NaN values with 0
    data.fillna(0, inplace=True)

    # Discard the initial rows that are less than the minimum required length
    data = data.iloc[min_required_length:]

    # Logging the time taken to compute features
    print(f"名称{data['名称'].iloc[0]} 计算特征耗时 {time.time() - start_time}")

    return data


def get_all_data_performance():
    """
    加载所有数据，并获取每天的表现情况。
    此函数首先遍历指定目录下的所有文件，然后使用多进程并行加载这些文件的数据。
    加载后，数据被合并并按日期分组，以分析每天的数据表现，并将结果保存到JSON文件中。
    """
    file_path = '../daily_data_exclude_new_can_buy'

    # 获取目录下所有文件的完整路径
    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]
    # 合并各个进程加载的数据
    all_data_df = load_and_merge_data(all_files)
    # 将日期列转换为日期时间格式，并处理格式错误
    all_data_df['日期'] = pd.to_datetime(all_data_df['日期'], errors='coerce')

    # 按日期分组
    grouped_data = all_data_df.groupby(all_data_df['日期'].dt.date)

    # 分析每组（每天）的数据表现
    performance_results = {}
    for date, group in grouped_data:
        date_str = date.strftime('%Y-%m-%d')  # 格式化日期为字符串
        performance = analyze_data_performance(group)
        performance_results[date_str] = performance

    # 保存分析结果到JSON文件
    results_file_path = '../final_zuhe/other/all_data_performance.json'
    write_json(results_file_path, performance_results)
    print(f'所有数据的表现分析已保存到 {results_file_path}')

def analyze_data_performance(data, min_profit_list=[1,2, 3]):
    """
    分析给定数据的表现情况。3天内的成功率
    :param data:
    :return:
    """
    result = {}
    for min_profit in min_profit_list:
        for days in [1, 2, 3]:
            key_name = '后续{}日最高价'.format(days)
            success_rate = (data[key_name] / data['收盘']).ge(1 + min_profit / 100).mean()
            result[f'后续{days}日{min_profit}成功率'] = success_rate
    return result

def generate_features_for_file(file_path, save_path):
    """
    读取指定路径的文件，生成其特征，并将特征数据保存到给定的保存路径。

    :param file_path: str, 待处理文件的完整路径。该文件是生成特征的数据源。
    :param save_path: str, 特征数据保存的目标目录路径。生成的文件将以源文件名保存在此路径下。
    :return: None
    """
    data = load_data(file_path, end_date='2024-03-01')
    new_data = get_data_feature(data)
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
    save_all_data_mul()
    with Pool() as pool:
        for root, _, files in os.walk(source_path):
            for file in files:
                full_path = os.path.join(root, file)
                pool.apply_async(generate_features_for_file, args=(full_path, save_path))
        pool.close()
        pool.join()

def load_and_merge_data(batch):
    start_time = time.time()
    # 使用多进程并行加载当前批次的数据
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)
    file_chunks = [batch[i::cpu_count] for i in range(cpu_count)]
    chunk_dfs = pool.map(load_file_chunk, file_chunks)
    pool.close()
    pool.join()

    # 合并当前批次加载的数据
    all_data_df = pd.concat(chunk_dfs)
    merge_time = time.time()
    print(f'当前批次数据合并耗时：{merge_time - start_time:.2f}秒')
    return all_data_df

def load_bad_data(ratio=0.5, profit=1, day=1):
    results_file_path = '../final_zuhe/other/all_data_performance.json'
    performance_results = read_json(results_file_path)
    key_name = f'后续{day}日{profit}成功率'
    bad_ratio_day = [pd.to_datetime(date).date() for date, result in performance_results.items() if result[key_name] < ratio]


    file_path = '../feature_data_exclude_new_can_buy'

    # 获取目录下所有文件的完整路径
    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]

    # 分成3个批次进行加载
    total_files = len(all_files)
    batch_size = total_files // 10
    batches = [all_files[i:i + batch_size] for i in range(0, total_files, batch_size)]

    # 定义一个空的DataFrame用于存储所有批次的数据
    batch_count = 0
    for batch in batches:
        batch_count += 1
        # 合并当前批次加载的数据
        all_data_df = load_and_merge_data(batch)
        # 获取日期在bad_ratio_day中的数据
        all_data_df['日期'] = pd.to_datetime(all_data_df['日期'])
        bad_data = all_data_df[all_data_df['日期'].dt.date.isin(bad_ratio_day)]
        print(f'总体数据量：{len(all_data_df)}，异常数据量：{len(bad_data)}')
        key = f'profit_{profit}_day_{day}'
        out_put_path = f'../train_data/{key}'
        file_name = f'bad_{ratio}_data_batch_count_{batch_count}.csv'
        os.makedirs(out_put_path, exist_ok=True)
        bad_data.to_csv(os.path.join(out_put_path, file_name), index=False)
        # 释放内存
        del all_data_df
        del bad_data

if __name__ == '__main__':
    # file_path = '../daily_data_exclude_new_can_buy'
    # out_path = '../feature_data_exclude_new_can_buy'
    # generate_features_for_all_files(file_path, out_path)


    # file_path = '../daily_data_exclude_new_can_buy/东方电子_000682.txt'
    # data = load_data(file_path)
    # new_data = get_data_feature(data)
    # print(new_data)

    # get_all_data_performance()
    # load_bad_data()

    file_path = '../train_data/profit_1_day_1'

    # 获取目录下所有文件的完整路径
    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(file_path) for file in files]
    data_list = []
    for file in all_files:

        data = pd.read_csv(file)
        data_list.append(data)
        print(f'文件{file}的数据量为{len(data)}')
    all_df = pd.concat(data_list)
    all_df.to_csv('../train_data/profit_1_day_1/bad_0.5_data_batch_count.csv', index=False)