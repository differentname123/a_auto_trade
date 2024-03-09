# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/19 18:14
:last_date:
    2023/10/19 18:14
:description:

"""
import datetime
import json
import os
import sys
import traceback

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import talib
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from plotly.subplots import make_subplots

# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/19 18:14
:last_date:
    2023/10/19 18:14
:description:

"""
import datetime
import math
import os
import sys
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import talib
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from plotly.subplots import make_subplots
from datetime import timedelta

import time
import functools

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 耗时 {end_time - start_time} 秒")
        return result
    return wrapper

def parse_filename(file_path):
    # 提取文件名（不包括扩展名）
    base_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]

    # 以'_'分割文件名以获取名称和编号
    stock_name, stock_code = file_name_without_ext.split('_')

    return stock_name, stock_code

# @timeit
def load_data(file_path, start_date='2018-01-01', end_date='2024-01-01'):
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
    data = data[data['日期'] < pd.Timestamp(end_date)]
    data = data[data['日期'] > pd.Timestamp(start_date)]

    # 如果有大于30天的断层
    if not filtered_diff.empty:
        cutoff_index = filtered_diff.idxmax()
        if cutoff_index and cutoff_index != 0:
            data = data.loc[cutoff_index + 1:]  # 跳过第一个数据点

    # 重置索引
    data.reset_index(drop=True, inplace=True)
    data['Buy_Signal'] = (data['涨跌幅'] < 0.95 * data['Max_rate'])
    return data

def infer_dtype_convert(dtype):
    """
    Helper function to convert data types for memory efficiency.
    """
    if dtype == 'float64':
        return 'float16'
    elif dtype == 'int64':
        return 'int8'
    else:
        return dtype


def low_memory_load(file_path):
    """
    Load a file with memory-efficient data types.

    Args:
    file_path: Path to the file to load.

    Returns:
    A pandas DataFrame with float64 and int64 columns converted to float32 and int32.
    """
    # Read the first few rows to infer data type
    temp_df = pd.read_csv(file_path, nrows=100)

    # Create a dictionary of column names and their optimized data types
    dtype_dict = {col: infer_dtype_convert(str(temp_df[col].dtype)) for col in temp_df.columns}

    # Re-load the file with optimized data types
    df = pd.read_csv(file_path, dtype=dtype_dict)

    return df

def load_file_chunk(file_chunk, start_date='2018-01-01' ,end_date='2024-03-01'):
    """
    加载文件块的数据
    """
    try:
        chunk_data = [load_data(fname, start_date=start_date, end_date=end_date) for fname in file_chunk]
        return pd.concat(chunk_data)
    except Exception:
        # traceback.print_exc()
        return pd.DataFrame()

def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        traceback.print_exc()
        return {}

def load_data(file_path, start_date='2018-01-01', end_date='2024-01-01'):
    data = pd.read_csv(file_path, low_memory=False)
    name, code = parse_filename(file_path)
    if '时间' in data.columns:
        data = data.rename(columns={'时间': '日期'})
    data['日期'] = pd.to_datetime(data['日期'])
    data['名称'] = name
    data['代码'] = code
    data.sort_values(by='日期', ascending=True, inplace=True)
    # 过滤掉收盘价小于等于0的数据
    data = data[data['收盘'] > 0]

    # 查找并移除第一个日期，如果与其他日期不连续超过30天
    date_diff = data['日期'].diff(-1).abs()
    filtered_diff = date_diff[date_diff > pd.Timedelta(days=30)]

    # 过滤时间大于2024年的数据
    data = data[data['日期'] < pd.Timestamp(end_date)]
    data = data[data['日期'] > pd.Timestamp(start_date)]

    # 如果有大于30天的断层
    if not filtered_diff.empty:
        cutoff_index = filtered_diff.idxmax()
        if cutoff_index and cutoff_index != 0:
            data = data.loc[cutoff_index + 1:]  # 跳过第一个数据点

    # 重置索引
    data.reset_index(drop=True, inplace=True)
    data['Buy_Signal'] = (data['涨跌幅'] < 0.95 * data['Max_rate'])
    return data

def T_indicators(data):
    """
    操盘做T相关指标
    :param data:
    :return:
    """

    # 计算28日内最低价的最低值和28日内最高价的最高值
    min_low_28d = data['收盘'].rolling(window=28).min()
    max_high_28d = data['收盘'].rolling(window=28).max()

    # 计算趋势的第一部分
    trend_part = ((data['收盘'] - min_low_28d) / (max_high_28d - min_low_28d)) * 100

    # 计算2日[1日权重]移动平均的2日[1日权重]移动平均
    sma_2d_2 = trend_part.rolling(window=2).mean().rolling(window=2).mean()

    # 计算HXJZ和BZTD
    data['HXJZ'] = ((2 * data['收盘'] + data['最高'] + data['最低']) / 4).rolling(window=5).mean()
    data['BZTD'] = data['HXJZ'] * 89 / 100

    # 必涨条件
    data['必涨'] = (data['最低'] > data['BZTD'].shift(1)) & (data['最低'].shift(1) <= data['BZTD'].shift(1))

    # 计算未来哪一天的最高价会高于今天的收盘价
    data['Days_Until_Higher'] = data.shape[0]  # 初始化为最大值
    for i in range(len(data)):
        higher_prices = data.loc[i + 1:, '最高'] > data.loc[i, '收盘']
        if not higher_prices.empty:
            higher_days = higher_prices.idxmax()
            if higher_days:
                data.loc[i, 'Days_Until_Higher'] = higher_days - i

    # 其他指标计算
    data['Trend'] = sma_2d_2
    data['Trend_Rate'] = (data['Trend'].shift(1) - data['Trend']) / data['Trend']
    data['Q'] = (3 * data['收盘'] + data['最低'] + data['开盘'] + data['最高']) / 6
    data['操盘线'] = (26 * data['Q'] + data['Q'].shift(np.arange(1, 27)).mul(np.arange(25, -1, -1)).sum(axis=1)) / 351

    return data


def T_indicators_optimized(data):
    # 计算28日内最低价的最低值和28日内最高价的最高值
    min_low_28d = data['收盘'].rolling(window=28).min()
    max_high_28d = data['收盘'].rolling(window=28).max()

    # 计算趋势的第一部分
    trend_part = ((data['收盘'] - min_low_28d) / (max_high_28d - min_low_28d)) * 100

    # 计算2日[1日权重]移动平均的2日[1日权重]移动平均
    sma_2d_1 = trend_part.rolling(window=2).mean()
    sma_2d_2 = sma_2d_1.rolling(window=2).mean()

    # 计算HXJZ
    data['HXJZ'] = ((2 * data['收盘'] + data['最高'] + data['最低']) / 4).rolling(window=5).mean()

    # 计算BZTD
    data['BZTD'] = data['HXJZ'] * 89 / 100

    # 检查低点穿越BZTD的条件
    data['必涨'] = (data['最低'].shift(1) <= data['BZTD'].shift(1)) & (data['最低'] > data['BZTD'])

    # 将结果添加到data数据框中
    data['Trend'] = sma_2d_2
    data['Trend_Rate'] = (data['Trend'].shift(1) - data['Trend']) / data['Trend']

    # 计算Q值
    data['Q'] = (3 * data['收盘'] + data['最低'] + data['开盘'] + data['最高']) / 6

    # 使用滚动窗口和lambda函数来计算操盘线
    data['操盘线'] = data['Q'].rolling(window=27).apply(
        lambda x: (26 * x[-1] + sum([(26 - j) * x[j] for j in range(26)])) / 351, raw=True)

    return data


def get_indicators(data, is_debug=False):
    """
    为data计算各种指标
    :param data:
    :return:
    """
    if is_debug:
        # 初始化新的指标列
        data['Days_Until_Higher'] = data.shape[0]  # 使用None作为默认值，表示未找到满足条件的日期

        # 检查每一行，找出未来哪一天的最高价会高于今天的收盘价
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if data.loc[j, '最高'] > data.loc[i, '收盘']:
                    data.loc[i, 'Days_Until_Higher'] = j - i  # 计算天数
                    break  # 找到后就跳出内层循环

    T_indicators_optimized(data)

    # 计算20日支撑位和压力位
    data['Support'] = data['收盘'].rolling(window=20).min()
    data['Resistance'] = data['收盘'].rolling(window=20).max()

    # 计算到目前为止最低的收盘价
    data['LL'] = data['收盘'].expanding().min()

    # 计算到60天最低的收盘价
    data['LL_60'] = data['收盘'].rolling(window=60).min()

    # 计算SMA和EMA
    data['SMA'] = talib.SMA(data['收盘'], timeperiod=40)
    data['EMA'] = talib.EMA(data['收盘'], timeperiod=40)
    data['SMA_10'] = talib.SMA(data['收盘'], timeperiod=10)

    # 计算MACD
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['收盘'], fastperiod=12, slowperiod=26,
                                                                      signalperiod=9)
    data['BAR'] = (data['MACD'] - data['MACD_Signal']) * 2
    data['abs_BAR'] = abs(data['BAR'])

    # 计算RSI
    data['RSI'] = talib.RSI(data['收盘'], timeperiod=14)

    # 计算VWAP
    data['VWAP'] = np.cumsum(data['成交量'] * data['收盘']) / np.cumsum(data['成交量'])

    # 计算Bollinger Bands
    data['Bollinger_Upper'], data['Bollinger_Middle'], data['Bollinger_Lower'] = talib.BBANDS(data['收盘'],
                                                                                              timeperiod=20)

    # 计算Momentum
    data['Momentum'] = data['收盘'] - data['收盘'].shift(14)

    # 计算Stochastic Oscillator
    data['Stochastic_Oscillator'], data['Stochastic_Oscillator_Signal'] = talib.STOCH(data['最高'], data['最低'],
                                                                                      data['收盘'], fastk_period=14,
                                                                                      slowk_period=3, slowd_period=3)

    # 计算Williams %R
    data['WR10'] = abs(talib.WILLR(data['最高'], data['最低'], data['收盘'], timeperiod=5))

    # 计算ATR
    data['ATR'] = talib.ATR(data['最高'], data['最低'], data['收盘'], timeperiod=14)

    # 计算ADX
    data['ADX'] = talib.ADX(data['最高'], data['最低'], data['收盘'], timeperiod=14)

    # 计算CCI
    data['CCI'] = talib.CCI(data['最高'], data['最低'], data['收盘'], timeperiod=14)

    # 计算Parabolic SAR
    data['Parabolic_SAR'] = talib.SAR(data['最高'], data['最低'], acceleration=0.02, maximum=0.2)

    # 计算OBV
    data['OBV'] = talib.OBV(data['收盘'], data['成交量'])

    # 计算过去15日的最高收盘价
    data['Max_Close_Last_15_Days'] = data['收盘'].rolling(window=15).max()

    # 获取这些最大值的索引
    data['Max_Close_Idx'] = data['收盘'].where(data['收盘'] == data['Max_Close_Last_15_Days']).ffill()

    # 计算20日最小macd
    data['BAR_20d_low'] = data['BAR'].rolling(window=20).min()

    # 使用argsort来获取滚动窗口中最小值的索引
    min_idx = data['BAR'].rolling(window=20).apply(lambda x: np.argsort(x)[0] if len(x) == 20 else np.nan, raw=True)

    # 根据索引来计算全局的位置
    min_idx_global = min_idx + np.arange(len(data)) - 19

    # 先赋值NaN
    data['BAR_20d_low_price'] = np.nan

    # 更新滚动窗口中最小BAR对应的收盘价
    valid_indices = min_idx_global.dropna().astype(int)
    data.loc[valid_indices.index, 'BAR_20d_low_price'] = data.loc[valid_indices, '收盘'].values

    # 计算boll下轨和收盘价的差值
    data['Bollinger_Lower_cha'] = data['Bollinger_Lower'] - data['收盘']

    # 计算macd相邻差值
    data['macd_cha'] = data['BAR'] - data['BAR'].shift(1)
    data['macd_cha_rate'] = data['涨跌幅'] / data['macd_cha']
    data['macd_cha_shou_rate'] = (data['收盘'] - data['开盘']) / data['macd_cha']

    # # 删除NaN值
    # data.dropna(inplace=True)


def backtest_strategy(data, initial_capital=100000):
    """
    为data进行回测，买入条件为当有Buy_Signal时，且上一只股票已售出的情况，卖出条件为当收盘股价高于买入价时就卖出，如果最后的时间还没卖出，那么强制卖出
    ，返回买入卖出的时间，价格，收益，持有时间
    :param data:
    :param initial_capital:
    :return:
    """
    # 确保data有所需的列
    if 'back_Buy_Signal' not in data.columns:
        data['back_Buy_Signal'] = 0
    if 'back_Sell_Signal' not in data.columns:
        data['back_Sell_Signal'] = 0

    capital = initial_capital
    results = []  # 存储回测结果
    position = 0  # 当前持仓数量
    buy_price = None  # 最近一次买入价格
    buy_date = None  # 最近一次买入日期
    buy_index = None  # 最近一次买入的索引
    total_profit = 0

    for i in range(1, len(data)):
        if data['Buy_Signal'].iloc[i] == 1 and capital >= data['收盘'].iloc[i]:
            buy_price = data['收盘'].iloc[i]
            buy_date = data['日期'].iloc[i]
            buy_index = i  # 保存买入的索引
            stocks_to_buy = capital // buy_price  # 购买尽可能多的股票
            capital -= stocks_to_buy * buy_price
            position += stocks_to_buy
            data.at[i, 'back_Buy_Signal'] = 1

        elif position > 0 and data['收盘'].iloc[i] >= buy_price:
            sell_price = data['收盘'].iloc[i]
            sell_date = data['日期'].iloc[i]
            profit = (sell_price - buy_price) * position
            total_profit += profit
            growth_rate = ((sell_price - buy_price) / buy_price) * 100
            days_held = i - buy_index  # 使用索引差来计算天数
            capital += position * sell_price
            results.append([buy_date, buy_price, sell_date, sell_price, profit, total_profit, growth_rate, days_held])
            position = 0
            buy_price = None
            buy_date = None
            buy_index = None  # 重置买入索引
            data.at[i, 'back_Sell_Signal'] = 1

    # 如果在最后一个数据点仍然持有股票，则强制卖出
    if position > 0:
        sell_price = data['收盘'].iloc[-1]
        sell_date = data['日期'].iloc[-1]
        profit = (sell_price - buy_price) * position
        total_profit += profit
        growth_rate = ((sell_price - buy_price) / buy_price) * 100
        days_held = len(data) - 1 - buy_index
        results.append([buy_date, buy_price, sell_date, sell_price, profit, total_profit, growth_rate, days_held])
        data.at[-1, 'back_Sell_Signal'] = 1

    results_df = pd.DataFrame(results,
                              columns=['Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit', 'Total_Profit',
                                       'Growth Rate (%)',
                                       'Days Held'])

    return results_df


def backtest_strategy_highest_continue(data):
    """
    固定买入股票数量,且未买入时继续买入
    为data进行回测，买入条件为当有Buy_Signal时，且上一只股票已售出的情况，卖出条件为当收盘股价高于买入价时就卖出，如果最后的时间还没卖出，那么强制卖出
    ，返回买入卖出的时间，价格，收益，持有时间
    :param data:
    :param initial_capital:
    :return:
    """
    results = []
    position = 0
    total_spent = 0  # 记录总的花费
    avg_buy_price = 0  # 平均成本价
    buy_date = None
    buy_index = None
    total_profit = 0
    name = data['名称'].iloc[0]
    symbol = data['代码'].iloc[0]

    for i in range(1, len(data)):
        if data['Buy_Signal'].iloc[i] == 1:
            if position == 0:
                buy_date = data['日期'].iloc[i]
                buy_index = i

            current_price = data['收盘'].iloc[i]
            total_spent += current_price * 100
            position += 100
            avg_buy_price = total_spent / position

        elif position > 0 and data['最高'].iloc[i] > avg_buy_price:
            if data['最高'].iloc[i] == data['开盘'].iloc[i]:
                continue
            sell_price = data['最高'].iloc[i]
            sell_date = data['日期'].iloc[i]
            profit = (sell_price - avg_buy_price) * position
            total_profit += profit
            growth_rate = ((sell_price - avg_buy_price) / avg_buy_price) * 100
            days_held = i - buy_index
            results.append(
                [name, symbol, buy_date, avg_buy_price, sell_date, sell_price, profit, total_profit, growth_rate,
                 days_held])
            position = 0
            total_spent = 0
            avg_buy_price = 0
            buy_date = None
            buy_index = None

    if position > 0:
        sell_price = data['最高'].iloc[-1]
        sell_date = data['日期'].iloc[-1]
        profit = (sell_price - avg_buy_price) * position
        total_profit += profit
        growth_rate = ((sell_price - avg_buy_price) / avg_buy_price) * 100
        days_held = len(data) - 1 - buy_index
        results.append(
            [name, symbol, buy_date, avg_buy_price, sell_date, sell_price, profit, total_profit, growth_rate,
             days_held])

    results_df = pd.DataFrame(results,
                              columns=['名称', '代码', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit',
                                       'Total_Profit', 'Growth Rate (%)',
                                       'Days Held'])

    return results_df


def capture_target_data(data):
    """
    捕捉目标日期
    :param data:
    :return:
    """
    # Calculate rolling 10-day high
    data['10日最高'] = data['最高'].rolling(window=10).max().shift(1)  # Shifted to exclude today's high

    # Calculate drop percentage from the 10-day high and round to 2 decimal places
    data['跌幅'] = ((data['10日最高'] - data['收盘']) / data['10日最高'] * 100).round(2)

    # Identify days before a rise:
    # Where the stock drops more than 10% from the 10-day high,
    # it doesn't rise on that day (close <= open), and then rises the next day
    conditions = (data['跌幅'] > 10) & (data['收盘'] <= data['开盘']) & (
            data['收盘'].shift(-1) > data['开盘'].shift(-1))
    data['标记'] = np.where(conditions, 1, 0)

    # Drop the intermediate columns used for calculations
    data = data.drop(columns=['10日最高', '跌幅'])

    return data


def backtest_strategy_highest_buy_all(data):
    """
    每个买入信号都入手，然后找到相应的交易日
    为data进行回测，买入条件为当有Buy_Signal时，卖出条件为当最高股价高于买入价时就卖出
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

            # 找到满足卖出条件的日期
            j = i + 1
            while j < len(data) and data['最高'].iloc[j] <= buy_price:
                j += 1

            # 如果找到了满足卖出条件的日期
            if j < len(data):
                sell_price = data['最高'].iloc[j]
            else:
                # 如果没有找到，强制在最后一天卖出
                j = len(data) - 1
                sell_price = data['收盘'].iloc[j]
            if buy_price == 0:
                print(buy_price)
            sell_date = data['日期'].iloc[j]
            profit = (sell_price - buy_price) * 100  # 每次买入100股
            total_profit += profit
            growth_rate = ((sell_price - buy_price) / buy_price) * 100
            days_held = j - buy_index
            results.append([name, symbol, buy_date, buy_price, sell_date, sell_price, profit, total_profit, growth_rate,
                            days_held, i])

        i += 1

    results_df = pd.DataFrame(results,
                              columns=['名称', '代码', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit',
                                       'Total_Profit', 'Growth Rate (%)',
                                       'Days Held', 'Buy_Index'])

    return results_df

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
            results.append([name, symbol, buy_date, buy_price, sell_date, sell_price, profit, total_profit, total_cost,
                            days_held, i])

        i += 1

    results_df = pd.DataFrame(results,
                              columns=['名称', '代码', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit',
                                       'Total_Profit', 'total_cost',
                                       'Days Held', 'Buy_Index'])

    return results_df


def backtest_strategy_highest_fix(data):
    """
    固定买入股票数量
    为data进行回测，买入条件为当有Buy_Signal时，且上一只股票已售出的情况，卖出条件为当收盘股价高于买入价时就卖出，如果最后的时间还没卖出，那么强制卖出
    ，返回买入卖出的时间，价格，收益，持有时间
    :param data:
    :param initial_capital:
    :return:
    """
    results = []
    position = 0
    buy_price = None
    buy_date = None
    buy_index = None
    total_profit = 0
    name = data['名称'].iloc[0]
    symbol = data['代码'].iloc[0]
    for i in range(1, len(data)):
        # # 将data['日期'].iloc[i]转换为字符串
        # str_time = str(data['日期'].iloc[i])
        # # 判断str_time是否包含'1999-07-28'
        # if '1999-07-28' in str_time:
        #     print('1999-07-28')
        # if '1999-07-27' in str_time:
        #     print('1999-07-27')

        if data['Buy_Signal'].iloc[i] == 1 and position == 0:
            buy_price = data['收盘'].iloc[i]
            buy_date = data['日期'].iloc[i]
            buy_index = i
            position = 100  # 购买100股

        elif position > 0 and data['最高'].iloc[i] > buy_price:
            if data['最高'].iloc[i] == data['开盘'].iloc[i]:
                continue
            sell_price = data['最高'].iloc[i]
            sell_date = data['日期'].iloc[i]
            profit = (sell_price - buy_price) * position
            total_profit += profit
            growth_rate = ((sell_price - buy_price) / buy_price) * 100
            days_held = i - buy_index
            results.append([name, symbol, buy_date, buy_price, sell_date, sell_price, profit, total_profit, growth_rate,
                            days_held])
            position = 0
            buy_price = None
            buy_date = None
            buy_index = None

    if position > 0:
        sell_price = data['最高'].iloc[-1]
        sell_date = data['日期'].iloc[-1]
        profit = (sell_price - buy_price) * position
        total_profit += profit
        growth_rate = ((sell_price - buy_price) / buy_price) * 100
        days_held = len(data) - 1 - buy_index
        results.append(
            [name, symbol, buy_date, buy_price, sell_date, sell_price, profit, total_profit, growth_rate, days_held])

    results_df = pd.DataFrame(results,
                              columns=['名称', '代码', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit',
                                       'Total_Profit', 'Growth Rate (%)',
                                       'Days Held'])

    return results_df


def backtest_strategy_highest(data, initial_capital=100000):
    """
    为data进行回测，买入条件为当有Buy_Signal时，且上一只股票已售出的情况，卖出条件为当最高股价高于买入价时就卖出，如果最后的时间还没卖出，那么强制卖出
    ，返回买入卖出的时间，价格，收益，持有时间
    :param data:
    :param initial_capital:
    :return:
    """

    capital = initial_capital
    results = []  # 存储回测结果
    position = 0  # 当前持仓数量
    buy_price = None  # 最近一次买入价格
    buy_date = None  # 最近一次买入日期
    buy_index = None  # 最近一次买入的索引
    total_profit = 0
    name = data['名称'].iloc[0]
    symbol = data['代码'].iloc[0]
    for i in range(1, len(data)):
        if data['Buy_Signal'].iloc[i] == 1 and capital >= data['收盘'].iloc[i] and position == 0:
            buy_price = data['收盘'].iloc[i]
            buy_date = data['日期'].iloc[i]
            buy_index = i  # 保存买入的索引
            stocks_to_buy = capital // buy_price  # 购买尽可能多的股票
            capital -= stocks_to_buy * buy_price
            position += stocks_to_buy

        elif position > 0 and data['最高'].iloc[i] > buy_price:
            if data['最高'].iloc[i] == data['开盘'].iloc[i]:
                continue  # 一直下跌形态是无法成交的
            sell_price = data['最高'].iloc[i]
            sell_date = data['日期'].iloc[i]
            profit = (sell_price - buy_price) * position
            total_profit += profit
            growth_rate = ((sell_price - buy_price) / buy_price) * 100
            days_held = i - buy_index  # 使用索引差来计算天数
            capital += position * sell_price
            results.append([name, symbol, buy_date, buy_price, sell_date, sell_price, profit, total_profit, growth_rate,
                            days_held])
            position = 0
            buy_price = None
            buy_date = None
            buy_index = None  # 重置买入索引

    # 如果在最后一个数据点仍然持有股票，则强制卖出
    if position > 0:
        sell_price = data['最高'].iloc[-1]
        sell_date = data['日期'].iloc[-1]
        profit = (sell_price - buy_price) * position
        total_profit += profit
        growth_rate = ((sell_price - buy_price) / buy_price) * 100
        days_held = len(data) - 1 - buy_index
        results.append(
            [name, symbol, buy_date, buy_price, sell_date, sell_price, profit, total_profit, growth_rate, days_held])

    results_df = pd.DataFrame(results,
                              columns=['名称', '代码', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit',
                                       'Total_Profit', 'Growth Rate (%)',
                                       'Days Held'])

    return results_df


def gen_buy_signal_one(data, down_rate=0.1):
    """
    为data生成相应的买入信号
    默认换手率都得高于0.5
    :param data:
    :return:
    """
    change_rate = 0.5
    max_down_rate = 0.9
    max_down_value = data['Max_rate'].iloc[-1]  # 最大跌幅

    # 策略一
    # 今日macd创20日新低，
    # 今天是红柱,
    # 股价是低位的（小于10日均线 同时小于40日均线）,
    # 并且股价较15日最高下降超down_rate，
    # 昨日涨跌幅小于0，并且今日股价下跌
    # 昨日跌幅小于跌停的10%
    # 尝试：
    # 1.增加对跌幅的限制，有些是红柱，但是跌幅超过5 2.股价不超过10
    data['Buy_Signal'] = (data['BAR'].rolling(window=20).min() == data['BAR']) & \
                         (data['BAR'].shift(1) < 0) & \
                         (data['收盘'] > data['开盘']) & \
                         (data['收盘'] < data['SMA_10']) & \
                         (data['收盘'] < data['SMA']) & \
                         (data['涨跌幅'].shift(1) <= 0) & \
                         (data['涨跌幅'] <= 0) & \
                         (data['涨跌幅'].shift(1) >= -max_down_rate * max_down_value) & \
                         (data['换手率'] > change_rate) & \
                         ((data['收盘'].rolling(window=15).max() - data['收盘']) / data['收盘'] > down_rate)


def gen_buy_signal_three(data):
    """
    实体大于昨日实体 20%
    今日是下跌
    昨日也是下跌
    :param data:
    :return:
    """
    more_than_rate = 1.5
    data['Buy_Signal'] = (abs(data['开盘'] - data['收盘']) > more_than_rate * abs(
        data['开盘'].shift(1) - data['收盘'].shift(1))) & \
                         (data['涨跌幅'] < 0) & \
                         (data['涨跌幅'].shift(1) < 0)


def gen_buy_signal_four(data):
    """
    boll线3天内接近下轨，且今日阳线
    :param data:
    :return:
    """
    change_rate = 0.5
    boll_windows = 3
    # 近期boll突破下轨
    # 今日收阳
    # 昨日收阴
    # 昨日下跌
    # 昨日不跌停
    data['Buy_Signal'] = (data['Bollinger_Lower_cha'].rolling(window=boll_windows).max() > 0) & \
                         (data['收盘'] > data['开盘']) & \
                         (data['涨跌幅'].shift(1) < 0) & \
                         (data['收盘'].shift(1) <= data['开盘'].shift(1))


def gen_buy_signal_five(data):
    """
    macd创新低
    boll突破下轨
    默认换手率都得高于0.5
    :param data:
    :return:
    """
    change_rate = 0.5
    boll_windows = 3
    macd_windows = 5

    # macd创新低
    # boll突破下轨
    data['Buy_Signal'] = (data['Bollinger_Lower_cha'].rolling(window=boll_windows).max() > 0) & \
                         (data['收盘'] > data['开盘']) & \
                         (data['换手率'] > change_rate) & \
                         (data['BAR'].rolling(window=macd_windows).min() == data['BAR'])


def gen_buy_signal_six(data):
    """
    为data生成相应的买入信号
    默认换手率都得高于0.5
    :param data:
    :return:
    """
    change_rate = 0.5
    # 策略三
    # 昨日macd创20日新低，
    # 股价较20日最高价下跌25%
    # 底部逗留连续3天(任意两天差异小于0.2%)
    data['Buy_Signal'] = True


def gen_buy_signal_seven(data):
    """
    为data生成相应的买入信号
    默认换手率都得高于0.5
    :param data:
    :return:
    """
    down_rate = 0.1
    change_rate = 0.5
    deal_number_rate = 1
    # 策略七
    # 成交量叫前日减少一半
    data['Buy_Signal'] = (data['换手率'] > change_rate) & \
                         ((data['收盘'].rolling(window=15).max() - data['收盘']) / data['收盘'] < down_rate) & \
                         ((data['成交量'].shift(1) - data['成交量']) / data['成交量'] > deal_number_rate)


def gen_buy_signal_mix_one_seven(data):
    """
    为data生成相应的买入信号
    默认换手率都得高于0.5
    :param data:
    :return:
    """
    down_rate = 0.1
    change_rate = 0.5
    deal_number_rate = 1
    # 策略七
    # 成交量叫前日减少一半

    change_rate = 0.5
    max_down_rate = 0.9
    max_down_value = data['Max_rate'].iloc[-1]  # 最大跌幅

    # 策略一
    # 今日macd创20日新低，
    # 今天是红柱,
    # 股价是低位的（小于10日均线 同时小于40日均线）,
    # 并且股价较15日最高下降超down_rate，
    # 昨日涨跌幅小于0，并且今日股价下跌
    # 昨日跌幅小于跌停的10%
    # 昨日未跌停
    # 尝试：
    # 1.增加对跌幅的限制，有些是红柱，但是跌幅超过5
    data['Buy_Signal'] = (data['BAR'].rolling(window=20).min() == data['BAR']) & \
                         (data['BAR'].shift(1) < 0) & \
                         (data['收盘'] > data['开盘']) & \
                         (data['收盘'] < data['SMA_10']) & \
                         (data['收盘'] < data['SMA']) & \
                         (data['涨跌幅'].shift(1) <= 0) & \
                         (data['涨跌幅'] <= 0) & \
                         (data['涨跌幅'].shift(1) >= -max_down_rate * max_down_value) & \
                         (data['换手率'] > change_rate) & \
                         ((data['收盘'].rolling(window=15).max() - data['收盘']) / data['收盘'] > down_rate) & \
                         (data['换手率'] > change_rate) & \
                         ((data['成交量'].shift(1) - data['成交量']) / data['成交量'] > deal_number_rate)


def gen_buy_signal_eight(data):
    """
    捕捉低位且前一日macd创新低，今日macd增加
    示例:
        603318 20230321
    默认换手率都得高于0.5
    :param data:
    :return:
    """
    change_rate = 0.5
    max_down_rate = 0.9
    max_down_value = data['Max_rate'].iloc[-1]  # 最大跌幅
    macd_rate = 0.97
    down_rate = 0.1
    # 策略八
    # 昨日macd创20日新低（3% 误差）
    # 今日macd增加
    # 股价较10日最高价下跌超10%

    # 待尝试:
    # macd增加幅度小于20
    # 考虑融合周数据
    # 且昨日股价要小于macd新低的股价
    # 今天是红柱,
    # 股价是低位的（小于10日均线 同时小于40日均线）,
    # 并且股价较15日最高下降超down_rate，
    # 昨日涨跌幅小于0
    # 昨日跌幅小于跌停的10%
    # 昨日未跌停
    data['Buy_Signal'] = (data['BAR'].rolling(window=20).min() * macd_rate >= data['BAR'].shift(1)) & \
                         (data['换手率'] > change_rate) & \
                         ((data['收盘'].rolling(window=10).max() - data['收盘']) / data['收盘'] > down_rate)


def gen_buy_signal_weekly_eight(data):
    """
    产生周维度的买入信号
    macd创20日新低

    默认换手率都得高于0.5
    :param data:
    :return:
    """
    change_rate = 0.5
    max_down_rate = 0.9
    max_down_value = data['Max_rate'].iloc[-1]  # 最大跌幅
    macd_rate = 0.97
    down_rate = 0.1
    shuairuo_rate = 1.2
    min_shuairuo_macd = 0.01
    # 策略八
    # 今日macd创20日新低
    # macd下降趋势减小 今天的减少量小于昨天的减少量

    # 待尝试:
    #

    # data['Buy_Signal'] = (data['BAR'] <= 0) & \
    #                       (data['BAR'].rolling(window=5).min() == data['BAR'])  & \
    #                      (data['BAR'].rolling(window=5).max() < abs(data['BAR'])) & \
    #                      ((data['BAR'] - data['BAR'].shift(1)) * shuairuo_rate > (data['BAR'].shift(1) - data['BAR'].shift(2))) & \
    #                      (data['收盘'] < data['SMA'])

    data['Buy_Signal'] = (data['BAR'] > 0) & \
                         (data['BAR'].rolling(window=5).min() == data['BAR']) & \
                         (data['涨跌幅'].shift(1) < 0) & \
                         (data['收盘'] > data['SMA']) & \
                         ((data['BAR'] - data['BAR'].shift(1)) * shuairuo_rate > (
                                 data['BAR'].shift(1) - data['BAR'].shift(2))) & \
                         (((data['BAR'] - data['BAR'].shift(1)) - (
                                 data['BAR'].shift(1) - data['BAR'].shift(2))) >= min_shuairuo_macd)


def mix_small_period_and_big_period_data(small_period_data, big_period_data):
    """
    综合daily_data和weekly_data的数据:
        找到daily_data中Buy_Signal为true的记录，然后在weekly_data找到daily_data日期所对应的数据(相应daily_data日期往后5天内遇到的第一个weekly_data的日期就是相应的数据)，如果weekly_data的Buy_Signal不为True那么就将该daily_data的Buy_Signal设置为False
    :param small_period_data: DataFrame with daily data. Expects a '日期' column with date values and a 'Buy_Signal' column with boolean values.
    :param big_period_data: DataFrame with weekly data. Expects a '日期' column with date values and a 'Buy_Signal' column with boolean values.
    :return: Adjusted daily_data DataFrame.
    """

    # Convert the '日期' columns to datetime objects for easier comparison
    small_period_data['日期'] = pd.to_datetime(small_period_data['日期'])
    big_period_data['日期'] = pd.to_datetime(big_period_data['日期'])

    # Find indices in daily_data where Buy_Signal is True
    buy_signal_indices = small_period_data[small_period_data['Buy_Signal']].index.tolist()

    for idx in buy_signal_indices:
        buy_date = small_period_data.at[idx, '日期']
        # # # 将data['日期'].iloc[i]转换为字符串
        # str_time = str(buy_date)
        # # 判断str_time是否包含'1999-07-28'
        # if '2023-10-20' in str_time:
        #     print('2022-04-08')

        # Find the corresponding date in weekly_data within the next 5 days
        matching_weekly_rows = big_period_data[(big_period_data['日期'] >= buy_date) &
                                               (big_period_data['日期'] <= buy_date + pd.Timedelta(days=30))]

        if not matching_weekly_rows.empty:
            # Take the first matching row
            weekly_buy_signal = matching_weekly_rows.iloc[0]['Buy_Signal']

            if not weekly_buy_signal:
                # Set Buy_Signal in daily_data to False if it's not True in weekly_data
                small_period_data.at[idx, 'Buy_Signal'] = False
        else:
            small_period_data.at[idx, 'Buy_Signal'] = False

    return small_period_data


def gen_buy_signal_nine(data):
    """
    捕捉那种低位，且macd没怎么减少，但是股价一直下跌的情况
    当实体变小就买入
    默认换手率都得高于0.5
    :param data:
    :return:
    """
    change_rate = 0.5
    max_down_rate = 0.9
    max_down_value = data['Max_rate'].iloc[-1]  # 最大跌幅
    macd_rate = 0.9
    down_rate = 0.1
    # 策略一
    # 昨日macd创20日新低（3% 误差），且昨日股价要小于macd新低的股价
    # 今天是红柱,
    # 股价是低位的（小于10日均线 同时小于40日均线）,
    # 并且股价较15日最高下降超down_rate，
    # 昨日涨跌幅小于0
    # 昨日跌幅小于跌停的10%
    # 昨日未跌停
    buy_signal_1 = (data['BAR'].rolling(window=20).min() * macd_rate >= data['BAR'].shift(1)) & \
                   (data['BAR'].shift(1) < 0) & \
                   (data['收盘'] <= data['BAR_20d_low_price']) & \
                   (data['收盘'] > data['开盘']) & \
                   (data['收盘'] < data['SMA_10']) & \
                   (data['收盘'] < data['SMA']) & \
                   (data['涨跌幅'].shift(1) <= 0) & \
                   (data['涨跌幅'] <= 0) & \
                   (data['涨跌幅'].shift(1) >= -max_down_rate * max_down_value) & \
                   (data['换手率'] > change_rate) & \
                   ((data['收盘'].rolling(window=15).max() - data['收盘']) / data['收盘'] > down_rate)

    # buy_signal_2 = (data['BAR'].rolling(window=20).min() * macd_rate >= data['BAR']) & \
    #                (data['收盘'] <= data['BAR_20d_low_price']) & \
    #                (data['BAR'] < 0) & \
    #                (data['收盘'] < data['SMA_10']) & \
    #                (abs(data['收盘'] - data['开盘']) < abs(data['收盘'].shift(1) - data['开盘'].shift(1))) & \
    #                (data['涨跌幅'].shift(1) <= 0) & \
    #                (data['涨跌幅'].shift(1) >= -max_down_rate * max_down_value) & \
    #                (data['换手率'] > change_rate) & \
    #                ((data['收盘'].rolling(window=15).max() - data['收盘']) / data['收盘'] > down_rate)

    # 结合两个买入信号
    data['Buy_Signal'] = buy_signal_1


def gen_buy_signal_ten(data):
    """
    捕捉今天最高价小于昨天收盘价，wr大于95，今日最低价大于等于昨日最低价
    示例:

    默认换手率都得高于0.5
    :param data:
    :return:
    """
    change_rate = 0.5
    wr_threshold = 95
    down_rate = 0.1
    # 今天最高价小于昨天收盘价
    # wr大于95
    # 今日最低价大于等于昨日最低价
    data['Buy_Signal'] = (data['收盘'].shift(1) >= data['最高']) & \
                         (data['WR10'] >= wr_threshold) & \
                         (data['换手率'] > change_rate) & \
                         ((data['最低'] >= data['最低'].shift(1)))


def gen_buy_signal_eleven(data):
    """
    捕捉成交量减半的
    示例:

    默认换手率都得高于0.5
    :param data:
    :return:
    """
    deal_number_rate = 1
    # 今天最高价小于昨天收盘价
    # wr大于95
    # 今日最低价大于等于昨日最低价
    data['Buy_Signal'] = (((data['成交量'].shift(1) + data['成交量'].shift(2)) / 2 - data['成交量']) / data[
        '成交量'] > deal_number_rate)
    # data['Buy_Signal'] = ((data['成交量'].shift(1) - data['成交量']) / data[
    #     '成交量'] > deal_number_rate)


def gen_buy_signal_week(data):
    """
    旨在捕捉下跌的反弹
    步骤:
        先找出下跌反弹的日期，然后进行归纳总结特征
        整理出相应的规则，然后进行回测
    :param data:
    :return:
    """
    pass


def show_image(data, results_df, threshold=10):
    """
    展示数据
    :param data:
    :return:
    """

    # 创建一个带有滑动条的子图
    fig = make_subplots(rows=2, row_heights=[0.85, 0.15], shared_xaxes=True, vertical_spacing=0.02)

    # 添加收盘价线
    fig.add_trace(go.Scatter(x=data['日期'], y=data['收盘'], mode='lines', name='Close Price'), row=1, col=1)

    results_df = results_df[results_df['Days Held'] > threshold]
    # 添加买入卖出标记
    buy_dates = results_df['Buy Date'].tolist()
    buy_prices = results_df['Buy Price'].tolist()
    sell_dates = results_df['Sell Date'].tolist()
    sell_prices = results_df['Sell Price'].tolist()

    size_values = np.clip(results_df['Days Held'], 10, 100)  # 将大小限制在10到100之间
    hover_text = ['Buy Date: {} | Hold: {} days'.format(buy_date, days) for buy_date, days in
                  zip(buy_dates, results_df['Days Held'])]

    fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy Signal',
                             marker=dict(color='green', size=size_values, sizemode='diameter', symbol='triangle-up'),
                             text=hover_text, hoverinfo='text+y+x'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell Signal',
                             marker=dict(color='red', size=size_values, sizemode='diameter', symbol='triangle-down'),
                             text=hover_text, hoverinfo='text+y+x'), row=1, col=1)

    # 设置标题和坐标轴标签
    fig.update_layout(title='Stock Price with Buy/Sell Signals')

    # 设置日期的格式并限制一个页面显示的数据数量为40个
    fig.update_xaxes(range=[data['日期'].iloc[0], data['日期'].iloc[40]], tickformat="%Y-%m-%d", row=1, col=1)
    fig.update_xaxes(range=[data['日期'].iloc[0], data['日期'].iloc[40]], tickformat="%Y-%m-%d", row=2, col=1)

    # 设置滑动条
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type='date'))

    # 显示图形
    fig.show()


def get_buy_signal(file):
    """
    为file生成相应的买入信号
    :param file:
    :return:
    """
    data = load_data(file)
    get_indicators(data)
    gen_buy_signal_one(data)
    return data


def find_buy_signal(file_path, target_time=None):
    """
    遍历file_path下的说有数据，并读取，应用策略，找到target_time有买入信号的symbol
    :param file_path:
    :param target_time:
    :return:
    """
    if target_time is None:
        target_time = datetime.datetime.now().strftime('%Y-%m-%d')
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            try:
                fullname = os.path.join(root, f)
                stock_data_df = get_buy_signal(fullname)
                stock_data_df['日期'] = stock_data_df['日期'].astype(str)
                filtered_row = stock_data_df[
                    (stock_data_df['Buy_Signal'] == True) & (stock_data_df['日期'].str.startswith(target_time))]
                if filtered_row.shape[0] > 0:
                    print(f)
                    print(filtered_row)
                    print('\n')
            except Exception as e:
                print(fullname)


def calculate_change(data):
    """Calculate the percentage change in closing price."""
    data['涨跌幅度'] = data['收盘'].pct_change() * 100
    return data


def create_hovertext(data):
    """Generate the hover text for the K-line chart."""
    hovertext = data['日期'] + '<br>开盘: ' + data['开盘'].astype(str) + \
                '<br>收盘: ' + data['收盘'].astype(str) + '<br>最高: ' + data['最高'].astype(str) + \
                '<br>最低: ' + data['最低'].astype(str) + '<br>涨跌幅度: ' + data['涨跌幅'].astype(str) + "%" + \
                '<br>成交量: ' + data['成交量'].astype(str)
    return hovertext


def highlight_increase_days(fig, data):
    """Highlight the days before the rise in the K-line chart."""
    indices = data[data['Buy_Signal'] == 1].index
    for idx in indices:
        fig.add_annotation(
            x=data.loc[idx, '日期'],
            y=data.loc[idx, '最低'],
            text="↑",
            showarrow=True,
            arrowhead=4,
            ax=0,
            ay=40,
            bgcolor="red",
            arrowcolor="red"
        )
    return fig


def show_k(data, results_df, threshold=10):
    # Convert the date column to string
    data['日期'] = data['日期'].astype(str)
    results_df['Buy Date'] = results_df['Buy Date'].astype(str)
    results_df['Sell Date'] = results_df['Sell Date'].astype(str)

    # Calculate the change in closing price
    data = calculate_change(data)

    # Generate the hover text for the K-line chart
    hovertext = create_hovertext(data)

    # Create the K-line chart
    fig = go.Figure(data=[go.Candlestick(x=data['日期'],
                                         open=data['开盘'],
                                         high=data['最高'],
                                         low=data['最低'],
                                         close=data['收盘'],
                                         hovertext=hovertext,
                                         hoverinfo="text",
                                         increasing_line_color='red',
                                         decreasing_line_color='green'
                                         )])

    # Highlight the days before the rise
    fig = highlight_increase_days(fig, data)

    results_df = results_df[results_df['Days Held'] > 5]
    # Add buy and sell markers
    buy_dates = results_df['Buy Date'].tolist()
    buy_prices = results_df['Buy Price'].tolist()
    sell_dates = results_df['Sell Date'].tolist()
    sell_prices = results_df['Sell Price'].tolist()

    size_values = np.clip(results_df['Days Held'], 10, 100)  # 将大小限制在10到100之间
    hover_text_buy = ['Buy Date: {} | Hold: {} days'.format(buy_date, days) for buy_date, days in
                      zip(buy_dates, results_df['Days Held'])]

    hover_text_sell = ['Sell Date: {} after holding for {} days'.format(sell_date, days) for sell_date, days in
                       zip(sell_dates, results_df['Days Held'])]

    fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy Signal',
                             marker=dict(color='green', size=size_values, sizemode='diameter', symbol='triangle-up'),
                             text=hover_text_buy, hoverinfo='text+y+x'))
    fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell Signal',
                             marker=dict(color='red', size=size_values, sizemode='diameter', symbol='triangle-down'),
                             text=hover_text_sell, hoverinfo='text+y+x'))

    start_idx = max(0, len(data) - 40)
    end_idx = len(data) - 1

    fig.update_layout(title='K线图',
                      xaxis_title='日期',
                      yaxis_title='价格',
                      xaxis_rangeslider_visible=True,
                      xaxis_type='category',
                      yaxis_fixedrange=False,
                      bargap=0.02,
                      xaxis_range=[start_idx, end_idx])

    fig.update_layout(xaxis_rangeslider_visible=True)
    fig['layout']['xaxis']['rangeslider']['yaxis']['rangemode'] = 'auto'
    fig['layout']['xaxis']['rangeslider']['thickness'] = 0.05

    # Show the plot
    fig.show()

if __name__ == '__main__':
    print(get_sell_price(1))
    print(get_sell_price(2))
    print(get_sell_price(3))
    print(get_sell_price(4))
    print(get_sell_price(5))