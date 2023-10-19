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
import os
import sys
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import talib
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from plotly.subplots import make_subplots


def parse_filename(file_path):
    # 提取文件名（不包括扩展名）
    base_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]

    # 以'_'分割文件名以获取名称和编号
    stock_name, stock_code = file_name_without_ext.split('_')

    return stock_name, stock_code

def load_data(file_path):
    """
    加载数据
    :param file_path:
    :return:
    """
    data = pd.read_csv(file_path)
    name, code = parse_filename(file_path)
    data['日期'] = pd.to_datetime(data['日期'])
    data['名称'] = name
    data['代码'] = code
    # 判断name是否包含st，不区分大小写，如果包含，那么Max_rate为5%，否则为10%
    data['Max_rate'] = data['名称'].str.contains('st', case=False).map({True: 5, False: 10})
    return data


def get_indicators(data):
    """
    为data计算各种指标
    :param data:
    :return:
    """
    # 计算20日支撑位和压力位
    data['Support'] = data['收盘'].rolling(window=20).min()
    data['Resistance'] = data['收盘'].rolling(window=20).max()

    # 计算SMA和EMA
    data['SMA'] = talib.SMA(data['收盘'], timeperiod=40)
    data['EMA'] = talib.EMA(data['收盘'], timeperiod=40)
    data['SMA_10'] = talib.SMA(data['收盘'], timeperiod=10)

    # 计算MACD
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['收盘'], fastperiod=12, slowperiod=26,
                                                                      signalperiod=9)
    data['BAR'] = (data['MACD'] - data['MACD_Signal']) * 2

    # 计算RSI
    data['RSI'] = talib.RSI(data['收盘'], timeperiod=14)

    # 计算VWAP
    data['VWAP'] = np.cumsum(data['成交量'] * data['收盘']) / np.cumsum(data['成交量'])

    # 计算Bollinger Bands
    data['Bollinger_Upper'], data['Bollinger_Middle'], data['Bollinger_Lower'] = talib.BBANDS(data['收盘'], timeperiod=20)

    # 计算Momentum
    data['Momentum'] = data['收盘'] - data['收盘'].shift(14)

    # 计算Stochastic Oscillator
    data['Stochastic_Oscillator'], data['Stochastic_Oscillator_Signal'] = talib.STOCH(data['最高'], data['最低'],
                                                                                      data['收盘'], fastk_period=14,
                                                                                      slowk_period=3, slowd_period=3)

    # 计算Williams %R
    data['Williams_%R'] = talib.WILLR(data['最高'], data['最低'], data['收盘'], timeperiod=14)

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

    # 删除NaN值
    data.dropna(inplace=True)


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
        # 将data['日期'].iloc[i]转换为字符串
        str_time = str(data['日期'].iloc[i])
        # 判断str_time是否包含'1999-07-28'
        if '1999-07-28' in str_time:
            print('1999-07-28')
        if '1999-07-27' in str_time:
            print('1999-07-27')
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
                         ((data['收盘'].rolling(window=15).max() - data['收盘']) / data['收盘'] > down_rate)


def gen_buy_signal_two(data):
    """
    为data生成相应的买入信号
    默认换手率都得高于0.5
    :param data:
    :return:
    """
    change_rate = 0.5
    # 示例情形：1.403729 20180209
    # 策略二
    # 今日macd创20日新低
    # 今天股价有所增加,股价是低位的（小于10日均线）
    # 并且股价较20日最高下降超20%，
    # macd 减少值变小
    # 待尝试:1.涨幅不应该大于3 2.这个涨幅较开盘价格来算会不会好一点 2.前一日不能涨
    data['Buy_Signal'] = True


def gen_buy_signal_three(data):
    """
    为data生成相应的买入信号
    默认换手率都得高于0.5
    :param data:
    :return:
    """
    change_rate = 0.5
    # 策略三
    # 昨日macd创20日新低，
    # 今天macd有所增加,
    # 股价是降低了的（小于10日均线）
    # 回测记录：(东方电子 116833)
    data['Buy_Signal'] = True


def gen_buy_signal_four(data):
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
    # 今日跌幅小于昨日跌幅
    data['Buy_Signal'] = True


def gen_buy_signal_five(data):
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
