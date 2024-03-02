# -- coding: utf-8 --
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
import numpy as np

from StrategyExecutor.common import load_data
from itertools import combinations
import talib

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
            change_col_name = f'{key}_涨跌幅_{period}'
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
            col_name = f'{key1}_{key2}_diff_相较于_{key}_振幅'
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
    data['Stochastic_Oscillator_K'], data['Stochastic_Oscillator_D'] = talib.STOCH(
        data['最高'], data['最低'], data['收盘'],
        fastk_period=14, slowk_period=3, slowd_period=3
    )

    # 计算威廉姆斯%R指标，反映市场超买或超卖情况
    data['Williams_R'] = talib.WILLR(data['最高'], data['最低'], data['收盘'], timeperiod=14)

    # 计算平均真实范围（Average True Range，ATR），反映市场波动性
    data['ATR_14'] = talib.ATR(data['最高'], data['最低'], data['收盘'], timeperiod=14)

    # 计算平均方向指标（Average Directional Index，ADX），评估趋势强度
    data['ADX_14'] = talib.ADX(data['最高'], data['最低'], data['收盘'], timeperiod=14)

    # 计算商品渠道指数（Commodity Channel Index，CCI），识别新趋势或警告极端条件
    data['CCI_14'] = talib.CCI(data['最高'], data['最低'], data['收盘'], timeperiod=14)

    # 计算抛物线转向指标（Parabolic SAR），识别市场趋势
    data['Parabolic_SAR'] = talib.SAR(data['最高'], data['最低'], acceleration=0.02, maximum=0.2)

    # 计算能量潮（On Balance Volume，OBV），通过成交量变化预测股价走势
    data['OBV'] = talib.OBV(data['收盘'], data['成交量'])

    # 可以考虑添加的其他重要技术指标
    # 计算相对强弱指数（Relative Strength Index，RSI），评估价格走势的速度和变化，以识别超买或超卖条件
    data['RSI'] = talib.RSI(data['收盘'], timeperiod=14)

    # 计算移动平均收敛散度（Moving Average Convergence Divergence，MACD）
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(
        data['收盘'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    data['BAR'] = (data['MACD'] - data['MACD_signal']) * 2


def calculate_trend_changes(data, key_list):
    """
    计算指定列的连续上涨或下跌天数，并标记趋势由涨转跌及由跌转涨的点。

    参数:
    - data: DataFrame，包含市场数据。
    - key_list: list of str，包含需要计算连续上涨或下跌天数和趋势变化的列名。

    返回值:
    - 修改原始DataFrame，为每个指定列添加四个新列，表示连续上涨天数、连续下跌天数及趋势转变点。
    """
    for key in key_list:
        daily_change = data[key].diff()
        rise_fall_signal = np.sign(daily_change)
        direction_change = rise_fall_signal.diff().ne(0)
        groups = direction_change.cumsum()
        consecutive_counts = rise_fall_signal.groupby(groups).cumsum().abs()

        data[f'{key}_连续上涨天数'] = np.where(rise_fall_signal > 0, consecutive_counts, 0)
        data[f'{key}_连续下跌天数'] = np.where(rise_fall_signal < 0, consecutive_counts, 0)

        # 标记由跌转涨和由涨转跌的点
        data[f'{key}_由跌转涨'] = ((rise_fall_signal > 0) & (data[f'{key}_连续下跌天数'].shift(1) == 1)).astype(int)
        data[f'{key}_由涨转跌'] = ((rise_fall_signal < 0) & (data[f'{key}_连续上涨天数'].shift(1) == 1)).astype(int)


def calculate_rolling_stats(data, key_list, windows=[3, 5, 10]):
    """
    计算指定列的3天、5天、10天滑动平均值，以及3天、5天、10天内的极大值和极小值。

    参数:
    - data: DataFrame，包含市场数据。
    - key_list: list of str，包含需要计算滑动统计数据的列名。

    返回值:
    - 无，函数将直接在传入的DataFrame中添加新列，每列代表一个滑动统计量。
    """
    for key in key_list:
        for window in windows:
            # 计算滑动平均值
            data[f'{key}_{window}日均值'] = data[key].rolling(window=window).mean()
            # 计算滑动窗口内的极大值
            data[f'{key}_{window}日最大值'] = data[key].rolling(window=window).max()
            # 计算滑动窗口内的极小值
            data[f'{key}_{window}日最小值'] = data[key].rolling(window=window).min()


def calculate_ratio_and_dynamic_frequency(data, key_list, ratio_windows=[5, 10, 20], frequency_windows=[30, 60, 120]):
    """
    计算指定列在不同滑动窗口内的值分布占比，并动态计算频率。

    参数:
    - data: DataFrame，包含市场数据。
    - key_list: list of str，需要计算统计量的列名。
    - ratio_windows: list of int，定义计算占比的滑动窗口的大小。
    - frequency_windows: list of int，定义计算频率的滑动窗口的大小。

    返回值:
    - 无，函数将直接在传入的DataFrame中添加新列，每列代表一个统计量。
    """
    for key in key_list:
        # 计算占比
        for window in ratio_windows:
            min_col = data[key].rolling(window=window).min()
            max_col = data[key].rolling(window=window).max()
            ratio_name = f'{key}_{window}日占比'
            data[ratio_name] = (data[key] - min_col) / (max_col - min_col)
            data[ratio_name].fillna(0, inplace=True)  # 处理NaN和除以0的情况

        for key in key_list:
            for window in frequency_windows:
                ratio_name = f'{key}_{window}日频率_占比'
                # 在初始化占比列时明确指定数据类型为float
                data[ratio_name] = 0.0  # 使用0.0代替0强制列为浮点类型

                for idx in range(window, len(data) + 1):
                    window_slice = data[key][idx - window:idx]
                    current_value = window_slice.iloc[-1]
                    min_val = window_slice.min()
                    max_val = window_slice.max()

                    if max_val != min_val:  # 避免除以零
                        ratio = (current_value - min_val) / (max_val - min_val)
                    else:
                        ratio = 1.0  # 如果窗口期内所有值相同，则占比设为1

                    # 使用astype(float)确保赋值时的数据类型一致性
                    data.at[data.index[idx - 1], ratio_name] = ratio.astype(float)


def get_data_feature(data):
    """
    为data生成相应的feature，data包含开盘，收盘，最高，最低，成交量，日期，代码，换手率
    :param data:
    :return:
    """
    # 计算收盘, 开盘, 最高, 最低在[1,3,5]周期的涨跌幅
    calculate_ratio_and_dynamic_frequency(data, ['收盘', '开盘', '最高', '最低'])
    calculate_change_percentages(data, [1, 3, 5], ['收盘', '开盘', '最高', '最低'])
    calculate_relative_amplitude(data, ['收盘', '开盘', '最高', '最低'])
    calculate_technical_indicators(data)
    calculate_trend_changes(data, ['收盘', '开盘', '最高', '最低'])
    calculate_rolling_stats(data, ['收盘', '开盘', '最高', '最低'], windows=[3, 5, 10])
    print(data.head(10))

if __name__ == '__main__':
    data = load_data('../daily_data_exclude_new_can_buy/龙洲股份_002682.txt')
    data = data[-100:]
    get_data_feature(data)