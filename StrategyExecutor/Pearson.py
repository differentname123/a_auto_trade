# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2024-02-19 15:39
:last_date:
    2024-02-19 15:39
:description:
    判断找出还未来得及反弹的数据
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use('Agg')  # 设置matplotlib使用Agg后端，无需GUI
import matplotlib.pyplot as plt


def plot_relative_returns_and_save(group_returns, overall_returns, filepath, group_label='Group',
                                   overall_label='Overall'):
    """
    绘制组数据与大盘数据的涨跌幅点线图，并将图表保存为图片。
    """
    plt.figure(figsize=(14, 7))  # 设置图像大小
    # 确保索引对齐
    dates = group_returns.index.intersection(overall_returns.index)

    # 绘制组数据的涨跌幅
    plt.plot(dates, group_returns.loc[dates], marker='o', linestyle='-', color='blue', label=group_label)
    # 绘制大盘数据的涨跌幅
    plt.plot(dates, overall_returns.loc[dates], marker='x', linestyle='--', color='red', label=overall_label)

    plt.title('Relative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Relative Return')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(filepath)  # 保存图表为图片
    plt.close()

def load_all_data():
    """
    加载所有数据
    :return:
    """
    result_list = []
    # 遍历../1_min_data_exclude_new_can_buy/目录下的所有文件
    file_path = '../1_min_data_exclude_new_can_buy'
    file_list = os.listdir(file_path)
    for file in file_list:
        code = file.split('.')[0].split('_')[1]
        file_path_path = os.path.join(file_path, file)
        data = pd.read_csv(file_path_path, low_memory=False)
        data['日期'] = pd.to_datetime(data['日期'])
        data = data[data['日期'] >= pd.Timestamp('2024-02-19')]
        data['code'] = code
        if len(data) > 0:
            result_list.append(data)
    return result_list


def calculate_moving_averages(data, short_window=5, long_window=10):
    """
    计算短期和长期移动平均线。
    """
    data['短期平均'] = data['收盘'].rolling(window=short_window, min_periods=1).mean()
    data['长期平均'] = data['收盘'].rolling(window=long_window, min_periods=1).mean()
    data['反弹信号'] = (data['短期平均'] > data['长期平均']) & (data['短期平均'].shift(1) <= data['长期平均'].shift(1))
    return data


def identify_first_overall_rebound_date(data):
    """
    确定整体趋势首次反弹日期。
    """
    rebound_dates = data[data['反弹信号']]['日期']
    return rebound_dates.min() if not rebound_dates.empty else None


def check_rebound_signal_within_window(group, date, window_size=3):
    """
    检查在指定日期前的时间窗口内是否存在反弹信号。
    """
    relevant_rows = group[group['日期'] <= date].tail(window_size + 1)  # 包括反弹点当天和之前的3个时间点
    return relevant_rows.iloc[:-1]['反弹信号'].any()  # 排除反弹点当天


def calculate_future_return(group):
    """
    为数据组计算后续最高价相较于当前价格的涨跌幅。
    """
    group['后续最高价'] = group['收盘'].rolling(window=len(group), min_periods=1).max().shift(-1)
    group['后续涨跌幅'] = (group['后续最高价'] - group['收盘']) / group['收盘']
    return group


def find_delayed_rebound_groups(data_group, first_overall_rebound_date):
    """
    识别滞后的数据组及其相应行，并计算后续涨跌幅。
    """
    delayed_groups_with_return = {}
    if first_overall_rebound_date:
        for code, group in data_group:
            group_with_ma = calculate_moving_averages(group.copy())  # 计算移动平均线和反弹信号
            group_with_return = calculate_future_return(group_with_ma)  # 计算后续涨跌幅
            if not check_rebound_signal_within_window(group_with_return, first_overall_rebound_date):
                new_rebounds = group_with_return[
                    (group_with_return['日期'] > first_overall_rebound_date) & (group_with_return['反弹信号'])]
                if not new_rebounds.empty:
                    first_new_rebound_row = new_rebounds.iloc[0]
                    delayed_groups_with_return[code] = first_new_rebound_row.to_dict()
    return delayed_groups_with_return


def identify_all_rebound_dates(data):
    """
    识别整体趋势中的所有反弹点。
    """
    data_with_ma = calculate_moving_averages(data)
    rebound_dates = data_with_ma[data_with_ma['反弹信号']]['日期']
    return rebound_dates

def calculate_relative_returns(series):
    """
    计算相对于序列中第一个有效数据点的涨跌幅。
    """
    first_value = series.iloc[0]  # 获取序列中的第一个有效数据点
    relative_returns = series / first_value - 1  # 计算相对于第一个点的涨跌幅
    return relative_returns


def calculate_correlation(group, overall_data):
    """
    计算组数据与大盘数据之间的涨跌幅相关系数。
    """
    # 在进行修改之前创建副本以避免SettingWithCopyWarning
    group_copy = group.copy()
    overall_data_copy = overall_data.copy()

    # 计算涨跌幅
    group_copy['涨跌幅'] = calculate_relative_returns(group_copy['收盘'])
    overall_data_copy['涨跌幅_overall'] = calculate_relative_returns(overall_data_copy['收盘'])

    # 确保两个序列在相同的日期对齐
    merged_data = pd.merge(group_copy[['日期', '涨跌幅']], overall_data_copy[['日期', '涨跌幅_overall']], on='日期')
    # 删除merged_data的最后一行
    merged_data = merged_data.iloc[:-1]

    plot_relative_returns_and_save(merged_data['涨跌幅'], merged_data['涨跌幅_overall'], '../temp/returns.png')
    correlation = np.corrcoef(merged_data['涨跌幅'], merged_data['涨跌幅_overall'])[0, 1]
    return correlation

def analyze_group_rebounds(data_group, rebound_dates, overall_data, correlation_threshold=0.5):
    """
    分析每个数据组在所有反弹点之后的行为，识别滞后反弹。
    返回一个包含DataFrame的列表，每个DataFrame包含每个数据组滞后的反弹点。
    """
    delayed_groups_with_return = []  # 初始化为列表
    for code, group in data_group:
        # 如果group['收盘']都是一个值，跳过
        if group['收盘'].nunique() == 1:
            continue
        group_with_return = calculate_future_return(calculate_moving_averages(group.copy()))
        delayed_rebounds_list = []
        for date in rebound_dates:
            group_to_date = group_with_return[group_with_return['日期'] <= date]
            overall_to_date = overall_data[overall_data['日期'] <= date]
            correlation = calculate_correlation(group_to_date, overall_to_date)
            print(correlation)
            if correlation < correlation_threshold or correlation > 1:
                continue  # 如果相关性低于阈值，跳过该组
            # 计算与大盘的相关性
            if check_rebound_signal_within_window(group_with_return, date):
                continue  # 如果在反弹点前存在反弹信号，则跳过
            # 检查在此反弹点之后的反弹行为
            rebound_after = group_with_return[(group_with_return['日期'] == date)]
            if not rebound_after.empty:
                delayed_rebounds_list.append(rebound_after.iloc[0].to_dict())  # 收集滞后的反弹点
        if delayed_rebounds_list:
            delayed_rebounds_df = pd.DataFrame(delayed_rebounds_list)  # 将列表转换为DataFrame
            delayed_rebounds_df['code'] = code  # 添加数据组标识
            delayed_groups_with_return.append(delayed_rebounds_df)  # 将DataFrame添加到列表
    return delayed_groups_with_return

if __name__ == '__main__':
    data = pd.read_csv('../temp/all_data.csv', low_memory=False, dtype={'code': str})
    data['日期'] = pd.to_datetime(data['日期'])
    data.sort_values(by='日期', inplace=True)

    # 计算整体趋势的移动平均线和识别所有反弹点
    overall_data = data.groupby('日期')['收盘'].mean().reset_index()
    all_rebound_dates = identify_all_rebound_dates(overall_data)

    # 分组处理数据
    data_group = data.groupby('code')
    delayed_groups_with_info = analyze_group_rebounds(data_group, all_rebound_dates, overall_data)

    print("滞后反弹的数据组及其信息:", delayed_groups_with_info)