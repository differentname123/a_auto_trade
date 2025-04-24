# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/21 14:39
:last_date:
    2023/10/21 14:39
:description:

"""
import time

import json
import multiprocessing
import os
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count

import akshare as ak
import logging
import concurrent.futures
import threading

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from StrategyExecutor.common import timeit
pd.options.mode.chained_assignment = None  # 关闭SettingWithCopyWarning

def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        traceback.print_exc()
        return {}

def get_price(symbol, start_time, end_time, period="daily"):
    """
    获取指定股票一定时间段内的收盘价
    :param symbol: 股票代码
    :param start_time:开始时间
    :param end_time:结束时间
    :param period:时间间隔
    :return:
    """
    if "daily" == period:
        result = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_time, end_date=end_time,
                                    adjust="qfq")
    else:
        result = ak.stock_zh_a_hist_min_em(symbol=symbol, period=period, start_date=start_time, end_date=end_time,
                                           adjust="qfq")
    # 如果 股票代码 在 result将它重命名为 代码
    if '股票代码' in result.columns:
        result.rename(columns={'股票代码': '代码'}, inplace=True)
    return result


# def save_all_data():
#     stock_data_df = ak.stock_zh_a_spot_em()
#     exclude_code = []
#     exclude_code.extend(ak.stock_kc_a_spot_em()['代码'].tolist())
#     exclude_code.extend(ak.stock_cy_a_spot_em()['代码'].tolist())
#     for _, stock_data in stock_data_df.iterrows():
#         name = stock_data['名称'].replace('*', '')
#         code = stock_data['代码']
#         if code in exclude_code:
#             continue
#         price_data = get_price(code, '19700101', '20231018', period='daily')
#         filename = 'daily_data_exclude_new/{}_{}.txt'.format(name, code)
#         price_data.to_csv(filename, index=False)

# Setting up the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def find_st_periods(posts):
    periods = []
    start = None
    end = None

    # 去除posts中notice_type包含 '深交所' 或 '上交所' 的帖子,或者是None
    posts = [post for post in posts if post['notice_type'] is not None and '深交所' not in post['notice_type'] and '上交所' not in post['notice_type']]

    for i in range(len(posts) - 1):
        current_post = posts[i]
        next_post = posts[i + 1]

        # 检查当前和下一个帖子是否都包含'ST'
        if 'ST' in current_post['post_title'] and 'ST' in next_post['post_title']:
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

def fix_st(price_data, notice_file):
    df = price_data
    df['Max_rate'] = 10
    df['日期'] = pd.to_datetime(df['日期'])
    if os.path.exists(notice_file):
        notice_list = read_json(notice_file)
        periods = find_st_periods_strict(notice_list)
        for start, end in periods:
            mask = (df['日期'] >= start) & (df['日期'] <= end)
            df.loc[mask, 'Max_rate'] = 5
    # 如果开盘，收盘，最高，最低价格都一样，那么设置为0
    df.loc[(df['开盘'] == df['收盘']) & (df['开盘'] == df['最高']) & (df['开盘'] == df['最低']), 'Max_rate'] = 0
    return df

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


def save_stock_data(stock_data, exclude_code):
    name = stock_data['名称'].replace('*', '')
    code = stock_data['代码']
    if code not in exclude_code:
        price_data = get_price(code, '20230101', '20291021', period='daily')
        filename = '../daily_data_exclude_new_can_buy/{}_{}.txt'.format(name, code)
        # price_data不为空才保存
        if not price_data.empty and len(price_data) > 26:
            price_data = fix_st(price_data, '../announcements/{}.json'.format(code))
            price_data = calculate_future_high_prices(price_data)
            price_data.to_csv(filename, index=False)
            # Logging the save operation with the timestamp
            logging.info(f"Saved data for {name} ({code}) to {filename}")


def calculate_total_estimate(row):
    # 获取每一行的净占比和净额
    net_inflow_ratios = {
        '主力净流入': row['主力净流入-净占比'],
        '超大单净流入': row['超大单净流入-净占比'],
        '大单净流入': row['大单净流入-净占比'],
        '中单净流入': row['中单净流入-净占比'],
        '小单净流入': row['小单净流入-净占比'],
    }

    net_inflow_values = {
        '主力净流入': row['主力净流入-净额'],
        '超大单净流入': row['超大单净流入-净额'],
        '大单净流入': row['大单净流入-净额'],
        '中单净流入': row['中单净流入-净额'],
        '小单净流入': row['小单净流入-净额'],
    }

    # 找到最大净占比及其对应的类型
    max_ratio_type = max(net_inflow_ratios, key=net_inflow_ratios.get)
    max_ratio = net_inflow_ratios[max_ratio_type]

    # 如果最大净占比大于0，计算总交易额，否则返回None
    if max_ratio > 0:
        return net_inflow_values[max_ratio_type] / max_ratio
    else:
        return None

def calculate_multi_day_metrics(df, col_name, days):
    # 计算滚动的多日净额
    df[f'{col_name}_{days}日'] = df[col_name].rolling(window=days).sum()

    # 计算滚动的多日总和
    df[f'总额估算_{days}日'] = df['总额估算'].rolling(window=days).sum()

    # 计算滚动的多日净占比
    df[f'{col_name.split("-")[0]}-净占比_{days}日'] = df[f'{col_name}_{days}日'] / df[f'总额估算_{days}日']
    # 计算总额估算较前一天的变化率
    df[f'总额估算_{days}日_变化率'] = df[f'总额估算_{days}日'].pct_change() * 100

    return df

def get_money_detail(stock_code):
    try:
        stock_individual_fund_flow_df = ak.stock_individual_fund_flow(stock=stock_code, market="sz")
    except Exception as e:
        stock_individual_fund_flow_df = ak.stock_individual_fund_flow(stock=stock_code, market="sh")
    # 计算总交易额
    # 应用该函数来计算每一行的总交易额
    stock_individual_fund_flow_df['总额估算'] = stock_individual_fund_flow_df.apply(calculate_total_estimate, axis=1)
    stock_individual_fund_flow_df[f'总额估算_1日_变化率'] = stock_individual_fund_flow_df[f'总额估算'].pct_change() * 100
    stock_individual_fund_flow_df['涨跌幅_3日'] = stock_individual_fund_flow_df['收盘价'].pct_change(periods=3) * 100
    stock_individual_fund_flow_df['涨跌幅_5日'] = stock_individual_fund_flow_df['收盘价'].pct_change(periods=5) * 100
    # 对每个需要计算的列进行多日计算
    for days in [3, 5]:
        stock_individual_fund_flow_df = calculate_multi_day_metrics(stock_individual_fund_flow_df, '主力净流入-净额',
                                                                    days)
        stock_individual_fund_flow_df = calculate_multi_day_metrics(stock_individual_fund_flow_df, '超大单净流入-净额',
                                                                    days)
        stock_individual_fund_flow_df = calculate_multi_day_metrics(stock_individual_fund_flow_df, '大单净流入-净额',
                                                                    days)
        stock_individual_fund_flow_df = calculate_multi_day_metrics(stock_individual_fund_flow_df, '中单净流入-净额',
                                                                    days)
        stock_individual_fund_flow_df = calculate_multi_day_metrics(stock_individual_fund_flow_df, '小单净流入-净额',
                                                                    days)


    # 计算涨跌幅比值
    stock_individual_fund_flow_df['主力涨跌幅比值'] = stock_individual_fund_flow_df['主力净流入-净占比'] / \
                                                      stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['超大单涨跌幅比值'] = stock_individual_fund_flow_df['超大单净流入-净占比'] / \
                                                        stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['大单涨跌幅比值'] = stock_individual_fund_flow_df['大单净流入-净占比'] / \
                                                      stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['中单涨跌幅比值'] = stock_individual_fund_flow_df['中单净流入-净占比'] / \
                                                      stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['小单涨跌幅比值'] = stock_individual_fund_flow_df['小单净流入-净占比'] / \
                                                      stock_individual_fund_flow_df['涨跌幅']
    # 计算3日涨跌幅比值
    stock_individual_fund_flow_df['3日主力涨跌幅比值'] = stock_individual_fund_flow_df['主力净流入-净占比_3日'] / \
                                                      stock_individual_fund_flow_df['涨跌幅_3日']
    stock_individual_fund_flow_df['3日超大单涨跌幅比值'] = stock_individual_fund_flow_df['超大单净流入-净占比_3日'] / \
                                                        stock_individual_fund_flow_df['涨跌幅_3日']
    stock_individual_fund_flow_df['3日大单涨跌幅比值'] = stock_individual_fund_flow_df['大单净流入-净占比_3日'] / \
                                                        stock_individual_fund_flow_df['涨跌幅_3日']
    stock_individual_fund_flow_df['3日中单涨跌幅比值'] = stock_individual_fund_flow_df['中单净流入-净占比_3日'] / \
                                                        stock_individual_fund_flow_df['涨跌幅_3日']
    stock_individual_fund_flow_df['3日小单涨跌幅比值'] = stock_individual_fund_flow_df['小单净流入-净占比_3日'] / \
                                                        stock_individual_fund_flow_df['涨跌幅_3日']
    # 计算5日涨跌幅比值
    stock_individual_fund_flow_df['5日主力涨跌幅比值'] = stock_individual_fund_flow_df['主力净流入-净占比_5日'] / \
                                                      stock_individual_fund_flow_df['涨跌幅_5日']
    stock_individual_fund_flow_df['5日超大单涨跌幅比值'] = stock_individual_fund_flow_df['超大单净流入-净占比_5日'] / \
                                                        stock_individual_fund_flow_df['涨跌幅_5日']
    stock_individual_fund_flow_df['5日大单涨跌幅比值'] = stock_individual_fund_flow_df['大单净流入-净占比_5日'] / \
                                                        stock_individual_fund_flow_df['涨跌幅_5日']
    stock_individual_fund_flow_df['5日中单涨跌幅比值'] = stock_individual_fund_flow_df['中单净流入-净占比_5日'] / \
                                                        stock_individual_fund_flow_df['涨跌幅_5日']
    stock_individual_fund_flow_df['5日小单涨跌幅比值'] = stock_individual_fund_flow_df['小单净流入-净占比_5日'] / \
                                                        stock_individual_fund_flow_df['涨跌幅_5日']

    # 计算净占比比值
    stock_individual_fund_flow_df['主力净占比比值'] = stock_individual_fund_flow_df['涨跌幅'] / stock_individual_fund_flow_df['主力净流入-净占比']
    stock_individual_fund_flow_df['超大单净占比比值'] = stock_individual_fund_flow_df['涨跌幅'] / stock_individual_fund_flow_df['超大单净流入-净占比']
    stock_individual_fund_flow_df['大单净占比比值'] = stock_individual_fund_flow_df['涨跌幅'] / stock_individual_fund_flow_df['大单净流入-净占比']
    stock_individual_fund_flow_df['中单净占比比值'] = stock_individual_fund_flow_df['涨跌幅'] / stock_individual_fund_flow_df['中单净流入-净占比']
    stock_individual_fund_flow_df['小单净占比比值'] = stock_individual_fund_flow_df['涨跌幅'] / stock_individual_fund_flow_df['小单净流入-净占比']
    # 计算3日净占比比值
    stock_individual_fund_flow_df['3日主力净占比比值'] = stock_individual_fund_flow_df['涨跌幅_3日'] / stock_individual_fund_flow_df['主力净流入-净占比_3日']
    stock_individual_fund_flow_df['3日超大单净占比比值'] = stock_individual_fund_flow_df['涨跌幅_3日'] / stock_individual_fund_flow_df['超大单净流入-净占比_3日']
    stock_individual_fund_flow_df['3日大单净占比比值'] = stock_individual_fund_flow_df['涨跌幅_3日'] / stock_individual_fund_flow_df['大单净流入-净占比_3日']
    stock_individual_fund_flow_df['3日中单净占比比值'] = stock_individual_fund_flow_df['涨跌幅_3日'] / stock_individual_fund_flow_df['中单净流入-净占比_3日']
    stock_individual_fund_flow_df['3日小单净占比比值'] = stock_individual_fund_flow_df['涨跌幅_3日'] / stock_individual_fund_flow_df['小单净流入-净占比_3日']
    # 计算5日净占比比值
    stock_individual_fund_flow_df['5日主力净占比比值'] = stock_individual_fund_flow_df['涨跌幅_5日'] / stock_individual_fund_flow_df['主力净流入-净占比_5日']
    stock_individual_fund_flow_df['5日超大单净占比比值'] = stock_individual_fund_flow_df['涨跌幅_5日'] / stock_individual_fund_flow_df['超大单净流入-净占比_5日']
    stock_individual_fund_flow_df['5日大单净占比比值'] = stock_individual_fund_flow_df['涨跌幅_5日'] / stock_individual_fund_flow_df['大单净流入-净占比_5日']
    stock_individual_fund_flow_df['5日中单净占比比值'] = stock_individual_fund_flow_df['涨跌幅_5日'] / stock_individual_fund_flow_df['中单净流入-净占比_5日']
    stock_individual_fund_flow_df['5日小单净占比比值'] = stock_individual_fund_flow_df['涨跌幅_5日'] / stock_individual_fund_flow_df['小单净流入-净占比_5日']


    # 计算涨跌幅乘积
    stock_individual_fund_flow_df['主力涨跌幅乘积'] = stock_individual_fund_flow_df['主力净流入-净占比'] * \
                                                        stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['超大单涨跌幅乘积'] = stock_individual_fund_flow_df['超大单净流入-净占比'] * \
                                                        stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['大单涨跌幅乘积'] = stock_individual_fund_flow_df['大单净流入-净占比'] * \
                                                        stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['中单涨跌幅乘积'] = stock_individual_fund_flow_df['中单净流入-净占比'] * \
                                                        stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['小单涨跌幅乘积'] = stock_individual_fund_flow_df['小单净流入-净占比'] * \
                                                        stock_individual_fund_flow_df['涨跌幅']
    # 计算3日涨跌幅乘积
    stock_individual_fund_flow_df['3日主力涨跌幅乘积'] = stock_individual_fund_flow_df['主力净流入-净占比_3日'] * \
                                                        stock_individual_fund_flow_df['涨跌幅_3日']
    stock_individual_fund_flow_df['3日超大单涨跌幅乘积'] = stock_individual_fund_flow_df['超大单净流入-净占比_3日'] * \
                                                        stock_individual_fund_flow_df['涨跌幅_3日']
    stock_individual_fund_flow_df['3日大单涨跌幅乘积'] = stock_individual_fund_flow_df['大单净流入-净占比_3日'] * \
                                                        stock_individual_fund_flow_df['涨跌幅_3日']
    stock_individual_fund_flow_df['3日中单涨跌幅乘积'] = stock_individual_fund_flow_df['中单净流入-净占比_3日'] * \
                                                        stock_individual_fund_flow_df['涨跌幅_3日']
    stock_individual_fund_flow_df['3日小单涨跌幅乘积'] = stock_individual_fund_flow_df['小单净流入-净占比_3日'] * \
                                                        stock_individual_fund_flow_df['涨跌幅_3日']
    # 计算5日涨跌幅乘积
    stock_individual_fund_flow_df['5日主力涨跌幅乘积'] = stock_individual_fund_flow_df['主力净流入-净占比_5日'] * \
                                                        stock_individual_fund_flow_df['涨跌幅_5日']
    stock_individual_fund_flow_df['5日超大单涨跌幅乘积'] = stock_individual_fund_flow_df['超大单净流入-净占比_5日'] * \
                                                        stock_individual_fund_flow_df['涨跌幅_5日']
    stock_individual_fund_flow_df['5日大单涨跌幅乘积'] = stock_individual_fund_flow_df['大单净流入-净占比_5日'] * \
                                                        stock_individual_fund_flow_df['涨跌幅_5日']
    stock_individual_fund_flow_df['5日中单涨跌幅乘积'] = stock_individual_fund_flow_df['中单净流入-净占比_5日'] * \
                                                        stock_individual_fund_flow_df['涨跌幅_5日']
    stock_individual_fund_flow_df['5日小单涨跌幅乘积'] = stock_individual_fund_flow_df['小单净流入-净占比_5日'] * \
                                                        stock_individual_fund_flow_df['涨跌幅_5日']
    return stock_individual_fund_flow_df

def save_money_stock_data(stock_data, exclude_code):
    name = stock_data['名称'].replace('*', '')
    code = stock_data['代码']
    if code not in exclude_code:
        stock_individual_fund_flow_df = get_money_detail(code)
        # kp_df = ak.stock_comment_detail_zlkp_jgcyd_em(symbol=code)
        stock_cyq_em_df = ak.stock_cyq_em(symbol=code, adjust="qfq")
        stock_cyq_em_df['获利比例排名'] = stock_cyq_em_df['获利比例'].expanding().apply(
            lambda x: x.rank(method='min', ascending=True).iloc[-1]
        )
        stock_cyq_em_df['与上一天的获利比例差值'] = stock_cyq_em_df['获利比例'] - stock_cyq_em_df['获利比例'].shift(1)
        # 将日期都转换为str
        stock_individual_fund_flow_df['日期'] = stock_individual_fund_flow_df['日期'].astype(str)
        stock_cyq_em_df['日期'] = stock_cyq_em_df['日期'].astype(str)
        # kp_df['日期'] = kp_df['date'].astype(str)
        # 计算kp_df中value和上一个value的差值
        # kp_df['value_diff'] = kp_df['value'] - kp_df['value'].shift(1)
        # stock_individual_fund_flow_df = pd.merge(stock_individual_fund_flow_df, kp_df, on='日期')
        # 以stock_individual_fund_flow_df为基础，合并stock_cyq_em_df，以日期为基础，stock_individual_fund_flow_df的行数不能变少
        stock_individual_fund_flow_df = pd.merge(stock_individual_fund_flow_df, stock_cyq_em_df, on='日期', how='left')
        price_data_df = get_price(code, '20240101', '20291021', period='daily')
        for days in [1, 2, 3]:
            # 获取后续最高价的利润率
            price_data_df[f'后续{days}日最高价利润率'] = round(
                (price_data_df['最高'].shift(-days).rolling(window=days, min_periods=1).max() - price_data_df['收盘']) / price_data_df[
                    '收盘'] * 100, 2)
            price_data_df[f'后续{days}日开盘价利润率'] = round(
                (price_data_df['开盘'].shift(-days).rolling(window=days, min_periods=1).max() - price_data_df['收盘']) / price_data_df[
                    '收盘'] * 100, 2)
            price_data_df[f'后续{days}日最低价利润率'] = round(
                (price_data_df['最低'].shift(-days).rolling(window=days, min_periods=1).min() - price_data_df['收盘']) / price_data_df[
                    '收盘'] * 100, 2)
            price_data_df[f'后续{days}日涨跌幅'] = round((price_data_df['收盘'].shift(-days) - price_data_df['收盘']) / price_data_df['收盘'] * 100, 2)

        price_data_df['日期'] = price_data_df['日期'].astype(str)
        # 合并price_data_df到stock_individual_fund_flow_df
        stock_individual_fund_flow_df = pd.merge(stock_individual_fund_flow_df, price_data_df, on='日期')

        # 将代码 和 名称添加到stock_individual_fund_flow_df，放在前面， 代码要以字符串形式添加
        stock_individual_fund_flow_df.insert(0, '代码', str(code))
        stock_individual_fund_flow_df.insert(1, '名称', name)

        stock_individual_fund_flow_df.to_csv(f'../money_detail/{name}_{code}.csv', index=False)
        print(f"Saved money detail for {name} ({code}) to ../money_detail/{name}_{code}.csv")

def fix_1_min_data(price_data):
    """
    完善分钟级别数据(字段重命名,添加新字段)
    :param price_data: DataFrame with price data
    :return: DataFrame with additional calculated data
    """
    # 重命名字段
    price_data.rename(columns={'时间': '日期', '开盘': '分钟_开盘', '收盘': '分钟_收盘', '最高': '分钟_最高', '最低': '分钟_最低',
                                '成交量': '分钟_成交量',
                               '成交额': '分钟_成交额'}, inplace=True)

    # 将日期列转换为datetime类型，保留具体时间
    price_data['日期'] = pd.to_datetime(price_data['日期'])

    # 提取日期作为一个新列，用于后续的分组
    price_data['日期_仅日期'] = price_data['日期'].dt.date

    # 计算到目前为止的统计数据
    price_data['开盘'] = price_data.groupby('日期_仅日期')['分钟_开盘'].transform('first')
    price_data['收盘'] = price_data['分钟_收盘']
    price_data['最高'] = price_data.groupby('日期_仅日期')['分钟_最高'].cummax()
    price_data['最低'] = price_data.groupby('日期_仅日期')['分钟_最低'].cummin()
    price_data['成交量'] = price_data.groupby('日期_仅日期')['分钟_成交量'].cumsum()
    price_data['成交额'] = price_data.groupby('日期_仅日期')['分钟_成交额'].cumsum()

    # 计算后续数据的统计数据
    reverse_grouped = price_data.iloc[::-1]  # 反转DataFrame以计算后续数据
    reverse_grouped['后续最高'] = reverse_grouped.groupby('日期_仅日期')['分钟_最高'].cummax()
    reverse_grouped['后续最低'] = reverse_grouped.groupby('日期_仅日期')['分钟_最低'].cummin()

    # 将反转后的统计数据再次反转回来，并合并到原始DataFrame
    reverse_grouped = reverse_grouped.iloc[::-1]
    price_data['后续最高'] = reverse_grouped['后续最高']
    price_data['后续最低'] = reverse_grouped['后续最低']

    # 删除临时列
    price_data.drop(columns=['日期_仅日期'], inplace=True)
    price_data['Max_rate'] = 10

    return price_data

def fix_min_data(price_data):
    """
    完善分钟级别数据(字段重命名,添加新字段)
    :param price_data: DataFrame with price data
    :return: DataFrame with additional calculated data
    """
    # 重命名字段
    price_data.rename(columns={'时间': '日期', '开盘': '分钟_开盘', '收盘': '分钟_收盘', '最高': '分钟_最高', '最低': '分钟_最低',
                                '成交量': '分钟_成交量',
                               '成交额': '分钟_成交额', '换手率': '分钟_换手率'}, inplace=True)

    # 将日期列转换为datetime类型，保留具体时间
    price_data['日期'] = pd.to_datetime(price_data['日期'])

    # 提取日期作为一个新列，用于后续的分组
    price_data['日期_仅日期'] = price_data['日期'].dt.date

    # 计算到目前为止的统计数据
    price_data['开盘'] = price_data.groupby('日期_仅日期')['分钟_开盘'].transform('first')
    price_data['收盘'] = price_data['分钟_收盘']
    price_data['最高'] = price_data.groupby('日期_仅日期')['分钟_最高'].cummax()
    price_data['最低'] = price_data.groupby('日期_仅日期')['分钟_最低'].cummin()
    price_data['成交量'] = price_data.groupby('日期_仅日期')['分钟_成交量'].cumsum()
    price_data['成交额'] = price_data.groupby('日期_仅日期')['分钟_成交额'].cumsum()
    price_data['换手率'] = price_data.groupby('日期_仅日期')['分钟_换手率'].cumsum()

    # 计算后续数据的统计数据
    reverse_grouped = price_data.iloc[::-1]  # 反转DataFrame以计算后续数据
    reverse_grouped['后续最高'] = reverse_grouped.groupby('日期_仅日期')['分钟_最高'].cummax()
    reverse_grouped['后续最低'] = reverse_grouped.groupby('日期_仅日期')['分钟_最低'].cummin()

    # 将反转后的统计数据再次反转回来，并合并到原始DataFrame
    reverse_grouped = reverse_grouped.iloc[::-1]
    price_data['后续最高'] = reverse_grouped['后续最高']
    price_data['后续最低'] = reverse_grouped['后续最低']

    # 删除临时列
    price_data.drop(columns=['日期_仅日期'], inplace=True)
    price_data['Max_rate'] = 10

    return price_data


def save_stock_data_min(stock_data, exclude_code):
    name = stock_data['名称'].replace('*', '')
    code = stock_data['代码']
    befor_price_data = pd.DataFrame()
    if code not in exclude_code:
        price_data = get_price(code, '19700101', '20291021', period='5')
        filename = '../min_data_exclude_new_can_buy/{}_{}.txt'.format(name, code)
        # 如果filename存在,则读取
        if os.path.exists(filename):
            befor_price_data = pd.read_csv(filename)
        # price_data不为空才保存
        if not price_data.empty:
            price_data = fix_min_data(price_data)
            price_data = fix_st(price_data, '../announcements/{}.json'.format(code))
            if not befor_price_data.empty:
                price_data = pd.concat([befor_price_data, price_data])
            price_data['日期'] = pd.to_datetime(price_data['日期'])
            price_data = price_data.drop_duplicates(subset=['日期'], keep='first')
            price_data.to_csv(filename, index=False)
            # Logging the save operation with the timestamp
            logging.info(f"Saved data for {name} ({code}) to {filename}")

def save_stock_data_1_min(stock_data, exclude_code):
    name = stock_data['名称'].replace('*', '')
    code = stock_data['代码']
    befor_price_data = pd.DataFrame()
    if code not in exclude_code:
        price_data = get_price(code, '19700101', '20291021', period='1')
        filename = '../1_min_data_exclude_new_can_buy/{}_{}.txt'.format(name, code)
        # 如果filename存在,则读取
        if os.path.exists(filename):
            befor_price_data = pd.read_csv(filename)
        # price_data不为空才保存
        if not price_data.empty:
            price_data = fix_1_min_data(price_data)
            price_data = fix_st(price_data, '../announcements/{}.json'.format(code))
            if not befor_price_data.empty:
                price_data = pd.concat([befor_price_data, price_data])
            price_data = price_data.drop_duplicates(subset=['日期'], keep='first')
            price_data.to_csv(filename, index=False)
            # Logging the save operation with the timestamp
            logging.info(f"Saved data for {name} ({code}) to {filename}")

def write_json(file_path, data):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing {file_path} exception: {e}")

def fetch_announcements(stock_code):
    extracted_data = []
    page = 1

    while True:
        if page == 1:
            url = f'https://guba.eastmoney.com/list,{stock_code},3,f.html'
        else:
            url = f'https://guba.eastmoney.com/list,{stock_code},3,f_{page}.html'

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.text, 'html.parser')
        # 找到包含article_list的脚本
        scripts = soup.find_all('script')
        article_list_script = None
        for script in scripts:
            if 'article_list' in script.text:
                article_list_script = script.text
                break

        article_list_str = article_list_script.split('var article_list=')[1].split('};')[0] + '}'
        article_list_json = json.loads(article_list_str)
        code_name = article_list_json['bar_name']
        # Extracting the required information from each article
        articles = article_list_json['re']
        if len(articles) == 0:
            break
        for article in articles:
            title = article['post_title']
            notice_type = article['notice_type']
            publish_time = article['post_publish_time']
            extracted_data.append({
                'post_title': title,
                'notice_type': notice_type,
                'post_publish_time': publish_time
            })
        page += 1
    # 将extracted_data写入文件
    write_json(f'../announcements/{stock_code}.json', extracted_data)
    print(f"Saved data for {code_name} ({stock_code}) to ../announcements/{code_name}_{stock_code}.json")

def get_all_notice():
    stock_data_df = ak.stock_zh_a_spot_em()
    exclude_code = []
    need_code = []
    exclude_code.extend(ak.stock_kc_a_spot_em()['代码'].tolist())
    exclude_code.extend(ak.stock_cy_a_spot_em()['代码'].tolist())
    # 获取以00或者30开头的股票代码
    need_code.extend([code for code in stock_data_df['代码'].tolist() if code.startswith('000') or code.startswith('002')  or code.startswith('003')
                      or code.startswith('001') or code.startswith('600') or code.startswith('601') or code.startswith('603') or code.startswith('605')])
    # # 读取../announcements/下的所有文件名
    # exclude_code = []
    # for file in os.listdir('../announcements'):
    #     if file.endswith('.json'):
    #         exclude_code.append(file.split('.')[0])
    # need_code = list(set(need_code) - set(exclude_code))
    for code in need_code:
        fetch_announcements(code)

def fix_announcements():
    """
    通过公告调整st 的max_rate
    :return:
    """
    # 将../announcements/下的所有文件名重命名为 股票代码.json
    for file in os.listdir('../announcements'):
        if file.endswith('.json'):
            os.rename('../announcements/' + file, '../announcements/' + file.split('_')[1])

@timeit
def save_all_data():

    stock_data_df = ak.stock_zh_a_spot_em()
    # 获取stock_data_df中的股票代码
    all_code = stock_data_df['代码'].tolist()
    exclude_code = []
    need_code = []
    exclude_code.extend(ak.stock_kc_a_spot_em()['代码'].tolist())
    exclude_code.extend(ak.stock_cy_a_spot_em()['代码'].tolist())
    # 获取以00或者30开头的股票代码
    need_code.extend([code for code in stock_data_df['代码'].tolist() if code.startswith('000') or code.startswith('002')  or code.startswith('003')
                      or code.startswith('001') or code.startswith('600') or code.startswith('601') or code.startswith('603') or code.startswith('605')])
    new_exclude_code = [code for code in all_code if code not in need_code]
    new_exclude_code.extend(exclude_code)
    # 将new_exclude_code去重
    new_exclude_code = list(set(new_exclude_code))
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(save_stock_data, stock_data, new_exclude_code) for _, stock_data in
                   stock_data_df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error occurred: {e}")

@timeit
def save_all_data_mul(save_fun=save_stock_data):
    stock_data_df = ak.stock_zh_a_spot_em()
    # all_code_set = set(stock_data_df['代码'].tolist())
    #
    # exclude_code_set = set(ak.stock_kc_a_spot_em()['代码'].tolist())
    # exclude_code_set.update(ak.stock_cy_a_spot_em()['代码'].tolist())
    #
    # need_code_set = {code for code in all_code_set if code.startswith(('000', '002', '003', '001', '600', '601', '603', '605'))}
    # new_exclude_code_set = all_code_set - need_code_set
    # new_exclude_code_set.update(exclude_code_set)

    # 扫描'../daily_data_exclude_new_can_buy'下面的所有文件
    exclude_code_set = set()
    for file in os.listdir('../daily_data_exclude_new_can_buy'):
        if file.endswith('.txt'):
            code = file.split('_')[1].split('.')[0]
            exclude_code_set.add(code)
    new_exclude_code_set = exclude_code_set
    print(f"exclude_code_set: {len(exclude_code_set)}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(save_fun, stock_data, new_exclude_code_set) for _, stock_data in stock_data_df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error occurred: {e}")

def fun():
    save_all_data_mul()

def save_index_data():
    """
    下载指数数据（目前是上证和深证指数），保存到output_path
    :param output_path:
    :return:
    """
    sz_data = ak.stock_zh_index_daily_em(symbol="sz399001")
    sh_data = ak.stock_zh_index_daily_em(symbol="sh000001")
    # 将sz_data和sh_data的列date改为日期
    sz_data.rename(columns={'date': '日期'}, inplace=True)
    sz_data.rename(columns={'open': '深证指数开盘'}, inplace=True)
    sz_data.rename(columns={'high': '深证指数最高'}, inplace=True)
    sz_data.rename(columns={'low': '深证指数最低'}, inplace=True)
    sz_data.rename(columns={'close': '深证指数收盘'}, inplace=True)
    sz_data.rename(columns={'volume': '深证指数成交额'}, inplace=True)
    sh_data.rename(columns={'date': '日期'}, inplace=True)
    sh_data.rename(columns={'open': '上证指数开盘'}, inplace=True)
    sh_data.rename(columns={'high': '上证指数最高'}, inplace=True)
    sh_data.rename(columns={'low': '上证指数最低'}, inplace=True)
    sh_data.rename(columns={'close': '上证指数收盘'}, inplace=True)
    sh_data.rename(columns={'volume': '上证指数成交额'}, inplace=True)
    # 将sz_data和sh_data按照日期合并
    index_data = pd.merge(sz_data, sh_data, on='日期')
    # 将index_data的日期改为datetime格式
    index_data['日期'] = pd.to_datetime(index_data['日期'])
    index_data.rename(columns={'日期': '指数日期'}, inplace=True)
    index_data.to_csv('../train_data/index_data.csv', index=False)
    return index_data

def calculate_future_high_prices(data):
    """
    计算后续1，2，3日的最高价。

    参数:
    - data: DataFrame，包含市场数据，其中应包含一个名为'最高价'的列。

    返回值:
    - 修改后的DataFrame，包含新增的后续1，2，3日最高价列。
    """
    # 计算后续1，2，3日的最高价
    data[f'后续1日开盘价利润率'] = round((data['开盘'].shift(-1) - data['收盘']) / data['收盘'] * 100, 2)
    for days in [1, 2, 3]:
        # 获取后续最高价的利润率
        data[f'后续{days}日最高价利润率'] = round((data['最高'].shift(-days).rolling(window=days, min_periods=1).max() - data['收盘']) / data['收盘'] * 100, 2)
        data[f'后续{days}日最低价利润率'] = round((data['最低'].shift(-days).rolling(window=days, min_periods=1).min() - data['收盘']) / data['收盘'] * 100, 2)
        data[f'后续{days}日涨跌幅'] = round((data['收盘'].shift(-days) - data['收盘']) / data['收盘'] * 100, 2)

    return data

def money_detail():
    # 限量: 单次获取指定类型的个股资金流排名数据
    stock_individual_fund_flow_df = ak.stock_individual_fund_flow_rank(indicator="今日")
    # 筛选出 今日涨跌幅 不是数字的行
    stock_individual_fund_flow_df = stock_individual_fund_flow_df[stock_individual_fund_flow_df['今日涨跌幅'].apply(lambda x: isinstance(x, (int, float)))]
    # 将stock_individual_fund_flow_df['今日涨跌幅']转换为float类型
    stock_individual_fund_flow_df['今日涨跌幅'] = stock_individual_fund_flow_df['今日涨跌幅'].astype(float)
    # 过滤掉今日涨跌幅为0的行
    stock_individual_fund_flow_df = stock_individual_fund_flow_df[stock_individual_fund_flow_df['今日涨跌幅'] != 0]
    stock_individual_fund_flow_df['主力涨跌幅比值'] = stock_individual_fund_flow_df['今日主力净流入-净占比'] / stock_individual_fund_flow_df['今日涨跌幅']
    stock_individual_fund_flow_df['超大单涨跌幅比值'] = stock_individual_fund_flow_df['今日超大单净流入-净占比'] / stock_individual_fund_flow_df['今日涨跌幅']
    stock_individual_fund_flow_df['大单涨跌幅比值'] = stock_individual_fund_flow_df['今日大单净流入-净占比'] / stock_individual_fund_flow_df['今日涨跌幅']
    stock_individual_fund_flow_df['中单涨跌幅比值'] = stock_individual_fund_flow_df['今日中单净流入-净占比'] / stock_individual_fund_flow_df['今日涨跌幅']
    stock_individual_fund_flow_df['小单涨跌幅比值'] = stock_individual_fund_flow_df['今日小单净流入-净占比'] / stock_individual_fund_flow_df['今日涨跌幅']
    # 计算最近3日内的主力净流入-净占比的和
    stock_individual_fund_flow_df['主力净流入-净占比3日和'] = stock_individual_fund_flow_df['今日主力净流入-净占比'].rolling(window=3, min_periods=1).sum()
    # 计算最近3日内的超大单净流入-净占比的和
    stock_individual_fund_flow_df['超大单净流入-净占比3日和'] = stock_individual_fund_flow_df['今日超大单净流入-净占比'].rolling(window=3, min_periods=1).sum()
    stock_individual_fund_flow_df['大单净流入-净占比3日和'] = stock_individual_fund_flow_df['今日大单净流入-净占比'].rolling(window=3, min_periods=1).sum()
    stock_individual_fund_flow_df['中单净流入-净占比3日和'] = stock_individual_fund_flow_df['今日中单净流入-净占比'].rolling(window=3, min_periods=1).sum()
    stock_individual_fund_flow_df['小单净流入-净占比3日和'] = stock_individual_fund_flow_df['今日小单净流入-净占比'].rolling(window=3, min_periods=1).sum()
    # 计算3日涨跌幅的和
    stock_individual_fund_flow_df['涨跌幅3日和'] = stock_individual_fund_flow_df['今日涨跌幅'].rolling(window=3, min_periods=1).sum()
    stock_individual_fund_flow_df['3日主力涨跌幅比值'] = stock_individual_fund_flow_df['主力净流入-净占比3日和'] / stock_individual_fund_flow_df['涨跌幅3日和']
    stock_individual_fund_flow_df['3日超大单涨跌幅比值'] = stock_individual_fund_flow_df['超大单净流入-净占比3日和'] / stock_individual_fund_flow_df['涨跌幅3日和']
    stock_individual_fund_flow_df['3日大单涨跌幅比值'] = stock_individual_fund_flow_df['大单净流入-净占比3日和'] / stock_individual_fund_flow_df['涨跌幅3日和']
    stock_individual_fund_flow_df['3日中单涨跌幅比值'] = stock_individual_fund_flow_df['中单净流入-净占比3日和'] / stock_individual_fund_flow_df['涨跌幅3日和']
    stock_individual_fund_flow_df['3日小单涨跌幅比值'] = stock_individual_fund_flow_df['小单净流入-净占比3日和'] / stock_individual_fund_flow_df['涨跌幅3日和']
    print(stock_individual_fund_flow_df)

    # 限量: 单次获取指定市场和股票的近 100 个交易日的资金流数据
    stock_individual_fund_flow_df = ak.stock_individual_fund_flow(stock="001317", market="sz")
    stock_individual_fund_flow_df['主力涨跌幅比值'] = stock_individual_fund_flow_df['主力净流入-净占比'] / stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['超大单涨跌幅比值'] = stock_individual_fund_flow_df['超大单净流入-净占比'] / stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['大单涨跌幅比值'] = stock_individual_fund_flow_df['大单净流入-净占比'] / stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['中单涨跌幅比值'] = stock_individual_fund_flow_df['中单净流入-净占比'] / stock_individual_fund_flow_df['涨跌幅']
    stock_individual_fund_flow_df['小单涨跌幅比值'] = stock_individual_fund_flow_df['小单净流入-净占比'] / stock_individual_fund_flow_df['涨跌幅']
    # 计算最近3日内的主力净流入-净占比的和
    stock_individual_fund_flow_df['主力净流入-净占比3日和'] = stock_individual_fund_flow_df['主力净流入-净占比'].rolling(window=3, min_periods=1).sum()
    # 计算最近3日内的超大单净流入-净占比的和
    stock_individual_fund_flow_df['超大单净流入-净占比3日和'] = stock_individual_fund_flow_df['超大单净流入-净占比'].rolling(window=3, min_periods=1).sum()
    stock_individual_fund_flow_df['大单净流入-净占比3日和'] = stock_individual_fund_flow_df['大单净流入-净占比'].rolling(window=3, min_periods=1).sum()
    stock_individual_fund_flow_df['中单净流入-净占比3日和'] = stock_individual_fund_flow_df['中单净流入-净占比'].rolling(window=3, min_periods=1).sum()
    stock_individual_fund_flow_df['小单净流入-净占比3日和'] = stock_individual_fund_flow_df['小单净流入-净占比'].rolling(window=3, min_periods=1).sum()
    # 计算3日涨跌幅的和
    stock_individual_fund_flow_df['涨跌幅3日和'] = stock_individual_fund_flow_df['涨跌幅'].rolling(window=3, min_periods=1).sum()
    stock_individual_fund_flow_df['3日主力涨跌幅比值'] = stock_individual_fund_flow_df['主力净流入-净占比3日和'] / stock_individual_fund_flow_df['涨跌幅3日和']
    stock_individual_fund_flow_df['3日超大单涨跌幅比值'] = stock_individual_fund_flow_df['超大单净流入-净占比3日和'] / stock_individual_fund_flow_df['涨跌幅3日和']
    stock_individual_fund_flow_df['3日大单涨跌幅比值'] = stock_individual_fund_flow_df['大单净流入-净占比3日和'] / stock_individual_fund_flow_df['涨跌幅3日和']
    stock_individual_fund_flow_df['3日中单涨跌幅比值'] = stock_individual_fund_flow_df['中单净流入-净占比3日和'] / stock_individual_fund_flow_df['涨跌幅3日和']
    stock_individual_fund_flow_df['3日小单涨跌幅比值'] = stock_individual_fund_flow_df['小单净流入-净占比3日和'] / stock_individual_fund_flow_df['涨跌幅3日和']
    print(stock_individual_fund_flow_df)


    #限量: 单次获取当前时点的所有大单追踪数据
    stock_fund_flow_big_deal_df = ak.stock_fund_flow_big_deal()
    print(stock_fund_flow_big_deal_df)

    # 限量: 单次获取指定 symbol 的概念资金流数据
    stock_fund_flow_individual_df = ak.stock_fund_flow_individual(symbol="即时")
    print(stock_fund_flow_individual_df)

def process_large_group(args):
    start_time = time.time()
    codes_chunk, money_detail_df, full_data = args
    result_list = []
    print(f"Processing chunk {len(codes_chunk)}")
    for code in codes_chunk:
        # 筛选出当前块中的所有日期
        grouped = money_detail_df[money_detail_df['代码'] == code].groupby('日期')
        for date, group in grouped:
            # 从full_data中找到对应的股票代码和日期的数据
            stock_data = full_data[(full_data['代码'] == code) & (full_data['日期'] == date)]
            if not stock_data.empty:
                # 将stock_data的换手率加入到group中
                group['换手率'] = stock_data['换手率'].values[0]
                result_list.append(group)
    print(f"Processed chunk in {time.time() - start_time:.2f} seconds")
    return pd.concat(result_list)

def anlyse_money_data():
    # 读取'../final_zuhe/other/money_detail.csv'
    # pd.read_csv('../final_zuhe/other/money_detail_full.csv')
    # money_detail_df = pd.read_csv('../final_zuhe/other/money_detail.csv', dtype={'代码': str})
    # full_data = pd.read_csv('../train_data/2024_data_2024.csv', dtype={'代码': str})
    # # 过滤出日期包含08-的数据
    # # key_data = '08-15'
    # # money_detail_df = money_detail_df[money_detail_df['日期'].str.contains(key_data)]
    # # full_data = full_data[full_data['日期'].str.contains(key_data)]
    # # 按照代码和日期分组
    #
    # # 获取唯一的股票代码列表并分块
    # unique_codes = money_detail_df['代码'].unique()
    # chunks = np.array_split(unique_codes, cpu_count())  # 根据CPU核心数分块
    #
    # # 准备任务列表，将 money_detail_df 和 full_data 一并传入
    # task_args = [(chunk, money_detail_df, full_data) for chunk in chunks]
    #
    # # 使用多进程处理
    # with Pool(processes=20) as pool:
    #     result_list = pool.map(process_large_group, task_args)
    #
    # # 合并结果并保存
    # result_df = pd.concat(result_list)
    # result_df.to_csv('../final_zuhe/other/money_detail_full.csv', index=False)
    # print('Done')
    current_date = time.strftime('%Y-%m-%d', time.localtime())
    start_time = time.time()
    # 扫描f'../money_detail/'下的所有文件，使用多线程进行合并，并且保存
    money_detail_files = [f'../money_detail/{file}' for file in os.listdir('../money_detail') if file.endswith('.csv')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(pd.read_csv, file, dtype={'代码': str}) for file in money_detail_files]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    money_detail_df = pd.concat(results)
    print(f"Loaded and concatenated all files in {time.time() - start_time:.2f} seconds")
    money_detail_df.to_csv(f'../final_zuhe/other/money_detail_{current_date}.csv', index=False)
    money_detail_df.to_csv(f'../final_zuhe/other/money_detail.csv', index=False)

if __name__ == '__main__':
    # # 删除../money_detail下面的所有文件
    # for file in os.listdir('../money_detail'):
    #     os.remove(f'../money_detail/{file}')
    # save_all_data_mul(save_fun=save_money_stock_data)
    # anlyse_money_data()


    # money_detail()
    # save_index_data()
    price_data = get_price('002492', '20230101', '20290124', period='1')
    print(price_data)
    # #
    # # stock_zh_b_minute_df = ak.stock_zh_b_minute(symbol='sh900901', period='1', adjust="qfq")
    # # print(stock_zh_b_minute_df)
    # # data = ak.stock_zh_index_daily_em(symbol="sh000001")
    # # print(data)
    # # data = ak.stock_zh_index_spot()
    # # print(data)
    # # data = ak.stock_financial_analysis_indicator('600242')
    # # data1 = ak.stock_zh_a_st_em()
    # # data2 = ak.stock_notice_report(symbol='全部', date="20231106")
    # # data3 = ak.stock_zh_a_gdhs_detail_em(symbol="600242")
    # # fun()
    # # get_all_notice()
    # # fix_announcements()
    # # fetch_announcements('002740')
    # # save_index_data()
    # save_all_data_mul(save_fun=save_stock_data_1_min)
    # save_all_data_mul(save_fun=save_stock_data_min)
    # save_all_data_mul(save_fun=save_money_stock_data)
