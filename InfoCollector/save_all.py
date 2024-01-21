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

import akshare as ak
import logging
import concurrent.futures
import threading

import pandas as pd
import requests
from bs4 import BeautifulSoup

from StrategyExecutor.common import timeit


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
    notice_list = read_json(notice_file)
    periods = find_st_periods_strict(notice_list)
    df = price_data
    df['Max_rate'] = 10
    df['日期'] = pd.to_datetime(df['日期'])
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
        price_data = get_price(code, '19700101', '20291021', period='daily')
        filename = '../daily_data_exclude_new_can_buy/{}_{}.txt'.format(name, code)
        # price_data不为空才保存
        if not price_data.empty:
            price_data = fix_st(price_data, '../announcements/{}.json'.format(code))
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
def save_all_data_mul():
    stock_data_df = ak.stock_zh_a_spot_em()
    all_code_set = set(stock_data_df['代码'].tolist())

    exclude_code_set = set(ak.stock_kc_a_spot_em()['代码'].tolist())
    exclude_code_set.update(ak.stock_cy_a_spot_em()['代码'].tolist())

    need_code_set = {code for code in all_code_set if code.startswith(('000', '002', '003', '001', '600', '601', '603', '605'))}
    new_exclude_code_set = all_code_set - need_code_set
    new_exclude_code_set.update(exclude_code_set)

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(save_stock_data, stock_data, new_exclude_code_set) for _, stock_data in stock_data_df.iterrows()]
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
    return index_data

if __name__ == '__main__':
    # price_data = get_price('000001', '19700101', '20291017', period='daily')
    # print(price_data)
    data = ak.stock_zh_index_daily_em(symbol="sh000001")
    print(data)
    data = ak.stock_zh_index_spot()
    print(data)
    # data = ak.stock_financial_analysis_indicator('600242')
    # data1 = ak.stock_zh_a_st_em()
    # data2 = ak.stock_notice_report(symbol='全部', date="20231106")
    # data3 = ak.stock_zh_a_gdhs_detail_em(symbol="600242")
    # fun()
    # get_all_notice()
    # fix_announcements()
    # fetch_announcements('002740')

