# -*- coding: utf-8 -*-
import math
import multiprocessing
from datetime import datetime

import json
import os
import traceback

from InfoCollector.save_all import get_price
from StrategyExecutor.full_zehe import save_and_analyse_all_data_mul
from thsauto import ThsAuto

import time

client_path = r'D:\install\THS\xiadan.exe'

def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        traceback.print_exc()
        return {}

def run_client():
    os.system('start ' + client_path)

def get_stock_and_price(file_path):
    result = []
    total_price = 0
    data = read_json(file_path)
    for stock_info in data:
        stock_no = stock_info['stock_name'].split('_')[1]
        price = stock_info['收盘']
        if stock_info['涨跌幅'] >= -(stock_info['Max_rate'] - 1.0 / price):
            total_price += price
            result.append((stock_no, price))
    print('总价值：', total_price)
    return result


def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def process_stock_data(file_path, auto, exist_codes, amount):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            try:
                stock_no, price = line.strip().split(',')
                price = float(price)
                if stock_no not in exist_codes:
                    exist_codes.append(stock_no)
                    result = auto.quick_buy(stock_no=stock_no, amount=amount, price=price)
            except Exception:
                traceback.print_exc()
                print(line)

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

if __name__ == '__main__':
    auto = ThsAuto()  # 连接客户端
    auto.active_mian_window()
    # auto.kill_client()
    # run_client()
    # time.sleep(5)
    auto.bind_client()
    # 撤销所有单
    # auto.cancel_all(entrust_no=None)
    print('持仓')
    data = auto.get_position()
    print(data)
    for detail_data in data['data']:
        stock_no = detail_data['证券代码']
        try:
            amount = int(detail_data['可用余额'])
        except Exception:
            amount = 100
        days_held = detail_data['持股天数']
        # 将days_held转换为整数，如果失败那就是1
        try:
            days_held = int(days_held)
        except Exception:
            days_held = 1

        price = float(detail_data['市价'])
        if amount > 0:
            daily_data = get_price(stock_no, '20240101', '20291021', period='daily')
            if daily_data is not None:
                # daily_data是dataFrame类型的数据，获取daily_data中最近days_held天的数据，并且计算收盘价的均值
                daily_data = daily_data.tail(days_held)
                price = daily_data['收盘'].mean()
            price = get_sell_price(price)

            auto.sell(stock_no=stock_no, amount=amount, price=price)



