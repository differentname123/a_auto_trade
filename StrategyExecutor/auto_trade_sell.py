# -*- coding: utf-8 -*-
import multiprocessing
from datetime import datetime

import json
import os
import traceback

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
        amount = detail_data['可用余额']
        price = detail_data['price']
        auto.sell(stock_no=detail_data['stock_no'], amount=detail_data['amount'], price=detail_data['price'])
        auto.cancel_all(entrust_no=detail_data['entrust_no'])



