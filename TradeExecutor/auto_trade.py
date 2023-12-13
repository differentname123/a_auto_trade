# -*- coding: utf-8 -*-
import json
import os
import traceback

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
        if price > 2 and price < 40:
            total_price += price
            result.append((stock_no, price))
    print('总价值：', total_price)
    return result


if __name__ == '__main__':

        auto = ThsAuto()                                        # 连接客户端
        auto.active_mian_window()
        # auto.kill_client()
        # run_client()
        # time.sleep(5)
        auto.bind_client()

        print('持仓')
        data = auto.get_position()
        print(data)
        stock_nos = []
        for detail_data in data['data']:
            if int(detail_data['实际数量']) > 0:
                stock_nos.append(detail_data['证券代码'])
        for stock_no in stock_nos:
            price = None
            # 开始计时
            start = time.time()
            print('买入')
            result = auto.buy(stock_no=stock_no, amount=100, price=price)    # 买入股票
            print(result)
            # 结束计时
            end = time.time()
            print('耗时：', end - start)

        stock_and_price = get_stock_and_price('../final_zuhe/select_2023-12-13.json')
        for stock_no, price in stock_and_price:
            # 开始计时
            start = time.time()
            print('买入')
            result = auto.buy(stock_no=stock_no, amount=100, price=price)    # 买入股票
            print(result)
            # 结束计时
            end = time.time()
            print('耗时：', end - start)





