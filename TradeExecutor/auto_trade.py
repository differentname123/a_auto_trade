# -*- coding: utf-8 -*-
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


if __name__ == '__main__':

        # auto = ThsAuto()                                        # 连接客户端
        # auto.active_mian_window()
        # # auto.kill_client()
        # # run_client()
        # # time.sleep(5)
        # auto.bind_client()


        target_date = datetime.now().strftime('%Y-%m-%d')
        output_file_path = '../final_zuhe/select/select_{}.txt'.format(target_date)
        save_and_analyse_all_data_mul(target_date)

        # select_file_path = '../final_zuhe/select/select_2023-12-22.json'
        # stock_and_price = get_stock_and_price(select_file_path)
        # for stock_no, price in stock_and_price:
        #     # 开始计时
        #     start = time.time()
        #     print('买入')
        #     result = auto.buy(stock_no=stock_no, amount=100, price=price)    # 买入股票
        #     print(result)
        #     # 结束计时
        #     end = time.time()
        #     print('耗时：', end - start)
        # # 复制select_file_path，名字增加当前时间戳
        # import shutil
        # shutil.copy(select_file_path, select_file_path.replace('.json', '_' + str(int(time.time())) + '.json'))





