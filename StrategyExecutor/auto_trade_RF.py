# -*- coding: utf-8 -*-
import multiprocessing
from datetime import datetime

import json
import os
import traceback

from StrategyExecutor.full_zehe import save_and_analyse_all_data_mul, save_and_analyse_all_data_mul_real_time, \
    save_and_analyse_all_data_mul_real_time_RF
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

def get_order_info(file_path):
    result = {}
    if not os.path.exists(file_path):
        return result
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                stock_no, price = line.strip().split(',')
                price = float(price)
                if stock_no not in result:
                    result[stock_no] = []
                result[stock_no].append(price)
            except Exception:
                traceback.print_exc()
                print(line)
    return result

def statistic_good_price(file_path):
    good_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            stock_no, min_price, max_price, current_price = line.strip().split(',')
            # 将price转换为float类型并且保留两位小数
            min_price = round(float(min_price), 2)
            max_price = round(float(max_price), 2)
            current_price = round(float(current_price), 2)
            if current_price <= max_price:
                good_list.append((stock_no, current_price))
    print('good_list:', good_list)

def process_stock_data(order_output_file_path, file_path, auto, amount):
    order_result = get_order_info(order_output_file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            try:
                stock_no, min_price, max_price, current_price = line.strip().split(',')
                # 将price转换为float类型并且保留两位小数
                min_price = round(float(min_price), 2)
                max_price = round(float(max_price), 2)
                current_price = round(float(current_price), 2)
                price = current_price
                if min_price >= current_price:
                    price = current_price
                if max_price <= current_price:
                    price = max_price
                price += 0.01
                if stock_no not in order_result:
                    # 将stock_no, price写入到order_output_file_path文件中
                    with open(order_output_file_path, 'a', encoding='utf-8') as f:
                        f.write('{},{}\n'.format(stock_no, price))
                    result = auto.quick_buy(stock_no=stock_no, amount=amount, price=price)
                else:
                    if price < min(order_result[stock_no]):
                        with open(order_output_file_path, 'a', encoding='utf-8') as f:
                            f.write('{},{}\n'.format(stock_no, price))
                        result = auto.quick_buy(stock_no=stock_no, amount=amount, price=price)
            except Exception:
                traceback.print_exc()
                print(line)
    if os.path.exists(file_path):
        os.remove(file_path)


def is_time_between(start_time, end_time, check_time=None):
    """检查当前时间是否在给定的时间范围内"""
    check_time = check_time or datetime.now().time()
    if start_time < end_time:
        return start_time <= check_time <= end_time
    else: # 跨过午夜
        return check_time >= start_time or check_time <= end_time

if __name__ == '__main__':
    # target_date = datetime.now().strftime('%Y-%m-%d')
    # output_file_path = '../final_zuhe/select/{}real_time_good_price.txt'.format(target_date)
    # statistic_good_price(output_file_path)
    auto = ThsAuto()  # 连接客户端
    auto.active_mian_window()
    auto.bind_client()
    auto.init_buy()
    amount = 100

    while True:
        # 检查当前时间是否在9:30到15:00之间
        if is_time_between(datetime.strptime("14:56", "%H:%M").time(),datetime.strptime("15:00", "%H:%M").time()):
            # 在这里执行您的主要逻辑
            target_date = datetime.now().strftime('%Y-%m-%d')
            output_file_path = '../final_zuhe/select/{}real_time_good_price.txt'.format(target_date)
            order_output_file_path = '../final_zuhe/select/{}_real_time_RF_order.txt'.format(target_date)

            delete_file_if_exists(output_file_path)

            process = multiprocessing.Process(target=save_and_analyse_all_data_mul_real_time_RF, args=(target_date,))
            process.start()

            while process.is_alive():
                if os.path.exists(output_file_path):
                    process_stock_data(order_output_file_path, output_file_path, auto, amount)
                else:
                    time.sleep(1)

            if os.path.exists(output_file_path):
                process_stock_data(order_output_file_path, output_file_path, auto, amount)
        else:
            # 如果不在运行时间内，则等待一段时间（例如30秒）后再次检查
            # 尝试将process杀死
            try:
                if process.is_alive():
                    process.terminate()
                else:
                    print('进程已经结束')
            except Exception:
                pass
            time.sleep(10)



