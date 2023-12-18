# -*- coding: utf-8 -*-
import os

from thsauto import ThsAuto

import time

client_path = r'D:\install\THS\xiadan.exe'

def run_client():
    os.system('start ' + client_path)

if __name__ == '__main__':
    
    auto = ThsAuto()                                        # 连接客户端
    auto.active_mian_window()
    # auto.kill_client()
    # run_client()
    # time.sleep(5)
    auto.bind_client()



    # print('持仓')
    # data = auto.get_position()
    # print(data)
    # stock_nos = []
    # for detail_data in data['data']:
    #     if int(detail_data['实际数量']) > 0:
    #         stock_nos.append(detail_data['证券代码'])
    # print(stock_nos)

    #
    for i in range(10):
        # 开始计时
        start = time.time()
        print('买入')
        result = auto.buy(stock_no='162411', amount=100, price=100)    # 买入股票
        print(result)
        # 结束计时
        end = time.time()
        print('耗时：', end - start)
    #
    # print('已成交')
    # print(auto.get_filled_orders())                                 # 获取已成交订单
    #
    # print('未成交')
    # print(auto.get_active_orders())                                 # 获取未成交订单
    #
    # if result and result['code'] == 0:                                # 如果买入下单成功，尝试撤单
    #     print('撤单')
    #     print(auto.cancel(entrust_no=result['entrust_no']))
    #
