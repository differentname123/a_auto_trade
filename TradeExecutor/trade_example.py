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
    auto.kill_client()
    run_client()
    time.sleep(5)
    auto.bind_client()

    print('可用资金')
    print(auto.get_balance())                               # 获取当前可用资金

    print('持仓')
    print(auto.get_position())                              # 获取当前持有的股票

    print('卖出')
    print(auto.sell(stock_no='000573', amount=100, price=3.5))   # 卖出股票

    print('买入')
    result = auto.buy(stock_no='162411', amount=100, price=0.41)    # 买入股票
    print(result)

    print('已成交')
    print(auto.get_filled_orders())                                 # 获取已成交订单
    
    print('未成交')
    print(auto.get_active_orders())                                 # 获取未成交订单

    if result and result['code'] == 0:                                # 如果买入下单成功，尝试撤单
        print('撤单')
        print(auto.cancel(entrust_no=result['entrust_no']))

