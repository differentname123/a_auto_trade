# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/6 11:56
:last_date:
    2023/10/6 11:56
:description:
    
"""
import akshare as ak

def get_all_a_stock_code():
    """
    获取所有的a股代码(沪深京 A 股)
    :return: a股代码列表
    """
    all_stocks = ak.stock_zh_a_spot_em()
    return all_stocks['代码']


def get_stock_price():

    pass

def fun():
    result = get_all_a_stock_code()
    print(result)


if __name__ == '__main__':
    fun()