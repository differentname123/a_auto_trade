# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/14 16:38
:last_date:
    2023/10/14 16:38
:description:
    
"""
def get_ma(stock_data,short_window=20,long_window=50):
    """
    计算股票的MA
    :param stock_data:  股票数据
    :param short_window:  短期均线
    :param long_window:  长期均线
    :return:
    """
    stock_data['Short_MA'] = stock_data['收盘'].rolling(window=short_window).mean()
    stock_data['Long_MA'] = stock_data['收盘'].rolling(window=long_window).mean()
    return stock_data