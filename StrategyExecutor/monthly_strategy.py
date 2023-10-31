# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/19 18:14
:last_date:
    2023/10/19 18:14
:description:

"""

def gen_monthly_buy_signal_one(data):
    """
    macd创period_time范围内新低
    :param data:
    :return:
    """
    macd_window = 3
    # 今日macd创period_window新低
    # 下降趋势减弱(即今日下降量小于昨日下降量)
    # 换手率大于10%

    # 待尝试:
    data['Buy_Signal'] = (data['BAR'].rolling(window=macd_window).min() == data['BAR'])& \
                         (data['换手率'] > 10) & \
                         ((data['BAR'] - data['BAR'].shift(1) > (data['BAR'].shift(1) - data['BAR'].shift(2))))

def gen_monthly_buy_signal_two(data):
    """
    macd变小，但是阳线或者上涨
    :param data:
    :return:
    """
    macd_window = 3
    # 今日macd变低
    # 收阳或者上涨

    # 待尝试:
    # 前一日不能收阳或者上涨
    data['Buy_Signal'] = (data['macd_cha'] < 0) & \
                         (data['macd_cha_rate'] < 0) & \
                         (data['macd_cha_shou_rate'] < 0)

def gen_monthly_buy_signal_mix_one_two(data):

    macd_window = 3
    Buy_Signal_one = (data['BAR'].rolling(window=macd_window).min() == data['BAR'])& \
                         (data['换手率'] > 10) & \
                         ((data['BAR'] - data['BAR'].shift(1) > (data['BAR'].shift(1) - data['BAR'].shift(2))))

    Buy_Signal_two = (data['macd_cha'] < 0) & \
                         (data['macd_cha_rate'] < 0) & \
                         (data['macd_cha_shou_rate'] < 0)

    data['Buy_Signal'] = Buy_Signal_two & Buy_Signal_one


