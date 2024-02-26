# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/19 18:13
:last_date:
    2023/10/19 18:13
:description:

"""
import time
import inspect

from StrategyExecutor.basic_daily_strategy import *
from StrategyExecutor.monthly_strategy import *
from StrategyExecutor.zuhe_daily_strategy import *
from common import *
from concurrent.futures import ThreadPoolExecutor
import os
from dateutil.relativedelta import relativedelta
import multiprocessing
import pandas as pd

from daily_strategy import *

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.expand_frame_repr', False)  # 确保不会因为宽度而换行
pd.set_option('display.max_rows', None)


def init():
    data = load_data('../daily_data_exclude_new_can_buy/ST宋都_600077.txt')
    get_indicators(data)
    return data


def strategy(file_path, gen_signal_func=gen_buy_signal_one, backtest_func=backtest_strategy_highest, threshold_day=1):
    # 加载数据
    data = load_data(file_path)

    # 将data数据只保留1996到1997年的数据
    # data = data[data['日期'] >= '1996-01-01']
    # data = data[data['日期'] <= '1997-01-01']

    # 计算指标
    # get_indicators(data, True)

    # 产生买入信号
    gen_signal_func(data)

    # 获取回测结果
    results_df = backtest_func(data)

    # 控制台打印回测结果，按照时间升序,不要省略
    print(results_df)
    if results_df.shape[0] != 0:
        Total_Profit = results_df['Total_Profit'].iloc[-1]
    total_days_held = results_df['Days Held'].sum()
    # 控制台打印回测结果，按照持有天数逆序
    result = results_df.sort_values(by='Days Held', ascending=True)
    print(result)
    if result.shape[0] == 0:
        return
    result_df = result[result['Days Held'] > threshold_day]

    print(f"trade_count: {result.shape[0]}")
    print(f"total_profit: {Total_Profit}")
    print(f"size of result_df: {result_df.shape[0]}")
    print(f"ratio: {result_df.shape[0] / result.shape[0] if result.shape[0] > 0 else 0}")
    print(f"average days_held: {total_days_held / result.shape[0]}")
    print(f"average profit: {result['Total_Profit'].iloc[-1] / result.shape[0]}")

    # k线 显示回测结果
    # show_image(data, results_df)
    # show_k(data, results_df)


def strategy_mix(small_period_file_path, big_period_file_path, biggest_period_file_path,
                 gen_small_period_signal_func=gen_buy_signal_eight,
                 gen_big_period_signal_func=gen_buy_signal_weekly_eight,
                 gen_biggest_period_signal_func=gen_buy_signal_weekly_eight, backtest_func=backtest_strategy_highest):
    # 加载数据
    small_period_data = load_data(small_period_file_path)
    big_period_data = load_data(big_period_file_path)
    biggest_period_data = load_data(biggest_period_file_path)

    # 计算指标
    get_indicators(small_period_data)
    get_indicators(big_period_data)
    get_indicators(biggest_period_data)

    # 产生买入信号
    gen_small_period_signal_func(small_period_data)
    gen_big_period_signal_func(big_period_data)
    gen_biggest_period_signal_func(biggest_period_data)
    mix_small_period_and_big_period_data(big_period_data, biggest_period_data)
    mix_small_period_and_big_period_data(small_period_data, big_period_data)

    # 获取回测结果
    results_df = backtest_func(small_period_data)

    # 控制台打印回测结果，按照时间升序,不要省略
    print(results_df)

    # 控制台打印回测结果，按照持有天数逆序
    result = results_df.sort_values(by='Days Held', ascending=False)
    print(result)

    # k线 显示回测结果
    # show_image(data, results_df)
    show_k(small_period_data, results_df)


def back_one(file_path, gen_signal_func=gen_buy_signal_one, backtest_func=backtest_strategy_highest):
    """
    获取一个文件的回测结果
    :param file_path:
    :param gen_signal_func: 生成买入信号的函数，默认为gen_buy_signal
    :param backtest_func: 回测策略的函数，默认为backtest_strategy_highest
    :return:
    """
    data = load_data(file_path)
    # get_indicators(data)
    gen_signal_func(data)
    results_df = backtest_func(data)
    return results_df


def back_one_mix(small_period_file_path, big_period_file_path, biggest_period_file_path,
                 gen_small_period_signal_func=gen_monthly_buy_signal_one,
                 gen_big_period_signal_func=gen_monthly_buy_signal_one,
                 gen_biggest_period_signal_func=gen_monthly_buy_signal_one, backtest_func=backtest_strategy_highest):
    """
    获取一个文件的回测结果(使用daily和week数据)
    :param file_path:
    :param gen_signal_func: 生成买入信号的函数，默认为gen_buy_signal
    :param backtest_func: 回测策略的函数，默认为backtest_strategy_highest
    :return:
    """
    small_period_data = load_data(small_period_file_path)
    get_indicators(small_period_data)
    gen_small_period_signal_func(small_period_data)

    big_period_data = load_data(big_period_file_path)
    get_indicators(big_period_data)
    gen_big_period_signal_func(big_period_data)

    biggest_period_data = load_data(biggest_period_file_path)
    get_indicators(biggest_period_data)
    gen_biggest_period_signal_func(biggest_period_data)

    mix_small_period_and_big_period_data(big_period_data, biggest_period_data)

    mix_small_period_and_big_period_data(small_period_data, big_period_data)

    results_df = backtest_func(small_period_data)
    return results_df


def get_buy_signal(file_path, target_time, gen_signal_func=gen_daily_buy_signal_seventeen):
    """
    遍历file_path下的说有数据，并读取，应用策略，找到target_time有买入信号的symbol
    :param file_path:
    :param target_time:
    :return:
    """
    if target_time is None:
        target_time = datetime.datetime.now().strftime('%Y-%m-%d')
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            try:
                fullname = os.path.join(root, f)
                # 加载数据
                data = load_data(fullname)
                # 计算指标
                get_indicators(data, True)

                # 产生买入信号
                stock_data_df = gen_signal_func(data)
                stock_data_df['日期'] = stock_data_df['日期'].astype(str)
                filtered_row = stock_data_df[
                    (stock_data_df['Buy_Signal'] == True) & (stock_data_df['日期'].str.startswith(target_time))]
                if filtered_row.shape[0] > 0:
                    print(f)
                    print(filtered_row)
                    print('\n')
            except Exception as e:
                print(e)
                print(fullname)


def process_single_file(args):
    fullname, gen_signal_func, backtest_func, threshold_day = args
    try:
        temp_back_df = back_one(fullname, gen_signal_func, backtest_func)
        total_cost = 0
        if not temp_back_df.empty:
            trade_count = temp_back_df.shape[0]
            total_profit = temp_back_df['Total_Profit'].iloc[-1]
            total_cost = temp_back_df['total_cost'].sum()
            # filtered_df = temp_back_df[temp_back_df['Days Held'] > threshold_day]
            filtered_df = temp_back_df
            if not filtered_df.empty:
                return filtered_df, trade_count, total_profit, total_cost
            return None, trade_count, total_profit, total_cost
    except Exception as e:
        print(f"{fullname} Error occurred: {e}")
    return None, 0, 0, 0


def back_all_stock(file_path, output_file_path, threshold_day=1, gen_signal_func=gen_buy_signal_one,
                   backtest_func=backtest_strategy_highest, is_keep_all=True):
    all_dfs = []
    trade_count = 0
    total_profit = 0
    total_cost = 0
    start_time = time.time()

    file_args = []
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            fullname = os.path.join(root, f)
            file_args.append((fullname, gen_signal_func, backtest_func, threshold_day))

    with multiprocessing.Pool() as pool:
        results = pool.map(process_single_file, file_args)

    for res in results:
        df, count, profit, cost = res
        trade_count += count
        total_profit += profit
        total_cost += cost
        if df is not None:
            all_dfs.append(df)

    # 如果all_dfs为空，说明没有任何一只股票满足条件
    total_days_held = 0
    result_df_size = 0
    result_df = None
    three_befor_year_count = 0
    two_befor_year_count = 0
    one_befor_year_count = 0
    three_befor_year_count_thread = 0
    # 统计'Buy Date'大于三年前的数据
    three_year = datetime.datetime.now() - relativedelta(years=2)
    # 将three_year只保留到年份，不要月和日
    three_befor_year = three_year.year
    two_year = datetime.datetime.now() - relativedelta(years=1)
    # 将three_year只保留到年份，不要月和日
    two_befor_year = two_year.year
    one_befor_year = datetime.datetime.now().year
    all_date = []
    try:
        all_df = pd.concat(all_dfs, ignore_index=True)
        result_df = all_df[all_df['Days Held'] > threshold_day]
        result_df = result_df.sort_values(by='Days Held', ascending=False)
        total_days_held = result_df['Days Held'].sum() + trade_count - result_df.shape[0]
        result_df_size = result_df.shape[0]
        three_befor_year_count = all_df[all_df['Buy Date'].dt.year >= three_befor_year].shape[0]
        two_befor_year_count = all_df[all_df['Buy Date'].dt.year >= two_befor_year].shape[0]
        one_befor_year_count = all_df[all_df['Buy Date'].dt.year >= one_befor_year].shape[0]
        three_befor_year_count_thread = result_df[result_df['Buy Date'].dt.year >= three_befor_year].shape[0]
        # 将all_df保存到文件中，尽量一个数据都在同一行
        all_df.to_csv(os.path.join(output_file_path, 'all_df.csv'), index=False)
        # # 将all_df['Buy Date']转为字符串
        # all_df['Buy Date'] = all_df['Buy Date'].astype(str)
        all_date = all_df['Buy Date'].tolist()
        # 将all_date变成set
        all_date = set(all_date)
    except Exception as e:
        print(e)

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"{gen_signal_func.__name__}_{backtest_func.__name__}.txt"
    full_path = os.path.join(output_file_path, filename)
    if not os.path.exists(full_path):
        open(full_path, 'w').close()

    with open(full_path, 'r') as f:
        existing_content = f.read()
    with open(full_path, 'w') as f:
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"trade_count: {trade_count}\n")
        f.write(f"total_profit: {total_profit}\n")
        f.write(f"total_cost: {total_cost}\n")
        f.write(f"size of result_df: {result_df_size}\n")
        f.write(f"ratio: {result_df_size / trade_count if trade_count > 0 else 0}\n")
        f.write(f"average days_held: {total_days_held / trade_count if trade_count > 0 else 0}\n")
        f.write(f"average profit: {total_profit / trade_count if trade_count > 0 else 0}\n")
        f.write(f"average 1w profit: {total_profit * 10000 / total_cost if total_cost > 0 else 0}\n")
        f.write(f"one_befor_year_count: {one_befor_year_count}\n")
        f.write(f"two_befor_year_count: {two_befor_year_count}\n")
        f.write(f"three_befor_year_count: {three_befor_year_count}\n")
        f.write(f"three_befor_year_rate: {three_befor_year_count / trade_count if trade_count > 0 else 0}\n")
        f.write(f"three_befor_year_count_thread: {three_befor_year_count_thread}\n")
        f.write(f"three_befor_year_count_thread_ratio: {three_befor_year_count_thread / three_befor_year_count if three_befor_year_count > 0 else 0}\n")
        f.write(f"two_befor_year_rate: {two_befor_year_count / trade_count if trade_count > 0 else 0}\n")
        f.write(f"one_befor_year_rate: {one_befor_year_count / trade_count if trade_count > 0 else 0}\n")


        f.write(f"date_count: {len(all_date)}\n")
        f.write(f"date_ratio: {round(len(all_date) / trade_count, 4)}\n")
        f.write(f"start_date: {min(all_date)}\n")
        f.write(f"end_date: {max(all_date)}\n")
        # Save the source code of gen_signal_func
        f.write("Source code for 'gen_signal_func':\n")
        f.write(inspect.getsource(gen_signal_func))
        f.write("\n\n")
        f.write('\n\n')
        f.write('\n\n')
        f.write(existing_content)

    filename = f"{timestamp}_{gen_signal_func.__name__}_{backtest_func.__name__}.txt"
    full_path = os.path.join(output_file_path + '/single', filename)
    if not os.path.exists(full_path):
        open(full_path, 'w').close()
    with open(full_path, 'w') as f:
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"trade_count: {trade_count}\n")
        f.write(f"total_profit: {total_profit}\n")
        f.write(f"total_cost: {total_cost}\n")
        f.write(f"size of result_df: {result_df_size}\n")
        f.write(f"ratio: {result_df_size / trade_count if trade_count > 0 else 0}\n")
        f.write(f"average days_held: {total_days_held / trade_count if trade_count > 0 else 0}\n")
        f.write(f"average profit: {total_profit / trade_count if trade_count > 0 else 0}\n")
        f.write(f"average 1w profit: {total_profit * 10000 / total_cost if total_cost > 0 else 0}\n")
        f.write(f"one_befor_year_count: {one_befor_year_count}\n")
        f.write(f"two_befor_year_count: {two_befor_year_count}\n")
        f.write(f"three_befor_year_count: {three_befor_year_count}\n")
        f.write(f"three_befor_year_rate: {three_befor_year_count / trade_count if trade_count > 0 else 0}\n")
        f.write(f"two_befor_year_rate: {two_befor_year_count / trade_count if trade_count > 0 else 0}\n")
        f.write(f"one_befor_year_rate: {one_befor_year_count / trade_count if trade_count > 0 else 0}\n")
        f.write(f"three_befor_year_count_thread: {three_befor_year_count_thread}\n")
        f.write(f"three_befor_year_count_thread_ratio: {three_befor_year_count_thread / three_befor_year_count if three_befor_year_count > 0 else 0}\n")
        f.write(f"date_count: {len(all_date)}\n")
        f.write(f"date_ratio: {round(len(all_date) / trade_count, 4)}\n")
        f.write(f"start_date: {min(all_date)}\n")
        f.write(f"end_date: {max(all_date)}\n")
        # Save the source code of gen_signal_func
        f.write("Source code for 'gen_signal_func':\n")
        f.write(inspect.getsource(gen_signal_func))
        f.write("\n\n")
        f.write('\n\n')
        if is_keep_all:
            all_df.to_string(f)
        if result_df is not None:
            result_df.to_string(f)
    data = process_results_with_every_period(all_df, 1)
    for k, v in data.items():
        v['ratio'] = v['size_of_result_df'] / v['trade_count'] if v['trade_count'] > 0 else 0
    print(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function executed in: {elapsed_time:.2f} seconds")
    return result_df


def back_mix_for_one_file(f, small_period_file_path, big_period_file_path, gen_small_period_signal_func,
                          gen_big_period_signal_func, backtest_func, threshold_day):
    try:
        small_period_fullname = os.path.join(small_period_file_path, f)
        big_period_fullname = os.path.join(big_period_file_path, f)
        temp_back_df = back_one_mix(small_period_fullname, big_period_fullname, gen_small_period_signal_func,
                                    gen_big_period_signal_func, backtest_func)
        # 筛选出'Days Held'大于threshold_day的记录
        filtered_df = temp_back_df[temp_back_df['Days Held'] > threshold_day]
        if not filtered_df.empty:
            return filtered_df, temp_back_df.shape[0], temp_back_df['Total_Profit'].iloc[
                -1] if not temp_back_df.empty else 0
    except Exception as e:
        print(f"Error occurred with file {f}: {e}")

    return None, 0, 0


def process_file(args):
    small_period_fullname, big_period_fullname, biggest_period_fullname, gen_small_period_signal_func, gen_big_period_signal_func, gen_biggest_period_signal_func, backtest_func, threshold_day = args
    try:
        temp_back_df = back_one_mix(small_period_fullname, big_period_fullname, biggest_period_fullname,
                                    gen_small_period_signal_func, gen_big_period_signal_func,
                                    gen_biggest_period_signal_func, backtest_func)
        if not temp_back_df.empty:
            trade_count = temp_back_df.shape[0]
            total_profit = temp_back_df['Total_Profit'].iloc[-1]
            filtered_df = temp_back_df[temp_back_df['Days Held'] > threshold_day]
            if not filtered_df.empty:
                return filtered_df, trade_count, total_profit
            else:
                return None, trade_count, total_profit
    except Exception as e:
        print(f"{small_period_fullname}Error occurred: {e}")
    return None, 0, 0


def back_mix_all_stock_process(small_period_file_path, big_period_file_path, biggest_period_file_path, output_file_path,
                               threshold_day=1, gen_small_period_signal_func=gen_monthly_buy_signal_one,
                               gen_big_period_signal_func=gen_monthly_buy_signal_one,
                               gen_biggest_period_signal_func=gen_monthly_buy_signal_one,
                               backtest_func=backtest_strategy_highest):
    all_dfs = []
    trade_count = 0
    total_profit = 0
    start_time = time.time()

    file_args = []
    for root, ds, fs in os.walk(small_period_file_path):
        for f in fs:
            small_period_fullname = os.path.join(root, f)
            big_period_fullname = os.path.join(big_period_file_path, f)
            biggest_period_fullname = os.path.join(biggest_period_file_path, f)
            file_args.append((small_period_fullname, big_period_fullname, biggest_period_fullname,
                              gen_small_period_signal_func, gen_big_period_signal_func, gen_biggest_period_signal_func,
                              backtest_func, threshold_day))

    # Use multiprocessing to process files
    with multiprocessing.Pool() as pool:
        results = pool.map(process_file, file_args)

    for res in results:
        df, count, profit = res
        trade_count += count
        total_profit += profit
        if df is not None:
            all_dfs.append(df)

    result_df = pd.concat(all_dfs, ignore_index=True)
    total_days_held = result_df['Days Held'].sum()

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"{timestamp}_{gen_small_period_signal_func.__name__}_{backtest_func.__name__}.txt"
    with open(os.path.join(output_file_path, filename), 'w') as f:
        f.write(f"trade_count: {trade_count}\n")
        f.write(f"total_profit: {total_profit}\n")
        f.write(f"size of result_df: {result_df.shape[0]}\n")
        f.write(f"ratio: {result_df.shape[0] / trade_count if trade_count > 0 else 0}\n")
        f.write(f"average days_held: {total_days_held / trade_count}\n")
        f.write(f"average profit: {total_profit / trade_count}\n")
        # Save the source code of gen_signal_func
        f.write("Source code for 'gen_small_period_signal_func':\n")
        f.write(inspect.getsource(gen_small_period_signal_func))
        f.write("\n\n")
        f.write("Source code for 'gen_big_period_signal_func':\n")
        f.write(inspect.getsource(gen_big_period_signal_func))
        f.write("\n\n")
        f.write("Source code for 'gen_biggest_period_signal_func':\n")
        f.write(inspect.getsource(gen_biggest_period_signal_func))
        f.write("\n\n")
        result_df.to_string(f)
        f.write('\n\n')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function executed in: {elapsed_time:.2f} seconds")
    return result_df


def show_image(file_path, gen_signal_func=gen_buy_signal_one, backtest_func=backtest_strategy_highest):
    data = load_data(file_path)

    # 计算指标
    get_indicators(data)

    # 产生买入信号
    gen_signal_func(data)

    # 获取回测结果
    results_df = backtest_func(data)
    show_k(data, results_df)



if __name__ == "__main__":
    # # daily macd新低买入
    # strategy('../daily_data_exclude_new_can_buy/力佳科技_835237.txt', gen_signal_func=mix,backtest_func=backtest_strategy_low_profit)

    # 各种组合的遍历
    # back_zuhe('../daily_data_exclude_new_can_buy/C润本_603193.txt.txt', backtest_func=backtest_strategy_low_profit)
    # back_zuhe_all('../daily_data_exclude_new_can_buy', backtest_func=backtest_strategy_low_profit)
    # back_sigle_all('../daily_data_exclude_new_can_buy', gen_signal_func=gen_full_all_basic_signal,backtest_func=backtest_strategy_low_profit)
    back_layer_all_op('../daily_data_exclude_new_can_buy', gen_signal_func=gen_full_all_basic_signal, backtest_func=backtest_strategy_low_profit)
    # back_layer_all_op_gen('../daily_data_exclude_new_can_buy', gen_signal_func=gen_full_all_basic_signal, backtest_func=backtest_strategy_low_profit)
    # statistics_zuhe('../back/zuhe', target_key="target_key")

    # statistics = read_json('../back/statistics.json')
    # # 遍历statistics,将key以':'分割得到list,将list长度小于3的key和value加入到新的dict中
    # statistics_new = {}
    # for key, value in statistics.items():
    #     if len(key.split(':')) < 3:
    #         statistics_new[key] = value
    # write_json('../back/statistics_new.json', statistics_new)

    # mix 买入
    # strategy_mix('../daily_data_exclude_new_can_buy/龙洲股份_002682.txt', '../weekly_data_exclude_new/中油工程_600339.txt', '../monthly_data_exclude_new/中油工程_600339.txt', gen_small_period_signal_func=gen_buy_signal_four, gen_big_period_signal_func=gen_buy_signal_four, gen_biggest_period_signal_func=gen_monthly_buy_signal_one, backtest_func=backtest_strategy_highest_buy_all)
    # strategy_mix('../weekly_data_exclude_new/黑牡丹_600510.txt', '../monthly_data_exclude_new/黑牡丹_600510.txt', gen_small_period_signal_func=gen_monthly_buy_signal_one, gen_big_period_signal_func=gen_monthly_buy_signal_one, backtest_func=backtest_strategy_highest)

    # 回测所有数据
    # back_all_stock('../daily_data_exclude_new_can_buy/', '../back/complex', gen_signal_func=mix, backtest_func=backtest_strategy_low_profit)
    # back_mix_all_stock_process('../daily_data_exclude_new_can_buy/', '../weekly_data_exclude_new/','../monthly_data_exclude_new/', '../back', gen_small_period_signal_func=gen_monthly_buy_signal_mix_one_two, gen_big_period_signal_func=gen_monthly_buy_signal_mix_one_two, gen_biggest_period_signal_func=gen_true, backtest_func=backtest_strategy_highest_buy_all)

    # 获取指定日期买入信号的symbol
    # get_buy_signal('../daily_data_exclude_new_can_buy/', '2023-11-17', gen_signal_func=gen_daily_buy_signal_seventeen)

    # 显示相应的图像
    # show_image('../InfoCollector/daily_data_exclude_new_can_buy/合力科技_603917.txt')
