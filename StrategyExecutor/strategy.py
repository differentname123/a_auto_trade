# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/19 18:13
:last_date:
    2023/10/19 18:13
:description:
    
"""
from common import *

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.expand_frame_repr', False)  # 确保不会因为宽度而换行
pd.set_option('display.max_rows', None)
def init():
    data = load_data('../daily_data_exclude_new/ST宋都_600077.txt')
    get_indicators(data)
    return data

def strategy(file_path, gen_signal_func=gen_buy_signal_one, backtest_func=backtest_strategy_highest):
    # 加载数据
    data = load_data(file_path)

    # 计算指标
    get_indicators(data)

    # 产生买入信号
    gen_signal_func(data)

    # 获取回测结果
    results_df = backtest_func(data)

    # 控制台打印回测结果，按照时间升序,不要省略
    print(results_df)

    # 控制台打印回测结果，按照持有天数逆序
    result = results_df.sort_values(by='Days Held', ascending=False)
    print(result)


    # k线 显示回测结果
    show_image(data, results_df)

def back_one(file_path, gen_signal_func=gen_buy_signal_one, backtest_func=backtest_strategy_highest):
    """
    获取一个文件的回测结果
    :param file_path:
    :param gen_signal_func: 生成买入信号的函数，默认为gen_buy_signal
    :param backtest_func: 回测策略的函数，默认为backtest_strategy_highest
    :return:
    """
    data = load_data(file_path)
    get_indicators(data)
    gen_signal_func(data)
    results_df = backtest_func(data)
    return results_df


def get_buy_signal():
    find_buy_signal('../daily_data_exclude_new', target_time='2023-10-18')


def back_all_stock(file_path, output_file_path, threshold_day=5, gen_signal_func=gen_buy_signal_one, backtest_func=backtest_strategy_highest):
    """
    回测所有的股票，将持股时间大于阈值的记录收集起来,交易次数和交易总收益也统计起来
    :param file_path:
    :return:
    """
    all_dfs = []  # 存放所有满足条件的DataFrame

    trade_count = 0
    total_profit = 0

    # 获取文件总数
    total_files = sum([len(files) for r, d, files in os.walk(file_path)])
    processed_files = 0

    for root, ds, fs in os.walk(file_path):
        for f in fs:
            try:
                fullname = os.path.join(root, f)
                temp_back_df = back_one(fullname, gen_signal_func, backtest_func)
                # 筛选出'Days Held'大于threshold_day的记录
                if not temp_back_df.empty:
                    trade_count += temp_back_df.shape[0]
                    total_profit += temp_back_df['Total_Profit'].iloc[-1]
                filtered_df = temp_back_df[temp_back_df['Days Held'] > threshold_day]
                if not filtered_df.empty:  # 只有当filtered_df不为空时，才添加到all_dfs
                    all_dfs.append(filtered_df)
                    print(filtered_df)
            except Exception as e:
                print(f"Error occurred: {e}")

            # 更新并打印已回测的文件数量
            processed_files += 1
            print(f"Processed {processed_files}/{total_files} files.")

    # 将所有满足条件的DataFrame合并成一个大的DataFrame
    result_df = pd.concat(all_dfs, ignore_index=True)
    # 计算result_df的总Days Held
    total_days_held = result_df['Days Held'].sum()

    # 保存 result_df 到一个文件中
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')  # 获取到分钟的时间戳
    filename = f"{gen_signal_func.__name__}_{backtest_func.__name__}_{timestamp}.txt"
    with open(output_file_path + '/' + filename, 'w') as f:
        result_df.to_string(f)
        f.write('\n\n')
        f.write(f"trade_count: {trade_count}\n")
        f.write(f"total_profit: {total_profit}\n")
        f.write(f"size of result_df: {result_df.shape[0]}\n")
        f.write(f"ratio: {result_df.shape[0] / trade_count if trade_count > 0 else 0}\n")
        f.write(f"average days_held: {total_days_held / trade_count}\n")
        f.write(f"average profit: {total_profit / trade_count}\n")

    return result_df

if __name__ == "__main__":
    # # 策略1 macd新低买入
    strategy('../InfoCollector/stock_data_exclude/daily/000682_东方电子.csv',gen_signal_func=gen_buy_signal_seven, backtest_func=backtest_strategy_highest)

    # 回测所有数据
    # back_all_stock('../InfoCollector/stock_data_exclude/daily', '../InfoCollector/back', gen_signal_func=gen_buy_signal_seven,backtest_func=backtest_strategy_highest_fix)

    # 获取指定日期买入信号的symbol
    # get_buy_signal()