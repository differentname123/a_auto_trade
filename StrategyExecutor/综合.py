# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/14 17:00
:last_date:
    2023/10/14 17:00
:description:
    
"""
import os

import numpy as np
import pandas as pd

import pandas as pd
import numpy as np


# 1. Data Loading
def load_data(file_path):
    return pd.read_csv(file_path)


# 2. Indicator Calculation
def calculate_indicators(df):
    short_window = 20
    long_window = 50
    df['Short_MA'] = df['收盘'].rolling(window=short_window).mean()
    df['Long_MA'] = df['收盘'].rolling(window=long_window).mean()
    df['Middle_Band'] = df['收盘'].rolling(window=20).mean()
    df['Upper_Band'] = df['Middle_Band'] + 2 * df['收盘'].rolling(window=20).std()
    df['Lower_Band'] = df['Middle_Band'] - 2 * df['收盘'].rolling(window=20).std()
    df['Buy_Signal'] = np.where(
        (df['Short_MA'] > df['Long_MA']) & (df['收盘'] <= df['Lower_Band']), 1, 0)
    df['Sell_Signal'] = np.where(
        (df['Short_MA'] < df['Long_MA']) & (df['收盘'] >= df['Upper_Band']), -1, 0)
    # df中的最后一个数据的Sell_Signal是0，这里将其改为-1
    df['Sell_Signal'].iloc[-1] = -1
    return df


# 3. Backtesting
def backtest(df, initial_balance=100000):
    balance = initial_balance
    stock_position = 0
    transactions = []
    for index, row in df.iterrows():
        if row['Buy_Signal'] == 1 and stock_position == 0:
            buy_price = row['收盘']
            stock_position = balance / buy_price
            balance = 0
            transactions.append({'日期': row['日期'], '类型': '买入', '价位': buy_price, '盈利': None})
        elif row['Sell_Signal'] == -1 and stock_position > 0:
            sell_price = row['收盘']
            profit = (sell_price - buy_price) * stock_position
            balance += stock_position * sell_price
            stock_position = 0
            transactions.append({'日期': row['日期'], '类型': '卖出', '价位': sell_price, '盈利': profit})
    return transactions


# 4. First Rise Calculation
def days_until_price_rise(data, start_index, buy_price):
    for i in range(start_index, len(data)):
        if data['收盘'].iloc[i] > buy_price:
            return i - start_index
    return None


def adjusted_backtest(df, initial_balance=100000):
    balance = initial_balance
    stock_position = 0
    buy_price = 0
    transactions = []

    for index, row in df.iterrows():
        if row['Buy_Signal'] == 1 and stock_position == 0:
            buy_price = row['收盘']
            stock_position = balance / buy_price
            balance = 0
            transactions.append({'日期': row['日期'], '类型': '买入', '价位': buy_price, '盈利': None})
            # 打印买入信息
            print('买入信息：')
            print('日期：', row['日期'])
            print('价位：', buy_price)
        elif stock_position > 0 and row['收盘'] > buy_price:
            sell_price = row['收盘']
            profit = (sell_price - buy_price) * stock_position
            balance += stock_position * sell_price
            stock_position = 0
            transactions.append({'日期': row['日期'], '类型': '卖出', '价位': sell_price, '盈利': profit})

    return transactions

def calculate_first_rise(transactions, df, initial_balance):
    detailed_transactions = []
    for transaction in transactions:
        if transaction['类型'] == '买入':
            current_buy_date = transaction['日期']
            current_buy_price = transaction['价位']
            buy_date_index = df[df['日期'] == current_buy_date].index[0]
            days_to_first_rise = days_until_price_rise(df, buy_date_index, current_buy_price)
            if days_to_first_rise:
                first_rise_date = df['日期'].iloc[buy_date_index + days_to_first_rise]
            else:
                first_rise_date = None
        elif transaction['类型'] == '卖出':
            sell_date = transaction['日期']
            sell_price = transaction['价位']
            days_to_sell = (pd.to_datetime(sell_date) - pd.to_datetime(current_buy_date)).days
            profit = (sell_price - current_buy_price) * (initial_balance / current_buy_price)
            detailed_transactions.append({
                '买入日期': current_buy_date,
                '买入价位': current_buy_price,
                '首次超买价日期': first_rise_date,
                '首次超买价天数': days_to_first_rise,
                '卖出日期': sell_date,
                '卖出价位': sell_price,
                '从买入到卖出天数': days_to_sell,
                '盈利': profit
            })
    return pd.DataFrame(detailed_transactions)


def pretty_print_transactions(transactions_df):
    # Adjust the data for better display
    transactions_df = transactions_df.copy()  # Creating a copy to avoid modifying original data

    # Check if the data is already formatted
    if "￥" not in str(transactions_df['买入价位'].iloc[0]):
        transactions_df['买入价位'] = transactions_df['买入价位'].apply(lambda x: f"￥{x:.2f}")
        transactions_df['卖出价位'] = transactions_df['卖出价位'].apply(lambda x: f"￥{x:.2f}")
        transactions_df['盈利'] = transactions_df['盈利'].apply(lambda x: f"￥{x:,.2f}")

    # Function to compute display length considering Chinese characters
    def display_length(s):
        return sum(2 if ord(c) > 256 else 1 for c in s)

    # Adjust column width for better alignment considering Chinese characters
    col_widths = {
        col: max(display_length(str(v)) for v in transactions_df[col].tolist() + [col])
        for col in transactions_df.columns
    }

    # Format dataframe strings with adjusted widths
    for col, width in col_widths.items():
        transactions_df[col] = transactions_df[col].apply(
            lambda x: f"{x:<{width}}"
        )

    # Using tabulate to create a pretty table with adjusted widths
    from tabulate import tabulate
    table = tabulate(transactions_df, headers='keys', tablefmt='grid', showindex=False)
    return table

# Main function to execute the entire strategy
def main_strategy(file_path, initial_balance=100000):
    # Load the data
    df = load_data(file_path)

    # Calculate the necessary indicators
    df = calculate_indicators(df)

    # Perform backtesting to get transactions
    transactions = adjusted_backtest(df, initial_balance)

    # Calculate the first rise details after buying
    detailed_transactions = calculate_first_rise(transactions, df, initial_balance)
    print(pretty_print_transactions(detailed_transactions))
    return detailed_transactions

if __name__ == "__main__":
    # 扫描../InfoCollector/stock_data/daily/下的所有股票数据
    # 逐个执行main_strategy
    # 将结果保存到../InfoCollector/stock_data/strategy/下
    # 保存格式为csv，文件名为股票代码
    main_strategy("../InfoCollector/stock_data/daily/000682_东方电子.csv")
    # for file in os.listdir("../InfoCollector/stock_data/daily/"):
    #     if file.endswith(".csv"):
    #         print(file)
    #         df = main_strategy("../InfoCollector/stock_data/daily/" + file)
    #         df.to_csv("../InfoCollector/stock_data/strategy/" + file, index=False)
