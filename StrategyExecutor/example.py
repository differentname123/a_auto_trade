import numpy as np
import pandas as pd

# 1. 数据加载
df_long = pd.read_csv("../InfoCollector/stock_data/daily/000682_东方电子.csv")

# # 取df_long的最近1000个
df_long = df_long[300:]
# 2. 计算移动平均线和布林带
short_window = 20
long_window = 50
df_long['Short_MA'] = df_long['收盘'].rolling(window=short_window).mean()
df_long['Long_MA'] = df_long['收盘'].rolling(window=long_window).mean()
df_long['Middle_Band'] = df_long['收盘'].rolling(window=20).mean()
df_long['Upper_Band'] = df_long['Middle_Band'] + 2 * df_long['收盘'].rolling(window=20).std()
df_long['Lower_Band'] = df_long['Middle_Band'] - 2 * df_long['收盘'].rolling(window=20).std()

# 3. 定义买卖信号
df_long['Buy_Signal'] = np.where(
    (df_long['Short_MA'] > df_long['Long_MA']) &
    (df_long['收盘'] <= df_long['Lower_Band']),
    1, 0)
df_long['Sell_Signal'] = np.where(
    (df_long['Short_MA'] < df_long['Long_MA']) &
    (df_long['收盘'] >= df_long['Upper_Band']),
    -1, 0)

# 4. 回测策略
initial_balance = 100000
balance = initial_balance
stock_position = 0
transactions = []

for index, row in df_long.iterrows():
    # Buy signal
    if row['Buy_Signal'] == 1 and stock_position == 0:
        buy_price = row['收盘']
        stock_position = balance / buy_price
        balance = 0
        transactions.append({'日期': row['日期'], '类型': '买入', '价位': buy_price, '盈利': None})

    # Sell signal
    if row['Sell_Signal'] == -1 and stock_position > 0:
        sell_price = row['收盘']
        profit = (sell_price - buy_price) * stock_position
        balance += stock_position * sell_price
        stock_position = 0
        transactions.append({'日期': row['日期'], '类型': '卖出', '价位': sell_price, '盈利': profit})

# If still holding the stock at the end of data, sell it
if stock_position > 0:
    profit = (df_long['收盘'].iloc[-1] - buy_price) * stock_position
    transactions.append(
        {'日期': df_long['日期'].iloc[-1], '类型': '卖出', '价位': df_long['收盘'].iloc[-1], '盈利': profit})

# 5. 输出结果
df_transactions = pd.DataFrame(transactions)
total_profit = df_transactions['盈利'].sum()
print(df_transactions)
print(f"总盈利: {total_profit}")