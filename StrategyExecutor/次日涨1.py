# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/15 13:25
:last_date:
    2023/10/15 13:25
:description:
    
"""
import pandas as pd

# 1. 读取数据
stock_data = pd.read_csv("../InfoCollector/stock_data/daily/000682_东方电子.csv")

# Calculate the short-term (40 days) and long-term (100 days) moving averages
stock_data['40D_MA'] = stock_data['收盘'].rolling(window=40).mean()
stock_data['100D_MA'] = stock_data['收盘'].rolling(window=100).mean()

# Initialize trade signals
stock_data['Buy_Signal'] = (stock_data['40D_MA'].shift(1) < stock_data['100D_MA'].shift(1)) & (
            stock_data['40D_MA'] > stock_data['100D_MA'])
stock_data['Sell_Signal'] = (stock_data['收盘'] > stock_data['40D_MA']) & (stock_data['收盘'] > stock_data['收盘'].shift(1))

# Lists to store the backtest results in the desired format
dates = []
transaction_types = []
prices = []
growth_rates = []
profits = []
days_to_sell = []
cumulative_profits = []

capital = 100000
cumulative_profit = 0

in_trade = False
buy_price = None
buy_date = None

for index, row in stock_data.iterrows():
    if in_trade and row['Sell_Signal'] and row['收盘'] > buy_price:
        # Sell transaction
        dates.append(row['日期'])
        transaction_types.append('卖出')
        prices.append(row['收盘'])

        growth_rate = ((row['收盘'] - buy_price) / buy_price) * 100
        growth_rates.append(f"{growth_rate:.2f}%")

        profit = (row['收盘'] - buy_price) * (capital / buy_price)
        profits.append(profit)

        days_held = (pd.to_datetime(row['日期']) - pd.to_datetime(buy_date)).days
        days_to_sell.append(days_held)

        cumulative_profit += profit
        cumulative_profits.append(cumulative_profit)

        in_trade = False

    elif not in_trade and row['Buy_Signal']:
        # Buy transaction
        dates.append(row['日期'])
        transaction_types.append('买入')
        prices.append(row['收盘'])
        growth_rates.append(None)
        profits.append(None)
        days_to_sell.append(None)
        cumulative_profits.append(cumulative_profit)

        buy_price = row['收盘']
        buy_date = row['日期']
        in_trade = True

# Create the backtest results dataframe in the desired format
backtest_results_formatted = pd.DataFrame({
    '日期': dates,
    '类型': transaction_types,
    '价位': prices,
    '涨幅': growth_rates,
    '盈利': profits,
    '首次超买价天数': days_to_sell,
    '总盈利': cumulative_profits
})
print(backtest_results_formatted)