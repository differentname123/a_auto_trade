# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/14 19:35
:last_date:
    2023/10/14 19:35
:description:
    
"""
import pandas as pd


def compute_RSI(data, window=14):
    delta = data['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data


def compute_stochastic_oscillator(data, k_window=14, d_window=3):
    low_min = data['收盘'].rolling(window=k_window).min()
    high_max = data['收盘'].rolling(window=k_window).max()
    data['%K'] = 100 * (data['收盘'] - low_min) / (high_max - low_min)
    data['%D'] = data['%K'].rolling(window=d_window).mean()
    return data


def compute_moving_averages(data, short_window=40, long_window=100):
    data['Short_MA'] = data['收盘'].rolling(window=short_window).mean()
    data['Long_MA'] = data['收盘'].rolling(window=long_window).mean()
    return data


def compute_MACD(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['收盘'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['收盘'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = data['MACD'].rolling(window=signal_window).mean()
    return data


def backtest_strict_no_future_data_strategy(df, initial_balance=100000):
    df = compute_RSI(df)
    df = compute_stochastic_oscillator(df)
    df = compute_moving_averages(df)
    df = compute_MACD(df)

    df['MA20'] = df['收盘'].rolling(window=20).mean()
    df['20dSTD'] = df['收盘'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['20dSTD'] * 2)
    df['Lower'] = df['MA20'] - (df['20dSTD'] * 2)
    df['Momentum'] = df['收盘'] - df['收盘'].shift(4)

    balance = initial_balance
    stock_position = 0
    buy_price = 0
    transactions = []
    total_profit = 0

    for index in range(1, len(df) - 1):
        current_price = df.iloc[index]['收盘']

        is_rsi_signal = df.iloc[index - 1]['RSI'] < 30 and df.iloc[index]['RSI'] > 30
        is_moving_avg_signal = df.iloc[index]['Short_MA'] > df.iloc[index]['Long_MA'] and df.iloc[index - 1][
            'Short_MA'] <= df.iloc[index - 1]['Long_MA']
        is_stochastic_signal = df.iloc[index]['%K'] > df.iloc[index]['%D'] and df.iloc[index]['%K'] < 30
        is_macd_signal = df.iloc[index]['MACD'] > df.iloc[index]['Signal_Line'] and df.iloc[index - 1]['MACD'] <= \
                         df.iloc[index - 1]['Signal_Line']
        is_bollinger_signal = df.iloc[index]['收盘'] < df.iloc[index]['Lower']
        is_momentum_signal = df.iloc[index]['Momentum'] > 0

        buy_signals_count = sum(
            [is_rsi_signal, is_moving_avg_signal, is_stochastic_signal, is_macd_signal, is_bollinger_signal,
             is_momentum_signal])
        if buy_signals_count >= 3 and stock_position == 0:
            buy_price = current_price
            stock_position = balance / buy_price
            balance = 0
            transactions.append(
                {'日期': df.iloc[index]['日期'], '类型': '买入', '价位': buy_price, '涨幅': None, '盈利': None, '首次超买价天数': None,
                 '总盈利': total_profit})

            for j in range(index + 1, len(df)):
                next_price = df.iloc[j]['收盘']
                if next_price > buy_price and stock_position > 0:
                    sell_price = next_price
                    profit = (sell_price - buy_price) * stock_position
                    total_profit += profit
                    balance += stock_position * sell_price
                    stock_position = 0
                    price_increase_percentage = ((sell_price - buy_price) / buy_price) * 100
                    transactions.append({'日期': df.iloc[j]['日期'], '类型': '卖出', '价位': sell_price,
                                         '涨幅': f"{price_increase_percentage:.2f}%", '盈利': profit, '首次超买价天数': j - index,
                                         '总盈利': total_profit})
                    break

    if stock_position > 0:
        final_sell_price = df.iloc[-1]['收盘']
        profit = (final_sell_price - buy_price) * stock_position
        total_profit += profit
        price_increase_percentage = ((final_sell_price - buy_price) / buy_price) * 100
        transactions.append(
            {'日期': df.iloc[-1]['日期'], '类型': '卖出', '价位': final_sell_price, '涨幅': f"{price_increase_percentage:.2f}%",
             '盈利': profit, '首次超买价天数': len(df) - df[df['日期'] == transactions[-1]['日期']].index.item() - 1,
             '总盈利': total_profit})

    return transactions

df = pd.read_csv("../InfoCollector/stock_data/daily/000682_东方电子.csv")
# Run the backtesting with the provided strategy
full_transactions = backtest_strict_no_future_data_strategy(df)

# Convert the transactions to a DataFrame for better visualization
full_transactions_df = pd.DataFrame(full_transactions)
print(full_transactions_df)
