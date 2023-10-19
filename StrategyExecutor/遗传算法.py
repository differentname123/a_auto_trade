# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023/10/15 14:26
:last_date:
    2023/10/15 14:26
:description:
    
"""
import pandas as pd
import numpy as np
import talib


import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv("../InfoCollector/stock_data/daily/000682_东方电子.csv")
data['日期'] = pd.to_datetime(data['日期'])

# 计算SMA和EMA
data['SMA'] = data['收盘'].rolling(window=40).mean()
data['EMA'] = data['收盘'].ewm(span=40, adjust=False).mean()

# 计算MACD
exp12 = data['收盘'].ewm(span=12, adjust=False).mean()
exp26 = data['收盘'].ewm(span=26, adjust=False).mean()
data['MACD'] = exp12 - exp26
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_Hist'] = data['MACD'] - data['Signal_Line']

# 计算RSI
delta = data['收盘'].diff(1)
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# 计算VWAP
data['Cumulative_Volume'] = data['成交量'].cumsum()
data['Cumulative_Volume_Price'] = (data['收盘'] * data['成交量']).cumsum()
data['VWAP'] = data['Cumulative_Volume_Price'] / data['Cumulative_Volume']

# 计算布林带
data['Bollinger_Middle'] = data['收盘'].rolling(window=20).mean()
data['Bollinger_Std'] = data['收盘'].rolling(window=20).std()
data['Bollinger_Upper'] = data['Bollinger_Middle'] + (data['Bollinger_Std'] * 2)
data['Bollinger_Lower'] = data['Bollinger_Middle'] - (data['Bollinger_Std'] * 2)

# 计算其他指标
data['Momentum'] = data['收盘'].diff(4)
low_min = data['最低'].rolling(window=14).min()
high_max = data['最高'].rolling(window=14).max()
data['Stochastic_Oscillator'] = 100 * ((data['收盘'] - low_min) / (high_max - low_min))
data['Williams_%R'] = -100 * ((high_max - data['收盘']) / (high_max - low_min))


# 适应度函数
def get_fitness_all_indicators_extended_force_sell(individual):
    w_sma, w_ema, w_macd, w_rsi, w_vwap, w_bband, w_momentum, w_stoch, w_williams = individual
    data['Weighted_Buy_Signal'] = (data['收盘'] - data['SMA']) * w_sma + (data['收盘'] - data['EMA']) * w_ema + \
                                  data['MACD_Hist'] * w_macd + (data['RSI'] - 30) * w_rsi + \
                                  (data['收盘'] - data['VWAP']) * w_vwap + (
                                              data['收盘'] - data['Bollinger_Lower']) * w_bband + \
                                  data['Momentum'] * w_momentum + data['Stochastic_Oscillator'] * w_stoch + \
                                  data['Williams_%R'] * w_williams
    data['Buy_Signal'] = data['Weighted_Buy_Signal'] > 0

    dates, transaction_types, prices, growth_rates, profits, days_to_sell, cumulative_profits = backtest_strategy_force_sell(
        data)

    num_trades = len(dates) // 2
    total_days_held = sum([0 if x is None else x for x in days_to_sell])
    average_days_held = total_days_held / num_trades if num_trades > 0 else 0

    total_profit = sum([x for x in profits if x is not None])

    # Adjusted weights for fitness function
    WEIGHT_PROFIT = 1.5
    WEIGHT_TRADE_FREQUENCY = 1.2
    WEIGHT_DAYS_HELD = -0.8

    fitness = WEIGHT_PROFIT * total_profit + WEIGHT_TRADE_FREQUENCY * num_trades * 1000 + WEIGHT_DAYS_HELD * average_days_held
    if num_trades < 1:
        fitness -= 100000

    for day in days_to_sell:
        if day is not None and day > 5:
            fitness -= 500 * (day - 5)

    return fitness


# 回测策略
def backtest_strategy_force_sell(data):
    initial_capital = 100000  # 初始资金为10w
    current_capital = initial_capital
    current_stocks = 0
    buy_date = None
    buy_price = None
    current_profit = 0

    dates = []
    transaction_types = []
    prices = []
    growth_rates = []
    profits = []
    days_to_sell = []
    cumulative_profits = []

    for i, row in data.iterrows():
        if buy_date and ((row['收盘'] > buy_price) or (i == data.index[-1])):  # Sell condition
            sell_date = row['日期']
            sell_price = row['收盘']
            growth_rate = (sell_price - buy_price) / buy_price * 100
            profit = (sell_price - buy_price) * current_stocks
            day_to_sell = (sell_date - buy_date).days

            dates.extend([buy_date, sell_date])
            transaction_types.extend(["买入", "卖出"])
            prices.extend([buy_price, sell_price])
            growth_rates.extend([None, growth_rate])
            profits.extend([None, profit])
            days_to_sell.extend([None, day_to_sell])

            current_profit += profit
            cumulative_profits.extend([None, current_profit])
            current_capital += initial_capital + current_profit # Update the capital after selling
            current_stocks = 0  # Reset stocks after selling

            buy_date = None
            buy_price = None
        elif not buy_date and row['Buy_Signal'] and row['收盘'] is not None:  # Buy condition
            buy_date = row['日期']
            buy_price = row['收盘']
            stocks_to_buy = int(current_capital // buy_price)  # Ensure we buy an integer number of stocks
            current_stocks += stocks_to_buy
            current_capital -= stocks_to_buy * buy_price

    return dates, transaction_types, prices, growth_rates, profits, days_to_sell, cumulative_profits



# 遗传算法参数
num_generations = 10000
population_size = 300
mutation_rate = 0.6
crossover_rate = 0.5
quanzhong = 10
rate_range = 0.5

# 选择、交叉和突变函数
def select_parents(population, fitnesses):
    idx = np.argsort(fitnesses)
    return population[idx[-2]], population[idx[-1]], fitnesses[idx[-2]], fitnesses[idx[-1]]


def crossover(parent1, parent2):
    crossover_idx = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:crossover_idx], parent2[crossover_idx:]])
    child2 = np.concatenate([parent2[:crossover_idx], parent1[crossover_idx:]])
    return child1, child2


def mutate(child, mutation_rate=0.05):
    new_child = child.copy()
    for i in range(len(new_child)):
        if np.random.rand() < mutation_rate:
            new_child[i] += np.random.uniform(-rate_range * quanzhong, rate_range * quanzhong)
    return new_child


# 主要的遗传算法循环
population = np.random.uniform(-quanzhong, quanzhong, size=(population_size, 9))
best_fitness = -np.inf
generation = 0
# 初始化best_individual，值为[  1.63196961   3.12597492  -5.26036775  -2.56364044 -15.3492927
#   -3.65101055  -7.6984696   -1.02220108   2.57710777]
best_individual = [1.63196961, 3.12597492, -5.26036775, -2.56364044, -15.3492927, -3.65101055, -7.6984696, -1.02220108, 2.57710777]
# 获取并打印最佳权重的回测数据
data['Weighted_Buy_Signal'] = (data['收盘'] - data['SMA']) * best_individual[0] + (data['收盘'] - data['EMA']) * \
                              best_individual[1] + \
                              data['MACD_Hist'] * best_individual[2] + (data['RSI'] - 30) * best_individual[3] + \
                              (data['收盘'] - data['VWAP']) * best_individual[4] + (
                                      data['收盘'] - data['Bollinger_Lower']) * best_individual[5] + \
                              data['Momentum'] * best_individual[6] + data['Stochastic_Oscillator'] * \
                              best_individual[7] + \
                              data['Williams_%R'] * best_individual[8]
data['Buy_Signal'] = data['Weighted_Buy_Signal'] > 0
dates, transaction_types, prices, growth_rates, profits, days_to_sell, cumulative_profits = backtest_strategy_force_sell(
    data)
backtest_results = pd.DataFrame({
    '日期': dates,
    '类型': transaction_types,
    '价位': prices,
    '涨幅': growth_rates,
    '盈利': profits,
    '首次超买价天数': days_to_sell,
    '总盈利': cumulative_profits
})

print(
    f"\nGeneration {generation + 1} - Best Backtest Results with fitness {best_fitness} best_individual:{best_individual}:\n")
print(backtest_results)
# 将上面的结果存入文件 out.txt
with open('out.txt', 'a') as f:
    f.write(
        f"\nGeneration {generation + 1} - Best Backtest Results with fitness {best_fitness} best_individual:{best_individual}:\n")
    f.write(str(backtest_results))
    f.write('\n')

for generation in range(num_generations):
    fitnesses = np.array([get_fitness_all_indicators_extended_force_sell(ind) for ind in population])
    parent1, parent2, fitness1, fitness2 = select_parents(population, fitnesses)

    best_fitness = fitness2
    best_individual = parent2

    # 获取并打印最佳权重的回测数据
    data['Weighted_Buy_Signal'] = (data['收盘'] - data['SMA']) * best_individual[0] + (data['收盘'] - data['EMA']) * \
                                  best_individual[1] + \
                                  data['MACD_Hist'] * best_individual[2] + (data['RSI'] - 30) * best_individual[3] + \
                                  (data['收盘'] - data['VWAP']) * best_individual[4] + (
                                              data['收盘'] - data['Bollinger_Lower']) * best_individual[5] + \
                                  data['Momentum'] * best_individual[6] + data['Stochastic_Oscillator'] * \
                                  best_individual[7] + \
                                  data['Williams_%R'] * best_individual[8]
    data['Buy_Signal'] = data['Weighted_Buy_Signal'] > 0
    dates, transaction_types, prices, growth_rates, profits, days_to_sell, cumulative_profits = backtest_strategy_force_sell(
        data)
    backtest_results = pd.DataFrame({
        '日期': dates,
        '类型': transaction_types,
        '价位': prices,
        '涨幅': growth_rates,
        '盈利': profits,
        '首次超买价天数': days_to_sell,
        '总盈利': cumulative_profits
    })

    print(f"\nGeneration {generation + 1} - Best Backtest Results with fitness {best_fitness} best_individual:{best_individual}:\n")
    print(backtest_results)
    # 将上面的结果存入文件 out.txt
    with open('out.txt', 'a') as f:
        f.write(f"\nGeneration {generation + 1} - Best Backtest Results with fitness {best_fitness} best_individual:{best_individual}:\n")
        f.write(str(backtest_results))
        f.write('\n')


    new_population = []
    for _ in range(int(population_size / 2)):
        if np.random.rand() < crossover_rate:
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate=mutation_rate))
            new_population.append(mutate(child2, mutation_rate=mutation_rate))
        else:
            new_population.append(parent1)
            new_population.append(parent2)
    population = np.array(new_population)

print("\nOverall Best Weights:", best_individual)
print("Overall Best Fitness:", best_fitness)
