# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023-12-04 5:11
:last_date:
    2023-12-04 5:11
:description:
    区别于之前的全排列组合，这里使用遗传算法来进行参数的优化
"""
import math
import random

from StrategyExecutor.common import load_data, backtest_strategy_low_profit
from StrategyExecutor.zuhe_daily_strategy import gen_full_all_basic_signal, back_layer_all_op_gen, \
    gen_full_all_basic_signal_gen, statistics_zuhe_gen, read_json


class GeneticAlgorithm:
    def __init__(self, population_size, crossover_rate, mutation_rate, signal_columns):
        self.gene_length = len(signal_columns)
        self.min_ones = 1
        self.max_ones = 4
        self.relation_map = {}
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = self._initialize_population()
        self.signal_columns = signal_columns
        self.combinations = []

    def _initialize_population(self):
        # 初始化种群，确保每个基因中1的数量在指定范围内
        population = set()
        for _ in range(self.population_size):
            ones_count = random.randint(self.min_ones, self.max_ones)
            gene = ['1'] * ones_count + ['0'] * (self.gene_length - ones_count)
            random.shuffle(gene)
            population.add(''.join(gene))
        return list(population)

    def _fitness(self, individual):
        # 适应度函数：计算该组合的得分
        if individual["ratio"] > 0.18:
            return 0
        if 'average_1w_profit' not in individual:
            individual["average_1w_profit"] = 0

        # 参数调整
        ratio_impact = 100  # 增大 ratio 的影响
        trade_count_impact = 5  # trade_count 的影响系数
        trade_count_min_threshold = 10  # trade_count 的最小阈值

        # 计算 ratio 的得分，使用指数衰减函数增强 ratio 接近阈值时的影响
        ratio_score = -math.exp(-ratio_impact * (individual["ratio"] - 0.18) ** 2)

        # 计算 trade_count 的得分，使用指数衰减函数
        trade_count_score = math.exp(
            -trade_count_impact * (1 / max(individual["trade_count"], trade_count_min_threshold)))

        # 其他因素
        profit_scale = 1000
        days_held_scale = 1
        profit_score = individual["average_1w_profit"] / profit_scale
        days_held_score = 1 / (1 + individual["average_days_held"])

        total_fitness = ratio_score + trade_count_score
        return total_fitness

    def _select(self):
        # 选择：轮盘赌选择法
        total_fitness = sum(value[0] for value in self.relation_map.values())
        selection_probs = [value[0] / total_fitness for value in self.relation_map.values()]
        return random.choices(self.population, weights=selection_probs, k=2)

    def cover_to_combination(self, individual):
        # 将个体转换为组合
        return [self.signal_columns[i] for i, gene in enumerate(individual) if gene == '1']

    def _crossover(self, parent1, parent2):
        # 交叉
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.gene_length - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1, parent2

    def _mutate(self, individual):
        # 变异
        individual = list(individual)
        for i in range(self.gene_length):
            if random.random() < self.mutation_rate:
                individual[i] = '1' if individual[i] == '0' else '0'
        return ''.join(individual)

    def get_fitness(self):
        """
        获取当前种群的所有适应度
        :return:
        """
        self.combinations = [self.cover_to_combination(individual) for individual in self.population]
        # 读取'../back/gen/sublist.json'
        # self.combinations = read_json('../back/gen/sublist.json')
        back_layer_all_op_gen('../daily_data_exclude_new_can_buy', self.combinations, gen_signal_func=gen_full_all_basic_signal,
                              backtest_func=backtest_strategy_low_profit)
        self.gen_combination_to_fitness()

    def gen_combination_to_fitness(self):
        """
        生成种群和相应的适应度关系
        :return:
        """
        statistics = read_json('../back/gen/statistics_target_key.json')
        # statistics = read_json('../back/statistics_target_key.json')
        self.relation_map = {}
        for individual in self.population:
            key = ':'.join(self.cover_to_combination(individual))
            if key in statistics:
                self.relation_map[key] = [self._fitness(statistics[key]), statistics[key]]
            else:
                # 是因为子组合就是0所以就没有统计到
                self.relation_map[key] = [0, 0]
        # for key in statistics.keys():
        #     self.relation_map[key] = [self._fitness(statistics[key]), statistics[key]]
        # # 将适应度排序
        # self.relation_map = sorted(self.relation_map.items(), key=lambda x: x[1][0], reverse=True)

        return self.relation_map

    def run_generation(self):
        # 运行一代
        new_population = set()
        self.get_fitness()
        while len(new_population) < self.population_size:
            parent1, parent2 = self._select()
            child1, child2 = self._crossover(parent1, parent2)
            new_population.add(self._mutate(child1))
            if len(new_population) < self.population_size:
                new_population.add(self._mutate(child2))
        self.population = list(new_population)
        # 去除self.population中全为0的个体
        self.population = [individual for individual in self.population if '1' in individual]

    def get_best_individual(self):
        # 获取最佳个体
        temp_map = sorted(self.relation_map.items(), key=lambda x: x[1][0], reverse=True)
        return temp_map[0]


if __name__ == '__main__':
    data = load_data('../daily_data_exclude_new_can_buy/龙洲股份_002682.txt')
    data = gen_full_all_basic_signal(data)
    signal_columns = [column for column in data.columns if 'signal' in column]
    # 示例参数
    population_size = 1000  # 种群大小
    crossover_rate = 0.7  # 交叉率
    mutation_rate = 0.01  # 变异率

    # 创建遗传算法实例并运行
    ga = GeneticAlgorithm(population_size, crossover_rate, mutation_rate, signal_columns)
    for _ in range(50):  # 运行50代
        ga.run_generation()
        best_info = ga.get_best_individual()
        # 增量写入文件
        with open('../back/gen/best.txt', 'a') as f:
            f.write(str(best_info) + '\n')


    # # 输出最佳个体及其适应度
    # best_individual = ga.get_best_individual()
    # best_fitness = ga._fitness(best_individual)
    # print("Best Individual:", best_individual, "Fitness:", best_fitness)
    # statistics_zuhe_gen('../back/gen/zuhe', target_key="target_key")