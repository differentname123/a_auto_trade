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
import multiprocessing
import random
import time

from StrategyExecutor.common import load_data, backtest_strategy_low_profit
from StrategyExecutor.zuhe_daily_strategy import gen_full_all_basic_signal, back_layer_all_op_gen, \
    gen_full_all_basic_signal_gen, statistics_zuhe_gen, read_json

def generate_offspring(args):
    parent1, parent2, crossover_func, judge_gene_func, mutate_func = args
    child1, child2 = crossover_func(parent1, parent2)
    offspring = []
    mutate_child1 = mutate_func(child1)
    mutate_child2 = mutate_func(child2)
    if judge_gene_func(mutate_child1):
        offspring.append(mutate_child1)
    if judge_gene_func(mutate_child2):
        offspring.append(mutate_child2)
    return offspring

def _generate_gene(min_ones, max_ones, gene_length, judge_gene):
    ones_count = random.randint(min_ones, max_ones)
    gene = ['1'] * ones_count + ['0'] * (gene_length - ones_count)
    random.shuffle(gene)
    gene_str = ''.join(gene)
    if judge_gene(gene_str):
        return gene_str
    return None

class GeneticAlgorithm:
    def __init__(self, population_size, crossover_rate, mutation_rate, signal_columns):
        self.gene_length = len(signal_columns)
        self.min_ones = 3
        self.max_ones = 4
        self.relation_map = {}
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.signal_columns = signal_columns
        self.combinations = []
        self.population = self._initialize_population_mul()

    def load_all_statistics(self):
        """
        读取所有组合的统计信息
        :return:
        """
        all_statistics = read_json('../back/gen/statistics_all.json')
        self.existed_combinations = set(all_statistics.keys())
        # 找出所有低于1000次交易的组合
        self.low_trade_count_combinations = set()
        for combination in self.existed_combinations:
            if all_statistics[combination]['trade_count'] <= 1000:
                self.low_trade_count_combinations.add(combination)

    def judge_gene(self, gene):
        """
        判断组合是否合法
        :param combination:
        :return:
        """
        # 如果gene中1的个数小于3，则不合法
        ones_count = gene.count('1')
        if ones_count < self.min_ones:
            return False
        combination_list = self.cover_to_combination(gene)
        combination = ':'.join(combination_list)
        if combination in self.existed_combinations:
            return False
        if any([set(combination_list) >= set(low_trade_count_combination.split(':')) for low_trade_count_combination in self.low_trade_count_combinations]):
            return False

        return True

    def _initialize_population(self):
        # 开始计时
        start_time = time.time()
        # 初始化种群，确保每个基因中1的数量在指定范围内
        population = set()
        self.load_all_statistics()
        while len(population) < self.population_size:
            ones_count = random.randint(self.min_ones, self.max_ones)
            gene = ['1'] * ones_count + ['0'] * (self.gene_length - ones_count)
            random.shuffle(gene)
            if self.judge_gene(gene):
                population.add(''.join(gene))
        # 结束计时
        end_time = time.time()
        print('sig初始化种群耗时：', end_time - start_time)
        return list(population)

    def _initialize_population_mul(self):
        # 开始计时
        start_time = time.time()
        # 初始化种群，确保每个基因中1的数量在指定范围内
        population = set()
        self.load_all_statistics()

        while len(population) < self.population_size:
            # 使用多进程
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            results = pool.starmap(_generate_gene,
                                   [(self.min_ones, self.max_ones, self.gene_length, self.judge_gene) for _ in
                                    range(self.population_size * 2)])  # 增加迭代次数以确保足够的种群大小
            pool.close()
            pool.join()
            for gene_str in results:
                if gene_str:
                    population.add(gene_str)
            print('初始化种群大小：', len(population))
        # 结束计时
        end_time = time.time()
        print('mul初始化种群耗时：', end_time - start_time)
        return list(population)

    # def _fitness(self, individual):
    #     # 适应度函数：计算该组合的得分
    #     if individual["ratio"] > 0.18:
    #         return 0
    #     if 'average_1w_profit' not in individual:
    #         individual["average_1w_profit"] = 0
    #
    #     # 参数调整
    #     ratio_impact = 100  # 增大 ratio 的影响
    #     trade_count_impact = 5  # trade_count 的影响系数
    #     trade_count_min_threshold = 10  # trade_count 的最小阈值
    #
    #     # 计算 ratio 的得分，使用指数衰减函数增强 ratio 接近阈值时的影响
    #     ratio_score = -math.exp(-ratio_impact * (individual["ratio"] - 0.18) ** 2)
    #
    #     # 计算 trade_count 的得分，使用指数衰减函数
    #     trade_count_score = math.exp(
    #         -trade_count_impact * (1 / max(individual["trade_count"], trade_count_min_threshold)))
    #
    #     # 其他因素
    #     profit_scale = 1000
    #     days_held_scale = 1
    #     profit_score = individual["average_1w_profit"] / profit_scale
    #     days_held_score = 1 / (1 + individual["average_days_held"])
    #
    #     total_fitness = ratio_score + trade_count_score
    #     return total_fitness

    def _fitness(self, individual):
        """
        崇尚交易数量，交易数量越多越好
        :param individual:
        :return:
        """
        total_trade_count = 1000
        trade_count_threshold = 100
        # 适应度函数：计算该组合的得分
        if 'average_1w_profit' not in individual:
            individual["average_1w_profit"] = 0
        if individual["trade_count"] == 0:
            return -10000
        trade_count_score = individual["trade_count"] / total_trade_count
        if individual["trade_count"] < trade_count_threshold:
            trade_count_score = individual["trade_count"] - trade_count_threshold * 10 / individual["trade_count"]
        total_fitness = trade_count_score
        return total_fitness

    def _select(self):
        # 选择：轮盘赌选择法
        return random.choices(self.population, weights=self.selection_probs, k=2)

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
        if len(individual) != self.gene_length:
            print('error')
        for i in range(self.gene_length):
            if random.random() < self.mutation_rate:
                individual[i] = '1' if individual[i] == '0' else '0'
        return ''.join(individual)

    def get_fitness(self):
        """
        获取当前种群的所有适应度
        :return:
        """
        # # 读取'../back/gen/sublist.json'
        # self.combinations = read_json('../back/gen/sublist_back.json')
        # # 将combinations转换为population
        # self.population = [''.join(['1' if self.signal_columns[i] in combination else '0' for i in
        #                             range(len(self.signal_columns))]) for combination in self.combinations]

        self.combinations = [self.cover_to_combination(individual) for individual in self.population]
        back_layer_all_op_gen('../daily_data_exclude_new_can_buy', self.combinations, gen_signal_func=gen_full_all_basic_signal,
                              backtest_func=backtest_strategy_low_profit)
        self.gen_combination_to_fitness()
        self.load_all_statistics()

    def gen_combination_to_fitness(self):
        """
        生成种群和相应的适应度关系
        :return:
        """
        statistics = read_json('../back/gen/statistics_target_key.json')
        self.relation_map = {}
        for individual in self.population:
            key = ':'.join(self.cover_to_combination(individual))
            if key in statistics:
                self.relation_map[key] = [self._fitness(statistics[key]), statistics[key]]
            else:
                # 是因为子组合就是0所以就没有统计到
                self.relation_map[key] = [-1, 0]

        # statistics = read_json('../back/statistics_target_key.json')
        # for key in statistics.keys():
        #     self.relation_map[key] = [self._fitness(statistics[key]), statistics[key]]
        # # 将适应度排序
        # self.relation_map = sorted(self.relation_map.items(), key=lambda x: x[1][0], reverse=True)
        self.total_fitness = sum(value[0] for value in self.relation_map.values())
        self.selection_probs = [value[0] / self.total_fitness for value in self.relation_map.values()]
        return self.relation_map

    def re_gen_population(self):
        """
        随机减少每个基因中1的个数直到小于等于4
        :return:
        """
        for i in range(len(self.population)):
            individual = self.population[i]
            while individual.count('1') > 4:
                individual = list(individual)
                index = random.randint(0, len(individual) - 1)
                if individual[index] == '1':
                    individual[index] = '0'
            self.population[i] = ''.join(individual)

    def run_generation(self):
        # 开始计时
        start_time = time.time()
        # 运行一代
        new_population = set()
        self.get_fitness()
        while len(new_population) < self.population_size:
            parent1, parent2 = self._select()
            child1, child2 = self._crossover(parent1, parent2)
            if self.judge_gene(child1):
                new_population.add(self._mutate(child1))
            if len(new_population) < self.population_size:
                if self.judge_gene(child2):
                    new_population.add(self._mutate(child2))
        self.population = list(new_population)
        # 去除self.population中全为0的个体
        self.population = [individual for individual in self.population if '1' in individual]
        # 结束计时
        end_time = time.time()
        print('sig生成新种群耗时：', end_time - start_time)

    def run_generation_mul(self):
        # 开始计时
        start_time = time.time()
        # 运行一代
        new_population = set()
        self.get_fitness()

        while len(new_population) < self.population_size:
            # 准备传递给 generate_offspring 的参数
            args_list = [(self._select()[0], self._select()[1], self._crossover, self.judge_gene, self._mutate) for
                         _ in range(self.population_size * 2)]

            # 使用多进程
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            results = pool.map(generate_offspring, args_list)
            pool.close()
            pool.join()

            for offspring in results:
                new_population.update(offspring)
            print('new_population:', len(new_population))

        self.population = list(new_population)
        self.population = [individual for individual in self.population if '1' in individual]

        # 结束计时
        end_time = time.time()
        print('mul生成新种群耗时：', end_time - start_time)

    def get_best_individual(self):
        # 获取最佳个体
        temp_map = sorted(self.relation_map.items(), key=lambda x: x[1][0], reverse=True)
        return temp_map[0]

def filter_good_zuhe():
    """
    过滤出好的指标，并且全部再跑一次
    :return:
    """
    statistics = read_json('../back/gen/statistics_all.json')
    # 所有的指标都应该满足10次以上的交易
    statistics_new = {k: v for k, v in statistics.items() if v['trade_count'] > 10} # 100交易次数以上 13859
    good_ratio_keys = {k: v for k, v in statistics_new.items() if v['ratio'] <= 0.125}
    result_combinations = good_ratio_keys.keys()
    print(result_combinations)

if __name__ == '__main__':
    # filter_good_zuhe()
    # statistics_zuhe_gen('../back/gen/zuhe', target_key='all')
    data = load_data('../daily_data_exclude_new_can_buy/龙洲股份_002682.txt')
    data = gen_full_all_basic_signal(data)
    signal_columns = [column for column in data.columns if 'signal' in column]
    # 示例参数
    population_size = 1000  # 种群大小
    crossover_rate = 0.7  # 交叉率
    mutation_rate = 0.001  # 变异率

    # 创建遗传算法实例并运行
    ga = GeneticAlgorithm(population_size, crossover_rate, mutation_rate, signal_columns)
    for _ in range(5000):  # 运行50代
        ga.run_generation_mul()
        best_info = ga.get_best_individual()
        try:
            if best_info[1][1]['trade_count'] < 100:
                ga._initialize_population_mul()
        except:
            pass
        # 增量写入文件
        with open('../back/gen/best.txt', 'a') as f:
            f.write(str(best_info) + '\n')


    # # # 输出最佳个体及其适应度
    # # best_individual = ga.get_best_individual()
    # # best_fitness = ga._fitness(best_individual)
    # # print("Best Individual:", best_individual, "Fitness:", best_fitness)
    # # statistics_zuhe_gen('../back/gen/zuhe', target_key="target_key")