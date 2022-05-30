import math
import time
import numpy as np
import copy


class Individual:
    def __init__(self, n: int):
        self.n = n  # 维度
        self.fitness = 0.0
        self.genes = np.array([0.0 for i in range(n)])  # 注意类型！！一定要设定为浮点型
        self.random_initial(n)

    def change_genes(self, x, y):  # 第x个基因赋值为y
        self.genes = y
        self.calc_fitness()  # 重新计算适应性

    def change_all_genes(self, y):
        self.genes = y
        self.calc_fitness()

    def random_initial(self, n):
        random_num = np.array([Genetic.get_random_num() for i in range(n)])
        self.change_all_genes(random_num)

    def calc_fitness(self):
        """
        这里是原始的函数y值，不能直接使用
        """
        self.fitness = Genetic.test_func(self.genes)


class Population:
    def __init__(self):
        self.individuals = []
        self.fittest_ind = None
        self.fittest = 0.0
        self.fitness_sum = 0.0
        self.min_fitness = float('inf')

    def add_individual(self, ind: Individual):
        self.min_fitness = min(self.min_fitness, ind.fitness)
        self.fitness_sum += ind.fitness
        self.individuals.append(ind)
        if not self.fittest_ind or self.fittest < ind.fitness:
            self.fittest_ind = ind
            self.fittest = ind.fitness

    def change_ind(self, ind_num, gene_num, value):
        """
        指定某个个体，再指定基因，赋值为value
        """
        ind = self.individuals[ind_num]
        ind.change_genes(gene_num, value)
        if self.fittest < ind.fitness:
            self.fittest_ind = ind
            self.fittest = ind.fitness

    def clear(self):
        self.__init__()

    def calc_all_fitness(self, is_minimize=False):
        """
        根据基因的原始适应值重新计算新的适应值
        用了线性变换后效果变差了
        """
        n_individual = len(self.individuals)
        if self.min_fitness < 0.0:  # 需要将所有适应值变为正数
            print("有负数适应值")
            for i in range(n_individual):
                self.individuals[i].fitness += -self.min_fitness
            self.fittest += -self.min_fitness
            self.fitness_sum += (-self.min_fitness)*n_individual
            self.min_fitness = 0.0
        if is_minimize:  # 最小值问题，适应值取倒数，原值越小适应值越大
            old_gen = copy.deepcopy(self.individuals)
            self.clear()
            for i in range(n_individual):
                new_value = round(1/old_gen[i].fitness, 5)
                old_gen[i].fitness = new_value
                self.add_individual(old_gen[i])

        # C = 1.7
        # avg_fitness = self.fitness_sum/n_individual
        # if self.min_fitness > ((C*avg_fitness-self.fittest)/(C-1)):
        #     a = (C-1)*avg_fitness/(self.fittest-avg_fitness)
        #     b = (self.fittest-C*avg_fitness)*avg_fitness/(self.fittest-avg_fitness)
        # else:
        #     a = avg_fitness/(avg_fitness-self.min_fitness)
        #     b = self.min_fitness*avg_fitness/(avg_fitness-self.min_fitness)
        # for i in range(n_individual):
        #     old_value = self.individuals[i].fitness
        #     self.individuals[i].fitness = a*old_value+b
        # self.fittest = a*self.fittest+b
        # self.fitness_sum = a*self.fitness_sum+b*n_individual
        # self.min_fitness = a*self.min_fitness+b


class Genetic:
    """
    种群：规定个体数量
    个体：染色体：list，存储各维度的取值
    适应度函数：计算每个个体的适应性。
    选择，交叉，变异
    超参数的设定十分重要，突变的步伐迈得过大，将永远无法收敛的最优值
    """
    value_range = None

    def __init__(self, **params):
        self.n = params.get('dimension')  # 基因维度
        self.params = params
        self.population = Population()
        Genetic.value_range = params.get('value_range')

    @staticmethod
    def test_func(x):
        # y = x*math.sin(10*math.pi*x)+2
        y = pow(x[0], 2) - 10*math.cos(2*math.pi*x[0])+10+\
            pow(x[1], 2)-10*math.cos(2*math.pi*x[1])+10
        return y

    @classmethod
    def get_random_num(cls):
        """
        :return: uniform distribution [Min, MAX)
        """
        MIN, MAX = cls.value_range
        return np.random.uniform(MIN, MAX)

    def initial(self, n_individual: int):
        self.population.clear()
        for i in range(n_individual):
            self.population.add_individual(Individual(self.n))
        self.population.calc_all_fitness(self.params.get("minimize"))

    def selection(self):
        """
        保留本代最优个体到下一代(不发生变异以及交叉)
        """
        fitness_sum = self.population.fitness_sum
        n_individual = len(self.population.individuals)
        old_gen = copy.deepcopy(self.population.individuals)
        old_best = copy.deepcopy(self.population.fittest_ind)
        self.population.clear()
        # for i in range(4):
        #     print(old_gen[i].fitness)
        # print("sum:",fitness_sum)
        choice_list = [old_gen[i].fitness / fitness_sum for i in range(n_individual)]
        choice_list = np.cumsum(choice_list)  # 构造轮盘
        # print(choice_list)
        # print("选择之后的:")
        # for i in range(4):
        #     print(old_gen[i].fitness)
        self.population.add_individual(old_best)  # 先将最优个体保留
        for i in range(n_individual-1):
            select = np.random.random()  # [0,1)
            idx = 0
            while choice_list[idx] < select:
                idx += 1
            self.population.add_individual(old_gen[idx])

    def mutation(self):
        """"
        对种群populations进行“变异”操作，同样为根据适应度函数来选择
        MAX,MIN 是当前种群个体取值的上限与下限
        x_a_2 = x_a_1 + k*(MAX-x_a_1)*r   when random.random()<0.5
        x_a_2 = x_a_1 - k*(x_a_1-MIN)*r   when random.random()>0.5
        k:constant number between (0,1]
        """
        MIN, MAX = self.params.get("value_range")
        k = 0.8
        old_gen = copy.deepcopy(self.population.individuals)
        old_best = copy.deepcopy(self.population.fittest_ind)
        assert old_best.fitness == self.population.individuals[0].fitness
        self.population.clear()
        prob = self.params.get("mutation_prob")
        self.population.add_individual(old_best)  # 突变后种群0号仍然是原最优个体
        for ind in old_gen[1:]:  # 0号是最优个体，直接不管
            # print("加入前原来:", ind.fitness)
            if np.random.random() < prob:
                r = np.random.random()
                if np.random.random() < 0.5:
                    new_genes = ind.genes + k*(MAX-ind.genes)*r
                else:
                    new_genes = ind.genes - k*(ind.genes-MIN)*r
                ind.change_all_genes(new_genes)
            else:  # 不做改变，当适应性需要重新计算，因为可能在交叉环节发生了改变
                ind.change_all_genes(ind.genes)
            # print("加入后:", ind.fitness)
            # print("before:",self.population.fitness_sum)
            # length = len(self.population.individuals)
            # for i in range(length):
            #     print("before:",self.population.individuals[i].fitness,end=' ')
            # print('\n')
            self.population.add_individual(copy.deepcopy(ind))
            # print("after:",self.population.fitness_sum)
            # length = len(self.population.individuals)
            # for i in range(length):
            #     print("after:", self.population.individuals[i].fitness, end=' ')
            # print('\n')
        self.population.calc_all_fitness(self.params.get('minimize'))

    def crossover(self):
        """
        交叉步骤不需要重新计算适应性，因为下一步异变时要进行重新计算
        这里只需要思考如何交叉
        交叉概率:cross_prob->不根据适应性来选择
        alpha = cross_factor, alpha取得小，说明交叉后主要值还是来自于前代父系
        交叉操作：x_a_2 = alpha*x_b_1 + (1-alpha)*x_a_1
                x_b_2 = alpha*x_a_1 + (1-alpha)*x_b_1
        """
        n_individual = len(self.population.individuals)
        alpha = self.params.get('cross_factor')
        prob = self.params.get('cross_prob')
        node_index = np.arange(self.params.get('n_individual'))
        np.random.shuffle(node_index)
        for i in range(0, n_individual, 2):
            if np.random.random() <= prob:
                if node_index[i] == 0 or node_index[i+1] == 0:
                    continue
                a = self.population.individuals[node_index[i]].genes
                b = self.population.individuals[node_index[i+1]].genes
                self.population.individuals[node_index[i]].genes = alpha*b + (1-alpha)*a
                self.population.individuals[node_index[i+1]].genes = alpha*a + (1-alpha)*b

    def run(self):
        n_individual = self.params.get('n_individual')
        self.initial(n_individual)
        test_times = self.params.get("test_times")
        rounds = self.params.get("rounds")
        success = fail = 0
        pre_fitness = 0
        for i in range(test_times):
            self.initial(n_individual)
            # print(f"第{i}次测试")
            for cur_round in range(rounds):
                if cur_round % 10 == 0:
                    # if pre_fitness == self.population.fittest:
                    #     # print("10轮内无变化，推断为已经收缩")
                    #     break
                    # pre_fitness = self.population.fittest
                    func_value = Genetic.test_func(self.population.fittest_ind.genes)
                    if abs(func_value - self.params.get("opt_value")) <= 0.1:
                        break
                # if cur_round % 100 == 0:
                #     print(np.round(self.population.fittest_ind.genes, 2),
                #           "最优适应度:",
                #           round(self.population.fittest, 2))
                self.selection()
                self.crossover()
                self.mutation()
            func_value = Genetic.test_func(self.population.fittest_ind.genes)
            if abs(func_value-self.params.get("opt_value")) <= 0.1:
                success += 1
            else:
                # print("不够匹配的函数值:", func_value)
                fail += 1
        print("正确次数：", success)
        print("错误次数：", fail)
        return success



def main():
    params = {"cross_prob": 0.8, "mutation_prob": 0.4, "cross_factor": 0.01,
              "value_range": (-10, 10), "dimension": 2, "n_individual": 200,  # 200->300,计算时间涨了4倍
              "rounds": 200, "test_times": 500,
              "minimize": True, "opt_value": 0}
    start = time.perf_counter()
    sol = Genetic(**params)
    success = sol.run()
    duration = time.perf_counter() - start
    print(f"{params.get('dimension')} dimension answer time per round:{duration/params['test_times']}s, "
          f"accuracy: {success/params['test_times']}")


main()
