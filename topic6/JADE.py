import math
import time
import numpy as np
import copy
from scipy import stats


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
        random_num = np.array([JADE.get_random_num() for i in range(n)])
        self.change_all_genes(random_num)

    def calc_fitness(self):
        """
        这里是原始的函数y值，不能直接使用
        """
        self.fitness = JADE.test_func(self.genes)


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
        """
        n_individual = len(self.individuals)
        if self.min_fitness < 0.0:  # 需要将所有适应值变为正数
            print("有负数适应值")
            for i in range(n_individual):
                self.individuals[i].fitness += -self.min_fitness
            self.fittest += -self.min_fitness
            self.fitness_sum += (-self.min_fitness) * n_individual
            self.min_fitness = 0.0
        if is_minimize:  # 最小值问题，适应值取倒数，原值越小适应值越大
            old_gen = copy.deepcopy(self.individuals)
            self.clear()
            for i in range(n_individual):
                new_value = round(1 / old_gen[i].fitness, 5)
                old_gen[i].fitness = new_value
                self.add_individual(old_gen[i])


class JADE:
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
        self.n_individual = params.get('n_individual')
        self.params = params
        self.population = Population()
        self.temp_popu = Population()
        self.mu_cr = self.mu_f = 0.5
        self.crossover_rate = self.scaling_factor = None
        self.sf = self.scr = np.array([])  # f and cr makes the individual successful evolution
        # self.delta_f = np.array([])  # delta_f = u_k-x_k = evolution value
        JADE.value_range = params.get('value_range')

    @staticmethod
    def test_func(x):
        # y = x*math.sin(10*math.pi*x)+2
        y = pow(x[0], 2) - 10 * math.cos(2 * math.pi * x[0]) + 10 + \
            pow(x[1], 2) - 10 * math.cos(2 * math.pi * x[1]) + 10
        return y

    @classmethod
    def get_random_num(cls):
        """
        :return: uniform distribution [Min, MAX)
        """
        MIN, MAX = cls.value_range
        return np.random.uniform(MIN, MAX)

    def generate_random_num(self, dist):
        if dist == "norm":
            random_num = np.random.normal(self.mu_cr, 0.1)
            random_num = np.clip(random_num, 0, 1)
        elif dist == "cauchy":
            random_num = stats.cauchy.rvs(self.mu_f, 0.1)
            while random_num < 0 or random_num > 1:
                random_num = stats.cauchy.rvs(self.mu_f, 0.1)
        else:
            print("unknown distribution!")
        return random_num

    def generate_random_nums(self, dist):
        random_nums = []
        if dist == "norm":
            random_nums = [np.random.normal(self.mu_cr, 0.1) for i in range(self.n_individual)]
            random_nums = np.clip(random_nums, 0, 1)

        elif dist == "cauchy":
            for i in range(self.n_individual):
                r = stats.cauchy.rvs(self.mu_f, 0.1)
                while r < 0 or r > 1:
                    r = stats.cauchy.rvs(self.mu_f, 0.1)
                random_nums.append(r)
        else:
            print("unknown distribution!")
        return np.array(random_nums)

    def initial(self, n_individual: int):
        self.population.clear()
        self.mu_cr = self.mu_f = 0.5
        # self.crossover_rate = np.array([self.generate_random_num(dist='norm')
        #                        for i in range(self.n_individual)])
        # self.scaling_factor = np.array([self.generate_random_num(dist='cauchy')
        #                                 for i in range(self.n_individual)])
        self.crossover_rate = self.generate_random_nums(dist='norm')
        self.scaling_factor = self.generate_random_nums(dist='cauchy')
        self.sf = np.array([])
        self.scr = np.array([])
        # self.delta_f = np.array([])
        for i in range(n_individual):
            self.population.add_individual(Individual(self.n))
        self.population.calc_all_fitness(self.params.get("minimize"))

    def selection(self):
        """
        贪欲选择，选择适应性更高的
        不使用包含delta_f的mean_WA->倒数适应性函数变化较大，难以衡量好坏之间的差距，需要优化适应性函数后再使用
        """
        c = 0.1  # learning rate
        old_gen = copy.deepcopy(self.population.individuals)
        new_gen = copy.deepcopy(self.temp_popu.individuals)
        self.population.clear()
        self.temp_popu.clear()
        n_individual = self.n_individual
        for num in range(n_individual):
            old = old_gen[num]
            new = new_gen[num]
            if old.fitness >= new.fitness:
                better = old
            else:  # 选择新的，更新池子和参数
                better = new   # 更新CR,F
                self.scr = np.append(self.scr, self.crossover_rate[num])
                self.sf = np.append(self.sf, self.scaling_factor[num])
            self.population.add_individual(better)
        if len(self.sf) > 0:
            mean_L = np.sum(self.sf ** 2) / np.sum(self.sf)
            mean_WA = np.mean(self.scr)
            # w_k = self.delta_f/np.sum(self.delta_f)
            # assert np.sum(w_k) == 1
            # print("delta_f", self.delta_f)
            # print("self.scr:",self.scr)
            # print("w_k:",w_k)
            # mean_WA = np.sum(self.scr*w_k)
            self.mu_f = (1 - c) * self.mu_f + c * mean_L
            self.mu_cr = (1 - c) * self.mu_cr + c * mean_WA
            # self.delta_f = np.append(self.delta_f, round(100*(new.fitness-old.fitness), 3))
            self.sf = np.array([])
            self.scr = np.array([])
        self.crossover_rate = self.generate_random_nums(dist='norm')
        self.scaling_factor = self.generate_random_nums(dist='cauchy')

    def mutation(self):
        """"
        对种群populations进行“变异”操作
        current_p_best/1: v_i = x_i + F_i*(x_pbest-x_i) + F_i*(x_r1-xr2)
        通常 pbest = random_choose from [0,n*p),p属于(0,1)
        这里单纯使用best，即最优个体
        选择xr2时，可以用到外部储备池(除非进化失败个体)与原种群的交集，这里只从原种群中选择
        """
        n_individual = self.n_individual
        old_gen = self.population.individuals
        old_best = self.population.fittest_ind
        for num in range(n_individual):
            ind = copy.deepcopy(old_gen[num])
            r1, r2 = np.random.randint(0, n_individual, 2)  # [0,n),没用储备池
            F = self.scaling_factor[num]
            new_genes = ind.genes + F * (old_best.genes - ind.genes) + \
                        F * (old_gen[r2].genes - old_gen[r1].genes)
            ind.change_all_genes(new_genes)
            self.temp_popu.add_individual(ind)
        self.temp_popu.calc_all_fitness()

    def crossover(self):
        """
        交叉，也需要重新计算适应性
        dimension_wise: 小于交叉率则继承v_ij，否则继承x_ij->u_ij->u_i
        """
        n_individual = self.n_individual
        old_gen = self.population.individuals
        new_gen = copy.deepcopy(self.temp_popu.individuals)
        self.temp_popu.clear()
        for num in range(n_individual):
            old = old_gen[num].genes
            new = new_gen[num].genes
            CR = self.crossover_rate[num]
            randoms = np.array([np.random.random() for i in range(self.params.get('dimension'))])
            new_genes = np.where(randoms < CR, new, old)
            new_gen[num].change_all_genes(new_genes)
            self.temp_popu.add_individual(new_gen[num])
        self.temp_popu.calc_all_fitness(self.params.get('minimize'))

    def run(self):
        n_individual = self.n_individual
        self.initial(n_individual)
        test_times = self.params.get("test_times")
        rounds = self.params.get("rounds")
        success = fail = 0
        # pre_fitness = 0
        for i in range(test_times):
            self.initial(n_individual)
            print(f"第{i}次测试")
            for cur_round in range(rounds):
                if cur_round % 10 == 0:
                    # if pre_fitness == self.population.fittest:
                    #     print(f"{cur_round}轮: 10轮内无变化，推断为已经收缩")
                    #     break
                    # pre_fitness = self.population.fittest
                    func_value = JADE.test_func(self.population.fittest_ind.genes)
                    if abs(func_value - self.params.get("opt_value")) <= 0.1:
                        break
                if cur_round % 100 == 0:
                    print(np.round(self.population.fittest_ind.genes, 2),
                          "最优适应度:",
                          round(self.population.fittest, 2))
                self.mutation()
                self.crossover()
                self.selection()
            func_value = JADE.test_func(self.population.fittest_ind.genes)
            if abs(func_value - self.params.get("opt_value")) <= 0.1:
                success += 1
            else:
                print("不够匹配的函数值:", func_value)
                fail += 1
        print("正确次数：", success)
        print("错误次数：", fail)
        return success


def main():
    params = {
        "value_range": (-10, 10), "dimension": 2, "n_individual": 200,  # 200->300,计算时间涨了4倍
        "rounds": 200, "test_times": 500, "pool_length": 20,
        "minimize": True, "opt_value": 0}
    start = time.perf_counter()
    sol = JADE(**params)
    success = sol.run()
    duration = time.perf_counter() - start
    print(f"{params.get('dimension')} dimension answer time per round:{duration/params['test_times']}s, "
          f"accuracy: {success/params['test_times']}")


main()
