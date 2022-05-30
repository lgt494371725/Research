#include "Genetic.h"

void Genetic::initializePopulation() {
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < n; j++) {
			population.individuals[i].genes[j] = rand() % n;
		// 4个种群，每个种群16个个体 
		}
		calculateFitness(population.individuals[i]);
	}
}

void Genetic::calculateFitness(Individual &individual) {
	int conflicts = 0;
	int *queens = individual.genes;
	int fitness;
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			if (queens[i] == queens[j] || abs(queens[i] - queens[j]) == j - i) {
				conflicts++;
			}
		}
	}
	if (conflicts == 0) {
		population.fittest = individual;
		flag = false;
		return;
	}
	individual.fitness = 1.0 / conflicts;//据一篇文章所说，用倒数而不是像书上用冲突数效果提高很多
}

void Genetic::selection() {//这里参考了sicolex的文章
	double sum_fitness = 0.0;
	for (int i = 0; i < k; i++)
		sum_fitness += population.individuals[i].fitness;
	for (int i = 0; i < n; i++) {//按轮盘赌选出的，顺序组成新种群，就相当于每次挑两个杂交了
		int magnify_fitness = sum_fitness * 10000;
		int random_fitness = rand() % magnify_fitness;
		double select = (double)random_fitness / (double)10000;
		int random_postion = std::lower_bound(population.individuals[i].genes, population.individuals[i].genes + n, select) - population.individuals[i].genes;//个人认为是upper，待会和lower做对比，可能还有更好的办法
		std::swap(population.individuals[i], population.individuals[random_postion]);//是一个基因，一个int数组
	}
}

void Genetic::crossover() {
	int cross_over_point;//这里只是为了熟悉算法，就选择了随机选一个点交换右边的基因片段
	for (int i = 0; i < k; i+=2) {
		cross_over_point = rand() % n;
		for (int j = cross_over_point; j < n; j++) {
			std::swap(population.individuals[i].genes[j], population.individuals[i + 1].genes[j]);
		}
	}
}

void Genetic::mutation() {
	for (int i = 0; i < k; i++) {
		if (rand() % 2 == 0) {//这个基因组有50%几率突变，改变了就更新适应值
			population.individuals[i].genes[rand() % n] = rand() % n;//gene有1/n/2的几率突变
			calculateFitness(population.individuals[i]);
		}
	}
}

void Genetic::compute() {
	initializePopulation();
	while (flag) {
		selection();
		crossover();
		mutation();
	}
	print(population.fittest.genes);
}

void Genetic::print(int *queens) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < queens[i]; j++)
			std::cout << " .";
		std::cout << " Q";
		for (int j = queens[i] + 1; j < n; j++)
			std::cout << " .";
		std::cout << std::endl;
	}
}

