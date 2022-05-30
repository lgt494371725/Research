#include "Genetic.h"

void Genetic::initializePopulation() {
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < n; j++) {
			population.individuals[i].genes[j] = rand() % n;
		// 4����Ⱥ��ÿ����Ⱥ16������ 
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
	individual.fitness = 1.0 / conflicts;//��һƪ������˵���õ����������������ó�ͻ��Ч����ߺܶ�
}

void Genetic::selection() {//����ο���sicolex������
	double sum_fitness = 0.0;
	for (int i = 0; i < k; i++)
		sum_fitness += population.individuals[i].fitness;
	for (int i = 0; i < n; i++) {//�����̶�ѡ���ģ�˳���������Ⱥ�����൱��ÿ���������ӽ���
		int magnify_fitness = sum_fitness * 10000;
		int random_fitness = rand() % magnify_fitness;
		double select = (double)random_fitness / (double)10000;
		int random_postion = std::lower_bound(population.individuals[i].genes, population.individuals[i].genes + n, select) - population.individuals[i].genes;//������Ϊ��upper�������lower���Աȣ����ܻ��и��õİ취
		std::swap(population.individuals[i], population.individuals[random_postion]);//��һ������һ��int����
	}
}

void Genetic::crossover() {
	int cross_over_point;//����ֻ��Ϊ����Ϥ�㷨����ѡ�������ѡһ���㽻���ұߵĻ���Ƭ��
	for (int i = 0; i < k; i+=2) {
		cross_over_point = rand() % n;
		for (int j = cross_over_point; j < n; j++) {
			std::swap(population.individuals[i].genes[j], population.individuals[i + 1].genes[j]);
		}
	}
}

void Genetic::mutation() {
	for (int i = 0; i < k; i++) {
		if (rand() % 2 == 0) {//�����������50%����ͻ�䣬�ı��˾͸�����Ӧֵ
			population.individuals[i].genes[rand() % n] = rand() % n;//gene��1/n/2�ļ���ͻ��
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

