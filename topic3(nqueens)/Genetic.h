#pragma once
#include <iostream>
#include <random>
#include <math.h>
#include <algorithm>
#include <functional>

//写这里的时候突然想到一个问题，给函数的命名用动词还是名词？
//想了一会感觉不必纠结，用动词就想象自己在操控，用名词代表我在旁观变化，模拟变化。均可。
class Genetic {//棋盘上有几个皇后，基因长度就是几，这样足够。
private:
	int n;//N-Queens
	int k;//K-Groups
	//int *queens;//棋盘状态刚好可以当作个体的基因，是一个n位0~n-1的串。
	struct Individual {
		int *genes; 
		double fitness;
	};
	struct Population {//种群 
		Individual *individuals;
		Individual fittest;
	}population;
	bool flag;//成功找到最优解了吗？成功则false
public:
	Genetic(int num_queens = 4, int num_groups = 16) : n(num_queens), k(num_groups), flag(true) {//初始化
		population.individuals = new Individual[k];//赋值
		for (int i = 0; i < k; i++)//初始化和赋值性质不一样，别忘了。
			population.individuals[i].genes = new int[n];
	}
	//key functions
	void initializePopulation();
	//初始化种群
	void calculateFitness(Individual &individual);//计算适应值
	void selection();//选择，选择种群中最好的个体出来杂交
	void crossover();//杂交，Crossover is the most significant phase in a genetic algorithm.
	void mutation();//变异，发生突变是为了维持种群内部的多样性
	void compute();//main
	//print 1 answer
	void print(int *queens);
};

