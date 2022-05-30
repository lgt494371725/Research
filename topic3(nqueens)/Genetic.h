#pragma once
#include <iostream>
#include <random>
#include <math.h>
#include <algorithm>
#include <functional>

//д�����ʱ��ͻȻ�뵽һ�����⣬�������������ö��ʻ������ʣ�
//����һ��о����ؾ��ᣬ�ö��ʾ������Լ��ڲٿأ������ʴ��������Թ۱仯��ģ��仯�����ɡ�
class Genetic {//�������м����ʺ󣬻��򳤶Ⱦ��Ǽ��������㹻��
private:
	int n;//N-Queens
	int k;//K-Groups
	//int *queens;//����״̬�պÿ��Ե�������Ļ�����һ��nλ0~n-1�Ĵ���
	struct Individual {
		int *genes; 
		double fitness;
	};
	struct Population {//��Ⱥ 
		Individual *individuals;
		Individual fittest;
	}population;
	bool flag;//�ɹ��ҵ����Ž����𣿳ɹ���false
public:
	Genetic(int num_queens = 4, int num_groups = 16) : n(num_queens), k(num_groups), flag(true) {//��ʼ��
		population.individuals = new Individual[k];//��ֵ
		for (int i = 0; i < k; i++)//��ʼ���͸�ֵ���ʲ�һ���������ˡ�
			population.individuals[i].genes = new int[n];
	}
	//key functions
	void initializePopulation();
	//��ʼ����Ⱥ
	void calculateFitness(Individual &individual);//������Ӧֵ
	void selection();//ѡ��ѡ����Ⱥ����õĸ�������ӽ�
	void crossover();//�ӽ���Crossover is the most significant phase in a genetic algorithm.
	void mutation();//���죬����ͻ����Ϊ��ά����Ⱥ�ڲ��Ķ�����
	void compute();//main
	//print 1 answer
	void print(int *queens);
};

