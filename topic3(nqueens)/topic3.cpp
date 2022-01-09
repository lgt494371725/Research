#pragma warning(disable:4996)
#include <iostream>
#include <string>
#include "Backtracking.h"
#include "Backtracking.cpp"
#include "Minconflicts.h"
#include "Minconflicts.cpp"
#include "Genetic.h"
#include <ctime>
using namespace std;

#define n 10

int main() {
	double duration;
	clock_t start, end;

//	cout << "Backtracking:" << endl;
//	Backtracking bt(n);
//	start = clock();
//	bt.backtracking(0);//recursive
//	end = clock();
//	duration = (double)(end - start);
//	cout << "���ݷ���� "<<bt.nSolutions() <<" �ִ𰸣���ʱ "<< duration / CLOCKS_PER_SEC << " ��" <<endl;
	cout << "Minconflicts:" << endl;
	Minconflicts mc(n);
	start = clock();
	mc.compute();
	end = clock();
	duration = (double)(end - start);
	cout << "��С��ͻ�ҵ�һ�ִ���ʱ " << duration / CLOCKS_PER_SEC << " ��" << endl;

//	cout << "Genetic:" << endl;
//	Genetic gg(n, 4*n);
//	start = clock();
//	gg.compute();
//	end = clock();
//	duration = (double)(end - start);
//	cout << "�Ŵ��㷨�ҵ�һ�ִ���ʱ " << duration / CLOCKS_PER_SEC << " ��" << endl;

	system("pause");
	return 0;
}

