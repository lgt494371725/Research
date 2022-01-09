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
//	cout << "回溯法算得 "<<bt.nSolutions() <<" 种答案，用时 "<< duration / CLOCKS_PER_SEC << " 秒" <<endl;
	cout << "Minconflicts:" << endl;
	Minconflicts mc(n);
	start = clock();
	mc.compute();
	end = clock();
	duration = (double)(end - start);
	cout << "最小冲突找到一种答案用时 " << duration / CLOCKS_PER_SEC << " 秒" << endl;

//	cout << "Genetic:" << endl;
//	Genetic gg(n, 4*n);
//	start = clock();
//	gg.compute();
//	end = clock();
//	duration = (double)(end - start);
//	cout << "遗传算法找到一种答案用时 " << duration / CLOCKS_PER_SEC << " 秒" << endl;

	system("pause");
	return 0;
}

