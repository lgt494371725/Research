#pragma once
#include <iostream>
#include <random>
#include <windows.h>
#include<string.h>
#include <algorithm>
#include<ctime>

//关于Min-Conflicts，在途中给搜索树剪枝回溯的基础上，每次选择可选位置最少的一个来尝试。是增量式的。
//先随机初始化一个分配好皇后的棋盘，然后选择一个皇后，检查棋盘，移动到一个使得冲突最少的位置，以此往复，这是看完维基的理解，是完全状态的。

class Minconflicts {
private:
	int n;//N-Queens
	//queens num of diagonal, but conflicts equals queens-1
	// three vector. n queens board have 2*n-1 diagonal
	int *mdiag;//store the num of queen in main diagonal 　　　
	int *adiag;//anti-diagonal
	int *queens;//board
	unsigned long long int RandSeed;
public:
	Minconflicts(int num = 8) : n(num), mdiag(new int[2 * num]), adiag(new int[2 * num]), queens(new int[num]) {
		unsigned long long int RandSeed = (unsigned)time(NULL);
		memset(queens, -1,num*sizeof(int));
		memset(mdiag, 0,(2*num-1)*sizeof(int)); //0 ~ 2*num-2, 2n-1个
		memset(adiag, 0,(2*num-1)*sizeof(int));
	}
	//which diagonal the queen is on
	int getMainDia(int row, int col);
	int getAntiDia(int row, int col);
	//key functions
	int initialize(int c);
	void compute();//main
	//print 1 answer
	void print();
	int get_c(int n);
	int swap_gain(int i, int j, int n);
	void update_state(int i, int j, int n);
	unsigned long long get_randindex(long long int n);
};

