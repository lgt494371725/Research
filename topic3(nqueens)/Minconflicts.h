#pragma once
#include <iostream>
#include <random>
#include <windows.h>
#include<string.h>
#include <algorithm>
#include<ctime>

//����Min-Conflicts����;�и���������֦���ݵĻ����ϣ�ÿ��ѡ���ѡλ�����ٵ�һ�������ԡ�������ʽ�ġ�
//�������ʼ��һ������ûʺ�����̣�Ȼ��ѡ��һ���ʺ󣬼�����̣��ƶ���һ��ʹ�ó�ͻ���ٵ�λ�ã��Դ����������ǿ���ά������⣬����ȫ״̬�ġ�

class Minconflicts {
private:
	int n;//N-Queens
	//queens num of diagonal, but conflicts equals queens-1
	// three vector. n queens board have 2*n-1 diagonal
	int *mdiag;//store the num of queen in main diagonal ������
	int *adiag;//anti-diagonal
	int *queens;//board
	unsigned long long int RandSeed;
public:
	Minconflicts(int num = 8) : n(num), mdiag(new int[2 * num]), adiag(new int[2 * num]), queens(new int[num]) {
		unsigned long long int RandSeed = (unsigned)time(NULL);
		memset(queens, -1,num*sizeof(int));
		memset(mdiag, 0,(2*num-1)*sizeof(int)); //0 ~ 2*num-2, 2n-1��
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

