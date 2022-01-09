#pragma once//include����
#include <iostream>

//ԭ���˻ʺ�������C_64^8��64ȡ8��C(64, 8) = 4426165368��44���ֿ��ܵ����������������ȡ��
//ÿ�ηŻʺ��ʱ�򶼼��ÿһ��ÿһ��ÿһб�����Ƿ���ڳ�ͻ�ʺ��Դ����������ϼ�֦���ݡ���ô����ͷǳ���д��
class Backtracking {
private:
	const int n;//board
	int total;//solutions
	int *queens;
public:
	Backtracking(int num = 8) : n(num), queens(new int[num]), total(0) {//queens 0~7
		for (int i = 0; i < n; i++)
			queens[i] = -1;
	};
	bool isOk(int row);
	void backtracking(int row);
	//print all answer
	void print();
	int nSolutions();
};

