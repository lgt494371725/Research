#pragma once//include防范
#include <iostream>

//原本八皇后问题是C_64^8，64取8是C(64, 8) = 4426165368，44亿种可能的情况，纯暴力不可取。
//每次放皇后的时候都检查每一行每一列每一斜角线是否存在冲突皇后，以此在搜索树上剪枝回溯。那么代码就非常好写。
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

