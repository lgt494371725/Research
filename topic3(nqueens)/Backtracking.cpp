#include "Backtracking.h"

bool Backtracking::isOk(int row) {
	for (int i = 0; i < row; i++)//Top-down
		//在对角线就是说明两个皇后之间行距等于列距 
		if (queens[row] - queens[i] == row - i || queens[row] - queens[i] == i - row || queens[row] == queens[i])
		//两个对角线以及列，行不用确认，因为我们是一行放一个皇后 
			return false;
	return true;
}

void Backtracking::backtracking(int row) {
	if (row >= n) {
		total++;
		print();//if you wanna print all solutions
		exit(0);//if you wanna find just 1 solution
	}
	else
		for (int col = 0; col < n; col++) {
			queens[row] = col;
			if (isOk(row))
				backtracking(row + 1);
		}
}

void Backtracking::print() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < queens[i]; j++)
			std::cout << " .";
		std::cout << " Q";
		for (int j = queens[i] + 1; j < n; j++)
			std::cout << " .";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int Backtracking::nSolutions() {
	return total;
}

