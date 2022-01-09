bool Minconflicts::initialize(int row){
int minconf;
int opt_col;
int conf;
for (int row = 0; row < n; row++) {//every row has only one queen
    minconf = n;
    opt_col = 0;
    for (int cur_col=0; cur_col<n ; cur_col++){
        conf = col[cur_col] + mdiag[getMainDia(row, cur_col)] + adiag[getAntiDia(row, cur_col)];
        if (conf < minconf){
            opt_col = cur_col;
            minconf = conf;
        }
    }
    queens[row] = opt_col;
    calculate(row, -1, queens[row]);
}
std::cout << "initialize board：" << std::endl;
print();
}

bool Minconflicts::checkitout(int row) {// find a optimal column for every row
	int currentCol = queens[row];
	int optimalCol = queens[row];
	int minConflicts = col[optimalCol] + mdiag[getMainDia(row, optimalCol)] + adiag[getAntiDia(row, optimalCol)] - 3;
	int conflicts;
	//conflicts number, -3 是因为我们要假设八皇后还没有放在这个位置 
	/*std::cout << col[optimalCol] << " " << mdiag[getMainDia(row, optimalCol)] << " " << adiag[getAntiDia(row, optimalCol)] << std::endl;
	std::cout << getMainDia(1, 1) << " " << mdiag[3] << " " << col[0] << std::endl;*/
	for (int i = 0; i < n; i++) {//检查第row行的每一列i
		if (i == currentCol)
			continue;
		conflicts = col[i] + mdiag[getMainDia(row, i)] + adiag[getAntiDia(row, i)];
		//这里要加上假设移过来的那个皇后，也就是要+2，在上面-2比在下面+2开销小
		/*std::cout << "main diagonal " << getMainDia(row, i) << "  anti diagonal " << getAntiDia(row, i) << std::endl;
		std::cout << "conflicts " << minConflicts << std::endl;*/
		if (conflicts < minConflicts) {
			optimalCol = i;
			minConflicts = conflicts;
		}
		else if (conflicts == minConflicts && rand() % 2) // rand()%2 == 0 or 1 
//为避免轻易进入局部最优的情况，比较简单的办法是让每一行的皇后随机改变，这里选择的是两列最小冲突值相等50%移动
			optimalCol = i;
	}
	/*std::cout << "col " << optimalCol << std::endl;
	std::cout << "conflicts " << minConflicts << std::endl;
	std::system("pause");*/
	if (currentCol != optimalCol) {//找到了且不是原来那一列，就把皇后移动过去
		queens[row] = optimalCol;
		calculate(row, currentCol, optimalCol);
		if (col[currentCol] <= 1 && col[optimalCol] <= 1 && mdiag[getMainDia(row, optimalCol)] <= 1 && adiag[getAntiDia(row, optimalCol)] <= 1) {
			for (int i = 0; i < n; i++)//每一行检查现在是不是符合要求了
				if (col[queens[i]] > 1 || mdiag[getMainDia(i, queens[i])] > 1 || adiag[getAntiDia(i, queens[i])] > 1)
					return false;
			return true;
		}
	}
	return false;//如果没改变，那肯定不对，不然检查上一行的时候就已经可以了
}

void Minconflicts::calculate(int row, int precol, int destcol) {//这里按行计算，所以行确定，移动列就行。
	col[destcol]++;
	mdiag[getMainDia(row, destcol)]++;
	adiag[getAntiDia(row, destcol)]++;
	if (precol == -1)
		return;
	col[precol]--;
	mdiag[getMainDia(row, precol)]--;
	adiag[getAntiDia(row, precol)]--;
}
