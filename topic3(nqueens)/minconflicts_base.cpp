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
std::cout << "initialize board��" << std::endl;
print();
}

bool Minconflicts::checkitout(int row) {// find a optimal column for every row
	int currentCol = queens[row];
	int optimalCol = queens[row];
	int minConflicts = col[optimalCol] + mdiag[getMainDia(row, optimalCol)] + adiag[getAntiDia(row, optimalCol)] - 3;
	int conflicts;
	//conflicts number, -3 ����Ϊ����Ҫ����˻ʺ�û�з������λ�� 
	/*std::cout << col[optimalCol] << " " << mdiag[getMainDia(row, optimalCol)] << " " << adiag[getAntiDia(row, optimalCol)] << std::endl;
	std::cout << getMainDia(1, 1) << " " << mdiag[3] << " " << col[0] << std::endl;*/
	for (int i = 0; i < n; i++) {//����row�е�ÿһ��i
		if (i == currentCol)
			continue;
		conflicts = col[i] + mdiag[getMainDia(row, i)] + adiag[getAntiDia(row, i)];
		//����Ҫ���ϼ����ƹ������Ǹ��ʺ�Ҳ����Ҫ+2��������-2��������+2����С
		/*std::cout << "main diagonal " << getMainDia(row, i) << "  anti diagonal " << getAntiDia(row, i) << std::endl;
		std::cout << "conflicts " << minConflicts << std::endl;*/
		if (conflicts < minConflicts) {
			optimalCol = i;
			minConflicts = conflicts;
		}
		else if (conflicts == minConflicts && rand() % 2) // rand()%2 == 0 or 1 
//Ϊ�������׽���ֲ����ŵ�������Ƚϼ򵥵İ취����ÿһ�еĻʺ�����ı䣬����ѡ�����������С��ͻֵ���50%�ƶ�
			optimalCol = i;
	}
	/*std::cout << "col " << optimalCol << std::endl;
	std::cout << "conflicts " << minConflicts << std::endl;
	std::system("pause");*/
	if (currentCol != optimalCol) {//�ҵ����Ҳ���ԭ����һ�У��Ͱѻʺ��ƶ���ȥ
		queens[row] = optimalCol;
		calculate(row, currentCol, optimalCol);
		if (col[currentCol] <= 1 && col[optimalCol] <= 1 && mdiag[getMainDia(row, optimalCol)] <= 1 && adiag[getAntiDia(row, optimalCol)] <= 1) {
			for (int i = 0; i < n; i++)//ÿһ�м�������ǲ��Ƿ���Ҫ����
				if (col[queens[i]] > 1 || mdiag[getMainDia(i, queens[i])] > 1 || adiag[getAntiDia(i, queens[i])] > 1)
					return false;
			return true;
		}
	}
	return false;//���û�ı䣬�ǿ϶����ԣ���Ȼ�����һ�е�ʱ����Ѿ�������
}

void Minconflicts::calculate(int row, int precol, int destcol) {//���ﰴ�м��㣬������ȷ�����ƶ��о��С�
	col[destcol]++;
	mdiag[getMainDia(row, destcol)]++;
	adiag[getAntiDia(row, destcol)]++;
	if (precol == -1)
		return;
	col[precol]--;
	mdiag[getMainDia(row, precol)]--;
	adiag[getAntiDia(row, precol)]--;
}
