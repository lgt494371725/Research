#include "Minconflicts.h"
#include<random>
#include<ctime>
#include<algorithm>
int Minconflicts::getMainDia(int row, int col) {
//找一个一条对角线的不变量，就是row - col = constant, 最小是1 - n, 所以加上n - 1，0 ~ 2n - 2共(2n - 1)个。
	return row - col + n - 1;
}

int Minconflicts::getAntiDia(int row, int col) {
	return row + col;
}

unsigned long long int Minconflicts::get_randindex(long long int n) {
	unsigned long long int x;
	double i;
	x = 0x7fffffff;
	x += 1;
	RandSeed *= ((unsigned long long int)134775813);
	RandSeed += 1;
	RandSeed = RandSeed % x;
	i = ((double)RandSeed) / (double)0x7fffffff;
	
	return (unsigned long long int)(n * i);
}

// 初始时冲突的皇后数量,这里的数字是指随机放置的皇后个数,不管是否产生冲突
int Minconflicts::get_c(int n) {
    if (n <= 10) return n > 8 ? 8 : n;
    else if (n <= 100) return 30;
    else if (n <= 10000) return 50;
    else if (n <= 100000) return 80;
    return 100;
}

int Minconflicts::initialize(int c){
    int m = n - c;
    for (int i = 0; i < n; ++i) queens[i] = i;//每列都是一个皇后，之后优化也是交换皇后的列，所以不会有列冲突存在 
    int conflicts = 0;
    // 首先在空闲列中随机放置m个皇后,保证无冲突
    int j;
	for (int i = 0, last = n; i < m; ++i, --last) {
        // 从[i, n)中选j,保证不影响已放置的皇后
//        std::default_random_engine e;
//		std::uniform_int_distribution<long long> u(i, n-1);
//        j = u(e);
		j = i + get_randindex(last);
        while (mdiag[getMainDia(i,queens[j])]>0 || adiag[getAntiDia(i, queens[j])]>0) j = i + get_randindex(last);
        std::swap(queens[i], queens[j]);
        mdiag[getMainDia(i,queens[i])]++;
        adiag[getAntiDia(i, queens[i])]++;
    }
    // 剩余c个皇后在空闲列中随便放置,不管是否产生冲突
    for (int i = m, last = c; i < n; ++i, --last) {
//        std::default_random_engine e;
//		std::uniform_int_distribution<long long> u(i, n-1);
//        j = u(e);
        j = i + get_randindex(last);
        std::swap(queens[i], queens[j]);
        mdiag[getMainDia(i,queens[i])]++;
        adiag[getAntiDia(i, queens[i])]++;
    }
    for (int i = 0; i < 2 * n - 1; ++i) {
        conflicts += mdiag[i] > 1 ? mdiag[i] - 1 : 0;
        conflicts += adiag[i] > 1 ? adiag[i] - 1 : 0;
    }
	std::cout << "初始皇后序列：" << std::endl;
	print();
    return conflicts;

}

// 交换第i和第j个皇后带来的冲突数变化
int Minconflicts::swap_gain(int i, int j, int n) {
    int gain = 0;
    // 原来位置对应的对角线上皇后数需要减1
    // 如果原来对应的对角线上有冲突,则gain--
    if (adiag[getAntiDia(i, queens[i])] > 1) gain--;
    if (adiag[getAntiDia(j, queens[j])] > 1) gain--;
    if (mdiag[getMainDia(i, queens[i])] > 1) gain--;
    if (mdiag[getMainDia(j, queens[j])] > 1) gain--;
    // 同理可知新对应的对角线上有皇后,则gain++
    if (adiag[getAntiDia(i, queens[j])] > 0) gain++;
    if (adiag[getAntiDia(j, queens[i])] > 0) gain++;
    if (mdiag[getMainDia(i, queens[j])] > 0) gain++;
    if (mdiag[getMainDia(j, queens[i])] > 0) gain++;
    return gain;
}

// 由于交换而更新冲突表和皇后位置
void Minconflicts::update_state(int i, int j, int n) {
    adiag[getAntiDia(i, queens[i])]--;
    adiag[getAntiDia(j, queens[j])]--;
    mdiag[getMainDia(i, queens[i])]--;
    mdiag[getMainDia(j, queens[j])]--;

    adiag[getAntiDia(i, queens[j])]++;
    adiag[getAntiDia(j, queens[i])]++;
    mdiag[getMainDia(i, queens[j])]++;
    mdiag[getMainDia(j, queens[i])]++;
    
    std::swap(queens[i], queens[j]);
}

void Minconflicts::compute() {//Main
	srand(unsigned(time(0)));
	bool restart = true;
	int curr;
    while (true) {
        if (restart) curr = initialize(get_c(n));
        if (curr <= 0) break;
        restart = true;
        int gain = 0;
        // 随机交换两个皇后,第一个从产生冲突的里面选取,第二个完全随机选取
        for (int i = 0; i < n; ++i) {
            if (mdiag[getMainDia(i,queens[i])] > 1 || adiag[getAntiDia(i, queens[i])] > 1) {
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        gain = swap_gain(i, j, n);
                        if (gain < 0) {
                            update_state(i, j, n);
                            curr += gain;
                            restart = false;
                            break;
                        }
                    }
                }
                if (restart) break;
            }
        }
    }
	std::cout << "最终皇后序列：" << std::endl;
	print();
}

void Minconflicts::print() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < queens[i]; j++)
			std::cout << " .";
		std::cout << " Q";
		for (int j = queens[i] + 1; j < n; j++)
			std::cout << " .";
		std::cout << std::endl;
	}
}

