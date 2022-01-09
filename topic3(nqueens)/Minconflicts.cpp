#include "Minconflicts.h"
#include<random>
#include<ctime>
#include<algorithm>
int Minconflicts::getMainDia(int row, int col) {
//��һ��һ���Խ��ߵĲ�����������row - col = constant, ��С��1 - n, ���Լ���n - 1��0 ~ 2n - 2��(2n - 1)����
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

// ��ʼʱ��ͻ�Ļʺ�����,�����������ָ������õĻʺ����,�����Ƿ������ͻ
int Minconflicts::get_c(int n) {
    if (n <= 10) return n > 8 ? 8 : n;
    else if (n <= 100) return 30;
    else if (n <= 10000) return 50;
    else if (n <= 100000) return 80;
    return 100;
}

int Minconflicts::initialize(int c){
    int m = n - c;
    for (int i = 0; i < n; ++i) queens[i] = i;//ÿ�ж���һ���ʺ�֮���Ż�Ҳ�ǽ����ʺ���У����Բ������г�ͻ���� 
    int conflicts = 0;
    // �����ڿ��������������m���ʺ�,��֤�޳�ͻ
    int j;
	for (int i = 0, last = n; i < m; ++i, --last) {
        // ��[i, n)��ѡj,��֤��Ӱ���ѷ��õĻʺ�
//        std::default_random_engine e;
//		std::uniform_int_distribution<long long> u(i, n-1);
//        j = u(e);
		j = i + get_randindex(last);
        while (mdiag[getMainDia(i,queens[j])]>0 || adiag[getAntiDia(i, queens[j])]>0) j = i + get_randindex(last);
        std::swap(queens[i], queens[j]);
        mdiag[getMainDia(i,queens[i])]++;
        adiag[getAntiDia(i, queens[i])]++;
    }
    // ʣ��c���ʺ��ڿ�������������,�����Ƿ������ͻ
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
	std::cout << "��ʼ�ʺ����У�" << std::endl;
	print();
    return conflicts;

}

// ������i�͵�j���ʺ�����ĳ�ͻ���仯
int Minconflicts::swap_gain(int i, int j, int n) {
    int gain = 0;
    // ԭ��λ�ö�Ӧ�ĶԽ����ϻʺ�����Ҫ��1
    // ���ԭ����Ӧ�ĶԽ������г�ͻ,��gain--
    if (adiag[getAntiDia(i, queens[i])] > 1) gain--;
    if (adiag[getAntiDia(j, queens[j])] > 1) gain--;
    if (mdiag[getMainDia(i, queens[i])] > 1) gain--;
    if (mdiag[getMainDia(j, queens[j])] > 1) gain--;
    // ͬ���֪�¶�Ӧ�ĶԽ������лʺ�,��gain++
    if (adiag[getAntiDia(i, queens[j])] > 0) gain++;
    if (adiag[getAntiDia(j, queens[i])] > 0) gain++;
    if (mdiag[getMainDia(i, queens[j])] > 0) gain++;
    if (mdiag[getMainDia(j, queens[i])] > 0) gain++;
    return gain;
}

// ���ڽ��������³�ͻ��ͻʺ�λ��
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
        // ������������ʺ�,��һ���Ӳ�����ͻ������ѡȡ,�ڶ�����ȫ���ѡȡ
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
	std::cout << "���ջʺ����У�" << std::endl;
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

