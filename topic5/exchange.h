#ifndef EXCHANGE_H
#define EXCHANGE_H
#include<iostream>
#include<math.h>
#include<vector>
#include<memory>
#include<time.h>
#include<queue>
#include<time.h>
#define block_length 3
using std::cout;
using std::vector;
using std::unique_ptr;
typedef vector<vector<int>> Board;
typedef vector<vector<int>> *ptr_to_b;
typedef vector<int> position;
int hamming(Board board);
int manhattan(Board board);
void print_board(Board board);
int single_manhattan(int x,int y,int value);
Board exchange(ptr_to_b p,int i,int j,int ii,int jj);


Board exchange(ptr_to_b p,int i,int j,int ii,int jj)
{
    Board new_board=*p;
    int temp=new_board[i][j];
    new_board[i][j]=new_board[ii][jj];
    new_board[ii][jj]=temp;
    return new_board;
}

void print_board(Board b)
{
    for (int i=0;i<block_length;i++){
        for (int j=0;j<block_length;j++){
            cout<<b[i][j]<<" ";
        }
        cout<<"\n";
    }
    cout<<"\n";
}

#endif
