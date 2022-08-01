#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include "error.cuh"
#include <iostream>
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

int N;
const int NUM_REPEATS = 20;
const int MN = 10;
const real cutoff = 1.9;
const real cutoff_square = cutoff * cutoff;

void find_neighbor(int *NN,int *NL,const real *x,const real *y);
void timing(int *NN,int *NL,std::vector<real>,std::vector<real>);

void find_neighbor(int *NN,int *NL,const real *x,const real *y)
{
    for(int n = 0;n < N;n++)
    {
        // 初始化每个原子的邻居原子
        NN[n] = 0;
    }
    // 遍历N1原子
    for(int n1 = 0;n1 < N;++n1)
    {
        // 读取原子的坐标
        real x1 = x[n1];
        real y1 = y[n1];
        // 遍历N2原子
        for(int n2 = n1 + 1;n2 < N;++n2)
        {
            // 找出临近坐标
            real x12 = x[n2] - x1;
            real y12 = y[n2] - y1;
            // 计算N1和N2原子的坐标距离平方
            real distance_square = x12 * x12 + y12 * y12;
            // 如果小于截断距离平方，则为邻居原子
            if(distance_square < cutoff_square)
            {
                // 则临近原子的坐标为n2,n1
                NL[n1 * MN + NN[n1]++] = n2;
                NL[n2 * MN + NN[n2]++] = n1;
            }
        }
    }
}