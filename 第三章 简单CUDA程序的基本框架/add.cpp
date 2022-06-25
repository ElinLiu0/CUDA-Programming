#include <math.h>
#include <stdlib.h>
#include <stdio.h>
// 定义EPSILON系数和a,b,c的基础系数

/*
    C++程序的一个中的核心思想：无论程序的逻辑多么复杂，
    都应尽可能的将变量以及指针在函数外进行声明，
    这样无论是对指针操作还是对具体值操作都将十分容易。
*/


const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
// 声明基础函数和参数等
void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    // 样本长度N为100000000
    const int N = 100000000;
    // 生成对应的整数内存空间
    const int M = sizeof(double) * N;
    // 动态分配x,y,z内存
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);
    // 循环赋值并初始化
    for (int n = 0; n < N; ++n)
    {
        x[n] = a;
        y[n] = b;
    }
    // 调用add函数，并将参数：x,y,z,N传入
    add(x, y, z, N);
    // 调用纠错函数check()
    check(z, N);
    // 释放x,y,z的内存
    free(x);
    free(y);
    free(z);
    return 0;
}
// 声明add函数的内容：double 类型指针 x,y,z以及整数迭代量N
void add(const double *x, const double *y, double *z, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}
// 声明函数check()
void check(const double *z, const int N)
{
    // 声明布尔类型变量：has_error
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        // 使用fabs函数计算z的绝对值与c系数的差值，当其大于EPSILON时，则说明存在异常
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}