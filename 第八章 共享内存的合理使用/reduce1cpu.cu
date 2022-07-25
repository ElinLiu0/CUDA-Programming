#include <stdio.h>
#include <stdlib.h>
#include "error.cuh"
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif
#define NUM_REPEATS 20
// 创建reduce归约函数
real reduce (const real *x, const int n)
{
    // 创建一个局部变量，用于存储归约结果，类型为float
    real sum = 0.0;
    for (int i = 0; i < n; i++)
    // 循环归约   
        sum += x[i];
    return sum;
}

void timing(const real *x,const int N)
{
    real sum = 0;
    for(int repeat = 0;repeat < NUM_REPEATS;++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);
        
        sum = reduce(x, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        real elapsedTime;
        CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        printf("Time = %g ms.\n",elapsedTime);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("Sum = %f\n", sum);
}

int main(void)
{
    // 创建数组
    const int n = 1000000;
    real *x = (real *)malloc(n * sizeof(real));
    for (int i = 0; i < n; i++)
        x[i] = 1.23;

    timing(x,n);

    free(x);
    return 0;
}

