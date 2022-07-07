#include "error.cuh"
#include <math.h>
#include <stdio.h>
// 定义宏
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif
// 定义循环次数
const int NUM_REPEATS = 10;
// 定义x0为整数数据
const real x0 = 100.0;
// 初始化arthmetic函数，参数为x指针，x0和N
void arithmetic(real *x, const real x0, const int N);

int main(void)
{
    // 初始化数组长度
    const int N = 10000;
    // 初始化数组内存大小
    const int M = sizeof(real) * N;
    real *x = (real*) malloc(M);
    // 初始化时间变量
    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        // 为数组赋值，期间GPU将执行10 * 10000，也就是十万次该操作
        for (int n = 0; n < N; ++n)
        {
            x[n] = 0.0;
        }
        // 初始化cudaEvent_t 对象用于表示开始和结束的cudaEvent事件
        cudaEvent_t start, stop;
        // 创建cudaEvent_t 对象
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        // 记录开始值
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);
        // 由于该函数为主机函数因此直接调用，不需要分配额外的grid_size和block_size
        arithmetic(x, x0, N);
        // 记录结束事件
        CHECK(cudaEventRecord(stop));
        // 同步结束事件
        CHECK(cudaEventSynchronize(stop));
        // 计算时间
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);
        // 防止内核初始化造成的性能准确度降低，因此记录第二次往后的时间
        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }
        // 销毁CUDA事件
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    // 计算平均时间
    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
    // 释放内存
    free(x);
    return 0;
}

void arithmetic(real *x, const real x0, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        real x_tmp = x[n];
        while (sqrt(x_tmp) < x0)
        {
            ++x_tmp;
        }
        x[n] = x_tmp;
    }
}
