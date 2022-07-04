#include "error.cuh"
#include <math.h>
#include <stdio.h>
// 定义USE_DP宏
#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    // 将float类型使用typeef定义为real
    typedef float real;
    // 声明EPSILON常量，并赋值为1.0e-6f
    const real EPSILON = 1.0e-6f;
#endif
// 声明循环次数
const int NUM_REPEATS = 10;
// 定义类型为a,b,c的指针
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;
// 初始化add和check函数
void add(const real *x, const real *y, real *z, const int N);
void check(const real *z, const int N);

int main(void)
{
    const int N = 100000000;
    // 通过typedef将float类型定义为real，因此使用malloc()分配空间时，可以使用real类型
    const int M = sizeof(real) * N;
    real *x = (real*) malloc(M);
    real *y = (real*) malloc(M);
    real *z = (real*) malloc(M);
    // 初始化x,y,z数组
    for (int n = 0; n < N; ++n)
    {
        x[n] = a;
        y[n] = b;
    }
    // 开始计时
    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        // 创建开始和结束的cudaEvent对象
        cudaEvent_t start, stop;
        // 使用cudaEventCreate创建时间，并使用CHECK函数检验该CUDA函数的正确性
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        // 使用cudaEventRecord记录事件
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);
        // 调用add函数
        add(x, y, z, N);
        CHECK(cudaEventRecord(stop));
        // 同步CUDA事件
        CHECK(cudaEventSynchronize(stop));
        // 创建float类型时间变量
        float elapsed_time;
        // 通过cudaEventElapsedTime获取时间，并将时间存储到elapsed_time中
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        // 打印单次循环的执行时间
        printf("Time = %.2f seconds.\n", elapsed_time / 1000);
        // 如果repeat不等于0，则计算平均时间
        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }
        // 销毁cudaEvent对象
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    // 计算平均时间和异常时间
    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %.2f +- %.2f seconds.\n", t_ave / 1000, t_err / 1000);

    check(z, N);

    free(x);
    free(y);
    free(z);
    return 0;
}

void add(const real *x, const real *y, real *z, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const real *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}
