#include <math.h>
#include <stdio.h>
// 定义EPSILON系数和a,b,c的基础系数
const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
// 定义add()函数模板为全局函数
void __global__ add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main(void)
{
    // 样本长度N为100000000
    const int N = 100000000;
    // 生成对应的整数内存空间
    const int M = sizeof(double) * N;
    // 动态分配h_x,h_y,h_z内存
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);
    // 循环赋值并初始化
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }
    // 重新定义GPU指针
    double *d_x, *d_y, *d_z;
    /* 使用cudaMalloc()函数生成内存空间
    接受两个参数：指向指针的内存空间，以及大小
     */
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);
    /* 使用cudaMemcpy()函数将h_x,h_y复制到d_x,d_y
    接收参数：GPU指针，CPU指针，内存大小，数据迁移模式
    */
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);
    // 定义块大小和网格大小
    const int block_size = 128;
    const int grid_size = N / block_size;
    // 启动内核，将GPU的x，y，z指针作为形参传入
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);
    // 使用cudaMemcpy()将GPU计算的Z值反传回h_z
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    // 调用check()函数检查是否完全相不相同
    check(h_z, N);
    // 释放内存
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

void __global__ add(const double *x, const double *y, double *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

void check(const double *z, const int N)
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