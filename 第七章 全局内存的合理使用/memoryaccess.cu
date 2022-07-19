#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
// 顺序的合并访问：该情况下，线程ID将会遍历内存中的所有元素
void __global__ add(float *x,float *y,float *z)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    z[n] = x[n] + y[n];
}

// 乱序合并访问：该情况下线程ID将会发生某种置换操作
void __global__ add_permuted(float *x,float *y,float *z)
{
    // 创建置换线程ID
    int tid_permuted = threadIdx.x ^ 0x1;
    /* 由于threadIdx.x发生了置换，但blockIdx.x和blockDim.x仍然是顺序的，
    因此这两个值仍然是从0开始，所以在这种情况下的元素读取顺序仍然是0-31*/
    int n = tid_permuted + blockIdx.x * blockDim.x;
}

// 不对齐的非合并访问：该情况下线程与内存之间会产生偏移
void __global__ add_offset(float *x,float *y,float *z)
{
    // 创建偏移线程ID
    int n = threadIdx.x + blockIdx.x * blockDim.x + 1;
    // 此时第一个block将获取1-32个元素。
    z[n] = x[n] + y[n];
}

/*跨越式的非合并访问：该情况下第一个线程块中的
线程束将访问数组下标题为0、128、256、384、512的元素，
*/
void __global__ add_stride(float *x,float *y,float *z)
{
    // 创建跨越式线程ID
    int n = blockIdx.x + threadIdx.x * blockDim.x;
    z[n] = x[n] + y[n];
}

/*
    广播式的非合并访问：该情况下x将永远只访问下标为0的元素
*/
void __global__ add_broadcast(float *x,float *y,float *z)
{
    // 创建线程ID
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    z[n] = x[0] + y[n];
}