#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif
const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
// 函数初始化
real reduce(const real *d_x);
void timing(const real *d_x);

int main(void)
{
    // 初始化主机数组
    real *h_x = (real *)malloc(M);
    for(int n = 0;n < N;++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x,M));
    CHECK(cudaMemcpy(d_x,h_x,M,cudaMemcpyHostToDevice));
    // 调用原子加法函数
    printf("\nusing atomicAdd:\n");
    timing(d_x);
    // 释放内存
    free(h_x);
    CHECK(cudaFree(d_x));
}

void __global__ reduce(const real *d_x,real *d_y,const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    // 创建动态共享内存
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    // 同步线程
    __syncthreads();

    for (int offset = blockDim.x >> 1;offset > 0;offset >>= 1)
    {
        if(tid < offset)
        {
            // 在共享内存中实现累加
            s_y[tid] += s_y[tid + offset];
        }
    }
    // 使用原子函数对齐内存
    if (tid == 0)
    {
        atomicAdd(d_y,s_y[0]);
    }
}

real reduce(const real *d_x)
{
    // 指定网格大小以及每个线程块中的动态内存大小
    const int grid_size = (N + BLOCK_SIZE) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;
    // 生成一个float类型栈内存
    real h_y[1] = {0};
    // 将该栈内存复制到GPU中
    real *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(cudaMemcpy(d_y,h_y,sizeof(real),cudaMemcpyHostToDevice));
    // 启动核函数
    reduce<<<grid_size,BLOCK_SIZE,smem>>>(d_x,d_y,N);
    // 拷贝回主机内存
    CHECK(cudaMemcpy(h_y,d_y,sizeof(real),cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));
    // 返回结果 h_y[0]
    return h_y[0];
}

void timing(const real *d_x)
{
    real sum = 0;

    for(int repeat = 0;repeat < NUM_REPEATS;++repeat)
    {
        cudaEvent_t start,stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time,start,stop));
        printf("Time = %g ms.\n",elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("Sum = %f.\n",sum);
}

