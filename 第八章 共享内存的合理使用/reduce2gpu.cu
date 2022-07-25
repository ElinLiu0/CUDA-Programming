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

void timing(real *h_x,real *d_x,const int method);

int main(void)
{
    // 创建主机数组
    real *h_x = (real *)malloc(M);
    for(int n = 0; n < N; n++)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc((void **)&d_x,M));
    // 分别测试使用全局内存、静态共享内存、动态共享内存三种方法
    printf("\nUsing global memory only:\n");
    timing(h_x,d_x,0);
    printf("\nUsing static shared memory:\n");
    timing(h_x,d_x,1);
    printf("\nUsing dynamic shared memory:\n");
    timing(h_x,d_x,2);
    // 释放资源
    free(h_x);
    CHECK(cudaFree(d_x));
}

void __global__ reduce_global(real *d_x,real *d_y)
{
    // 创建一个线程索引
    const int tid = threadIdx.x;
    // 创建x维度的索引
    real *x = d_x + blockDim.x * blockIdx.x;
    // 使用移位操作符将x的值累加到y中
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if(tid < offset)
        {
            // 用d_y[tid+offset]的元素值累加到d_y线程下标下
            d_y[tid] += d_y[tid + offset];
        }
        __syncthreads();
    }
    // 如果线程下标为0，则将x的值累加到y中
    if(tid == 0)
    {
        d_y[blockIdx.x] = x[0];
    }
}

void __global__ reduce_shared(real *d_x,real *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    // 创建二维线程索引
    const int n = bid * blockDim.x + tid;
    // 创建共享内存
    __shared__ real s_y[BLOCK_SIZE];
    // 使用条件表达式将d_x值拷贝到共享内存中
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();
    // 使用移位操作进行迭代
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if(tid < offset)
        {
            // 用s_y[tid+offset]的元素值累加到s_y线程下标下
            s_y[tid] += s_y[tid + offset];
        }
        // 同步线程
        __syncthreads();
    }
    if(tid == 0)
    {
        // 将共享内存拷贝到d_y中
        d_y[bid] = s_y[0];
    }
}

void __global__ reduce_dynamic(real *d_x,real *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    /*
    extern是一个关键字，它告诉编译存在着一个变量或者一个函数，如果在当前编译语句的前面中
    没有找到相应的变量或者函数，也会在当前文件的后面或者其他文件中定义，
    使用该关键字可以使得现有的共享内存跳脱出核函数的限制，实现动态内存共享
    */
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if(tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}
// 创建主机端函数
real reduce(real *d_x,const int method)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y;
    CHECK(cudaMalloc((void **)&d_y,ymem));
    real *h_y = (real *)malloc(ymem);
    // 使用switch关键字选择分配的核函数启动方式
    switch (method)
    {
    case 0:
        reduce_global<<<grid_size,BLOCK_SIZE>>>(d_x,d_y);
        break;
    case 1:
        reduce_shared<<<grid_size,BLOCK_SIZE>>>(d_x,d_y);
        break;
    case 2:
        reduce_dynamic<<<grid_size,BLOCK_SIZE>>>(d_x,d_y);
        break;
    default:
        break;
    }

    CHECK(cudaMemcpy(h_y,d_y,ymem,cudaMemcpyDeviceToHost));
    // 初始化result为0
    real result = 0.0;
    for(int n = 0; n < grid_size; n++)
    {
        // 通过将从GPU内存拷贝回主机内存，计算结果
        result += h_y[n];
    }
    // 释放内存
    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}
// 创建计时函数
void timing(real *h_x,real *d_x,const int method)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; repeat++)
    {
        CHECK(cudaMemcpy(d_x,h_x,M,cudaMemcpyHostToDevice));
        
        cudaEvent_t start,stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start,0));
        cudaEventQuery(start);

        sum += reduce(d_x,method);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed;
        CHECK(cudaEventElapsedTime(&elapsed,start,stop));
        printf("Time = %g ms.\n",elapsed);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    printf("Sum = %f5.\n",sum);
}