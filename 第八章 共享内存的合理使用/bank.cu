#include <stdlib.h>
#include <stdio.h>
#include "error.cuh"
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif
const int TILE_DIM = 32;
const int NUM_REPEATS = 10;

// 函数初始化
void timing(const real *d_A,real *d_B,const int N,const int task);
// 初始化两个转置函数，一个有内存bank冲突，一个没有内存bank冲突
__global__ void transpose1(const real *A,real *B,const int N);
__global__ void transpose2(const real *A,real *B,const int N);
void print_matrix(const real *A,const int N);


int main(int argc,char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <N>\n",argv[0]);
        return 1;
    }
    // 将运行参数转换为整数
    const int N = atoi(argv[1]);

    const int N2 = N * N;
    // 分配内存空间
    const int M = sizeof(real) * N2;
    real *h_A = (real *)malloc(M);
    real *h_B = (real *)malloc(M);
    // 初始化数据
    for(int n = 0;n < N2;++n)
    {
        h_A[n] = n;
    }
    // 初始化GPU内存
    real *d_A,*d_B;
    CHECK(cudaMalloc(&d_A,M));
    CHECK(cudaMalloc(&d_B,M));
    // 将数据传输到GPU内存
    CHECK(cudaMemcpy(d_A,h_A,M,cudaMemcpyHostToDevice));

    // 执行函数
    printf("\ntranspose with shared memory bank conflict\n");
    timing(d_A,d_B,N,1);
    printf("\ntranspose without shared memory bank conflict\n");
    timing(d_A,d_B,N,2);
    // 将数据传输回主内存
    CHECK(cudaMemcpy(h_B,d_B,M,cudaMemcpyDeviceToHost));
    if (N <= 10)
    {
        printf("A =\n");
        print_matrix(h_A,N);
        printf("B =\n");
        print_matrix(h_B,N);
    }

    // 释放内存
    free(h_A);
    free(h_B);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    return 0;
}

void timing(const real *d_A,real *d_B,const int N,const int task)
{
    // 初始化grid_size_x和grid_size_y
    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_size_y = (N + TILE_DIM - 1) / TILE_DIM;
    // 创建二维block_size和二维grid_size
    const dim3 block_size(TILE_DIM,TILE_DIM);
    const dim3 grid_size(grid_size_x,grid_size_y);

    // 初始化时间变量
    float t_sum = 0;
    float t2_sum = 0;
    // 循环执行NUM_REPEATS次
    for (int repeat = 0;repeat <= NUM_REPEATS;++repeat)
    {
        // 创建计时器
        cudaEvent_t start,stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        // 开始计时
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        // 条件执行函数
        switch (task)
        {
            case 1:
                // 执行函数1
                transpose1<<<grid_size,block_size>>>(d_A,d_B,N);
                break;
            case 2:
                // 执行函数2
                transpose2<<<grid_size,block_size>>>(d_A,d_B,N);
                break;
            default:
                printf("Invalid Task Specified\n");
                exit(1);
                break;
        }

        CHECK(cudaEventRecord(stop));
        // 同步事件
        CHECK(cudaEventSynchronize(stop));
        // 计时结束
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time,start,stop));
        printf("Time = %g ms.\n",elapsed_time);

        // 累加时间
        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        // 释放计时器
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_mean = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_mean * t_mean);
    printf("Time = %g +/- %g ms.\n",t_mean,t_err);
}

__global__ void transpose1(const real *A,real *B,const int N)
{
    // 创建一个共享内存用于承接A和B的一片(TILE)数据
    // 但是该操作会导致全局内存产生bank冲突
    // 例如s[0] = A[0]，但该过程没有加载完时，s[1] = A[0]
    __shared__ real shared[TILE_DIM][TILE_DIM];
    // 创建对应的块坐标
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    // 创建线程坐标
    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    // 复制A的片上数据到共享内存
    if (nx1 < N && ny1 < N)
    {
        shared[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    // 同步线程保证共享内存中的数据是完整的
    __syncthreads();

    // 创建第二个线程坐标负责将共享内存中的数据复制到B中
    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 * N + ny2] = shared[threadIdx.x][threadIdx.y];
    }
}
// 创建第二种方法的函数
__global__ void transpose2(const real *A,real *B,const int N)
{
    // 创建一个共享内存用于承接A和B的一片(TILE)数据
    // 但是这一次我们会尽可能的避免bank冲突
    // 通过让共享内存的坐标为32,33
    // 因此数据跨度为33，因此一个线程块中的线程分别访问bank中32层的数据
    __shared__ real shared[TILE_DIM][TILE_DIM+1];
    // 创建对应的块坐标
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;
    // 创建线程坐标
    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    // 复制A的片上数据到共享内存
    if (nx1 < N && ny1 < N)
    {
        shared[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    // 同步线程保证共享内存中的数据是完整的
    __syncthreads();

    // 创建第二个线程坐标负责将共享内存中的数据复制到B中
    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 * N + ny2] = shared[threadIdx.x][threadIdx.y];
    }
}

void print_matrix(const real *A,const int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%g\t ",A[i * N + j]);
        }
        printf("\n");
    }
}