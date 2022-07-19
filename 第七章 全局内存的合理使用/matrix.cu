#include <stdio.h>
#include <stdlib.h>
#include "error.cuh"
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif
// 初始化循环次数
const int NUM_REPEATS = 10;
// 设置单行矩阵长度为32
const int TILE_DIM = 32;
// 创建GPU上的复制函数，将目标GPU矩阵从A复制到B
__global__ void copy(const real *A, real *B, const int N)
{
    /*创建x和y两个变量，用于记录矩阵的行和列，该值随线程ID不断变化
    通过不断让blockIdx和TILE_DIM进行相乘，这样可以保证每一个矩阵
    切片的元素长度都能保持在32。
    */
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    // 创建一个索引
    const int index = ny * N + nx;
    if (nx < N && ny < N)
    {
        // 将矩阵下标为index的元素复制到矩阵B下
        B[index] = A[index];
    }
}
/*
    创建第一种转置函数，因为在我们在上述复制的过程中。
    矩阵的索引通过y * 长度 + x来计算，而转置的索引通过x * 长度 + y来计算。
    换算到数学公式上就是：B(xy) = A(yx)。
*/
__global__ void transpose1(const real *A,real *B,const int N)
{
    // 初始化x和y两个变量，用于记录矩阵的行和列，该值随线程ID不断变化
    const int nx = blockDim.x * blockDim.x + threadIdx.x;
    const int ny = blockDim.y * blockIdx.x + threadIdx.y;
    if (nx < N && ny < N){
        /* 进行第一种转置，这种情况下，矩阵A的读取顺序是顺序读写，而矩阵B的读取顺序是乱序读写
        因此，我们可以这样形容：transpose1函数是是合并和非合并（对应于A和B的读取顺序）的转置函数。
        */
        B[nx * N + ny] = A[ny * N + nx];
    }
}
/*
    创建第二种转置函数，结构与第一种转置函数保持一致，
    唯一的区别是，第二种转置函数的矩阵B的读取顺序是顺序读写，
    矩阵A的读取顺序是乱序读写。
*/
__global__ void transpose2(const real *A,real *B,const int N)
{
    const int nx = blockDim.x * blockDim.x + threadIdx.x;
    const int ny = blockDim.y * blockIdx.x + threadIdx.y;
    if (nx < N && ny < N){
        B[ny * N + nx] = A[nx * N + ny];
    }
}
/*
    创建第三种转置函数，该函数唯一与前两种不同的地方在于，
    在该函数内，矩阵A将使用__ldg()函数将片上元素读入到缓存中
*/
__global__ void transpose3(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = __ldg(&A[nx * N + ny]);
    }
}

void print_matrix(const int N,const real *A)
{
    for (int ny = 0;ny < N;ny++)
    {
        for (int nx = 0;nx < N;nx++)
        {
            printf("%g\t ",A[ny * N + nx]);
        }
        printf("\n");
    }
}

void timing(const real *d_A,real *d_B,const int N,const int task)
{
    // 初始化二维网格
    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_size_y = (N + TILE_DIM - 1) / TILE_DIM;
    const dim3 block_size(TILE_DIM,TILE_DIM);
    const dim3 grid_size(grid_size_x,grid_size_y);
    // 创建一个时间计数器
    float t_sum = 0;
    float t2_sum = 0;
    for(int repeat = 0;repeat <= NUM_REPEATS;++repeat)
    {
        // 创建循环开始计时和结束cuda事件
        cudaEvent_t start,stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        // 开始计时
        CHECK(cudaEventRecord(start,0));
        cudaEventQuery(start);
        // 创建switch来判断执行哪一种转置函数
        switch (task)
        {
        case 0:
            copy<<<grid_size,block_size>>>(d_A,d_B,N);
            break;
        case 1:
            transpose1<<<grid_size,block_size>>>(d_A,d_B,N);
            break;
        case 2:
            transpose2<<<grid_size,block_size>>>(d_A,d_B,N);
            break;
        case 3:
            transpose3<<<grid_size,block_size>>>(d_A,d_B,N);
            break;
        default:
            printf("Error Task not found\n");
            exit(1);
            break;
        }
        // 结束计时
        CHECK(cudaEventRecord(stop));
        // 同步事件
        CHECK(cudaEventSynchronize(stop));
        // 计时结束，获取时间
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time,start,stop));
        printf("Time = %g ms\n",elapsed_time);

        if(repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }
        // 销毁CUDA事件
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_avg = t_sum / NUM_REPEATS;
    const float t_err = sqrt((t2_sum / NUM_REPEATS) - t_avg * t_avg);
    printf("Average time = %g ms +- %g ms\n",t_avg,t_err);
}

int main(int argc,char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <N>\n",argv[0]);
        exit(1);
    }
    const int N = atoi(argv[1]);
    const int N2 = N * N;
    const int M = sizeof(real) * N2;
    real *h_A = (real *)malloc(M);
    real *h_B = (real *)malloc(M);
    for (int n = 0;n < N2;n++)
    {
        h_A[n] = n;
    }
    real *d_A,*d_B;
    CHECK(cudaMalloc(&d_A,M));
    CHECK(cudaMalloc(&d_B,M));
    CHECK(cudaMemcpy(d_A,h_A,M,cudaMemcpyHostToDevice));

    printf("\ncopy:\n");
    timing(d_A,d_B,N,0);
    CHECK(cudaMemcpy(h_B,d_B,M,cudaMemcpyDeviceToHost));
    printf("\ntranspose with coalesced read:\n");
    timing(d_A,d_B,N,1);
    CHECK(cudaMemcpy(h_B,d_B,M,cudaMemcpyDeviceToHost));
    printf("\ntranspose with coalesced write:\n");
    timing(d_A,d_B,N,2);
    CHECK(cudaMemcpy(h_B,d_B,M,cudaMemcpyDeviceToHost));
    printf("\ntranspose with coalesced write and __ldg read:\n");
    timing(d_A,d_B,N,3);
    CHECK(cudaMemcpy(h_B,d_B,M,cudaMemcpyDeviceToHost));

    if(N <= 10)
    {
        printf("\nA:\n");
        print_matrix(N,h_A);
        printf("\nB:\n");
        print_matrix(N,h_B);
    }
    free(h_A);
    free(h_B);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    return 0;
}