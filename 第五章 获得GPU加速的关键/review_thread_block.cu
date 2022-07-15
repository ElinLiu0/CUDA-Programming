#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
void __global__ kernel(int* x,int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N)
    {
        if(x[i] > x[i+1])
        {
            printf("%d is over than %d\n",x[i],x[i+1]);
        } else if (x[i] < x[i+1])
        {
            printf("%d is under than %d\n",x[i],x[i+1]);
        } else
        {
            printf("%d is equal to %d\n",x[i],x[i+1]);
        }
    }   
}
int main(void)
{
    const int N = 1000;
    const int M = sizeof(int) * N;
    int* x = (int*)malloc(M);
    srand((unsigned)time(NULL));
    for(int i = 0;i < N;i++)
    {
        x[i] = rand() % 100;
    }
    int* d_x;
    cudaMalloc((void**)&d_x,M);
    cudaMemcpy(d_x,x,M,cudaMemcpyHostToDevice);
    const int blockSize = 40;
    const int gridSize = N / blockSize;
    kernel<<<gridSize,blockSize>>>(d_x,N);
    cudaMemcpy(x,d_x,M,cudaMemcpyDeviceToHost);
    return 0;
}