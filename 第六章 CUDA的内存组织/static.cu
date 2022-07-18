#include "error.cuh"
#include <stdio.h>
// 创建静态设备变量
__device__ int d_x = 1;
// 创建静态设备数组
__device__ int d_y[2];

void __global__ kernel(void)
{
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x = %d\n", d_x);
    printf("d_y[0] = %d\n", d_y[0]);
    printf("d_y[1] = %d\n", d_y[1]);
}

int main(void)
{
    int h_y[2] = {10,20};
    // 使用cudaMemcpyToSymbol()函数将主机静态变量传递给设备静态变量
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));

    kernel<<<1,1>>>();
    CHECK(cudaDeviceSynchronize());
    // 使用cudaMemcpyFromSymbol()函数将设备静态变量传递给主机静态变量
    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    printf("h_y[0] = %d\n", h_y[0]);
    printf("h_y[1] = %d\n", h_y[1]);
    return 0;
}