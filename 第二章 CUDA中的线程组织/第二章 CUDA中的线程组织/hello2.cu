#include <stdio.h>
/*
    声明CUDA核函数时，需要在函数名前加上__global__修饰符，
    同时该函数不做任何值的返回，因此需要void类型声明
*/
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}