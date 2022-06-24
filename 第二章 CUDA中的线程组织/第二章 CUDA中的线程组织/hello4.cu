#include <stdio.h>

__global__ void hello_from_gpu(void){
    // 读取blockIdx.X
    const int bx = blockIdx.x;
    // 读取threadIdx.x和threadIdx.y
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // 打印出当前CUDA线程的线程束
    printf("Spawning GPU Thread from block-%d and thread-(%d,%d)!\n", bx, tx, ty);
}

int main(void){
    // 构建一个类型为dim3的block_size变量
    const dim3 block_size(2,4);
    hello_from_gpu<<<1,block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}