#include "error.cuh"
#include <stdio.h>

int main(int argc,char *argv[])
{
    // 设置一个整数变量存储设备ID
    int deviceId = 0;
    if(argc > 1)
    {
        deviceId = atoi(argv[1]);
    }
    // 设置当前CUDA设备为第deviceId号设备
    CHECK(cudaSetDevice(deviceId)); 
    // 创建cudaDeviceProp结构体变量存储设备属性
    cudaDeviceProp prop;
    // 使用cudaGetDeviceProperties函数获取设备属性，接收参数：cudaDeviceProp结构体指针，设备ID
    CHECK(cudaGetDeviceProperties(&prop,deviceId));
    // 打印设备ID
    printf("Device ID: %d\n",deviceId);
    // 打印设备名称
    printf("Device Name: %s\n",prop.name);
    // 打印设备计算能力
    printf("Compute Capability: %d.%d\n",prop.major,prop.minor);
    // 打印设备显存大小
    printf("Total Global Memory: %.2f GB\n",prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    // 打印常量显存大小
    printf("Constant Memory: %g KB\n",prop.totalConstMem / 1024.0);
    // 打印最大网格数量
    printf("Max Grid Size is : %d by %d by %d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
    // 打印最大线程块数量
    printf("Max block size is : %d by %d by %d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
    
    // 打印设备中的CUDA SM数量
    printf("Total number of SMs: %d\n",prop.multiProcessorCount);
    // 打印每个线程块下的共享内存大小
    printf("Shared Memory per Block: %g KB\n",prop.sharedMemPerBlock / 1024.0);
    // 打印每组SM下的共享内存大小
    printf("Shared Memory per SM: %g KB\n",prop.sharedMemPerMultiprocessor / 1024.0);
    // 打印每个线程块下的寄存器数量
    printf("Registers per Block: %d K\n",prop.regsPerBlock);
    // 打印每组SM下的寄存器数量
    printf("Registers per SM: %d K\n",prop.regsPerMultiprocessor);
    // 打印每组线程块下的最大线程数量
    printf("Max Threads per Block: %d\n",prop.maxThreadsPerBlock);
    // 打印每组SM下的最大线程数量
    printf("Max Threads per SM: %d\n",prop.maxThreadsPerMultiProcessor);
    return 0;
}
