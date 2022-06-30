#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
using namespace std;
__global__ void Solution(int length,int array[])
{
    int tempValue = 0;
    int MaxValue = 0;
    for(int i=0;i<length;i++){
        if(tempValue == 0){
            tempValue = array[i];
            MaxValue = tempValue;
            printf("Spawning Begin Value with : %d\n",tempValue);
        } else {
            if(tempValue > MaxValue){
                printf("Replacing Current MaxValue : %d with new Value : %d.\n",MaxValue,tempValue);
                MaxValue = tempValue;
            } else if (tempValue == MaxValue) {
                printf("Current Value is tied with MaxValue,skip!\n");
            } else if (tempValue < MaxValue){
                printf("Current Value : %d is less than MaxValue : %d,ignored!\n",tempValue,MaxValue);
            }
            tempValue = array[i];
        }
    }
}

int main(void)
{
    // 创建一个整数变量存储数组长度
    const int n_SAMPLE = 100;
    // 使用srand()函数定义随机种子
    srand((unsigned)time(NULL));
    // 使用一个int变量存储内存的空间大小
    const int MEM_SPACE = sizeof(int) * n_SAMPLE;
    // 使用malloc()分配数组内存
    int* Array = (int*)malloc(MEM_SPACE);
    // 使用随机数填充
    for(int i=0;i<n_SAMPLE;i++){
        Array[i] = rand();
    }
    // 创建一个整数指针用于存放CUDA数组
    int *cudaArray;
    /* 使用cudaMalloc()函数来对其进行内存分配，
    指定参数顺序为:整数指针，内存大小
    这里(int **)将把cudaArray强制转换为双重指针
    也可以使用不传递该类型直接调用该指针进行分配
    */
    cudaMalloc((int **)&cudaArray,MEM_SPACE);
    // 使用cudaMemcpy()函数
    cudaMemcpy(cudaArray,Array,MEM_SPACE,cudaMemcpyHostToDevice);
    // 启动CUDA核函数
    Solution<<<1,1>>>(n_SAMPLE,cudaArray);
    // 同步
    cudaDeviceSynchronize();
    // 释放内存
    free(Array);
    cudaFree(cudaArray);
    return 0;
}