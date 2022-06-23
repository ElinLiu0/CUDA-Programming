#include <stdio.h>

__global__ void hello_from_gpu(void){
    printf("Hello from GPU!\n");
}

int main(void){
    /*
        此时cuda内核将调用总共8个线程来运行程序，
        其中2代表gridNum，4代表BlockNum,
        此时函数也将被调用2*4=8次
    */
    hello_from_gpu<<<2,4>>>();
    return 0;
}