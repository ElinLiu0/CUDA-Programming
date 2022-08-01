#include <stdio.h>
__global__ void hello_orld(const int x)
{
    char target;
    target = x;
    printf("Hello %corld\n", target);
}

int main(void)
{
    hello_orld<<<1,1>>>(87);
    cudaDeviceSynchronize();
    return 0;
}