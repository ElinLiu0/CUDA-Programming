/*
    do while的执行逻辑为：优先do,后去while。
    即为：先执行do代码块中的代码，然后再去执行while中的逻辑判断
    因此，do while代码块中至少会执行一次do，也正因为此，我们需要
    使用do while来验证我们的CUDA核函数或设备函数执行是否发生了异常
*/
#pragma once
#include <stdio.h>
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error Occurred:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)
