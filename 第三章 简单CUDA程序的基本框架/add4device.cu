#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
// 现在头部声明三个函数的名称以及参数
void __global__ add1(const double *x, const double *y, double *z, const int N);
void __global__ add2(const double *x, const double *y, double *z, const int N);
void __global__ add3(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000001;
    const int M = sizeof(double) * N;
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);
    // 定义block_size为128
    const int block_size = 128;
    // 为了保障CUDA线程能顺利启动，因此我们需要再增加127以保障grid_size是一个整数
    const int grid_size = (N + block_size - 1) / block_size;
    /*
        因为此处调用了三次核函数，因此我们无论是校验，
        还是执行cudaMemcpy()函数将主机数组传递给核函数
        用需要执行三次
    */
    add1<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    add2<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    add3<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

// 版本一：有返回值的设备函数
double __device__ add1_device(const double x,const double y)
{
    return (x + y);
}
void __global__ add1(const double *x,const double *y,double *z,
    const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = add1_device(x[n],y[n]);
    }
}

// 版本二：使用指针的设备函数
void __device__ add2_device(const double x,const double y,double *z)
{
    *z = x + y;
}

void __global__ add2(const double *x,const double *y,double *z,const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N)
    {
        add2_device(x[n],y[n],&z[n]);
    }
}

// 版本三：用引用(reference)的设备函数
void __device__ add3_device(const double x,const double y,double &z)
{
    z = x + y;
}

void __global__ add3(const double *x,const double *y,double *z,const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        add3_device(x[n],y[n],z[n]);
    }
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}
