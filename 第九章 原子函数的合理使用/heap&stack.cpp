#include <iostream>
#include <stdlib.h>
using namespace std;
int main(void)
{
    const int N = 10;
    cout << "这是一个整数类型的栈内存N，其值为："  << N << endl;
    const int *M;
    cout << "这是一个整数类型的堆内存：指针M，其内存地址为：" << &M << endl;
    const int A[1] = {0};
    cout << "这是一个整数类型的栈内存：数组A，其内存地址为：" << &A << endl;
    cout << "A对应的具体值为：" << A[0] << endl;
    return 0;
}