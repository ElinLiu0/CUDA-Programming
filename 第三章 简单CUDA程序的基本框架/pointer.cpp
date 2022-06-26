#include <stdint.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
// 创建一个整数变量用于存放数组长度
const int n_SAMPLE = 100;
/*
	创建一个int类型变量用来存储内存变量的大小
	使用sizeof()函数获取到int类型数据的单位大小
	并乘以n_SAMPLE，即为数组将使用malloc()函数生成的大小
*/
const int MEM_SPACE = sizeof(int) * n_SAMPLE;
// 使用malloc()函数并结合之生成的整数矩阵空间生成数组
int* Array = (int*)malloc(MEM_SPACE);

int main(void)
{
	for (int x = 0; x < n_SAMPLE; x++) {
		// 迭代赋值
		Array[x] = x;
	}
	for (int i = 0; i < n_SAMPLE; i++) {
		// 迭代打印
		cout << Array[i] << endl;
	}
}