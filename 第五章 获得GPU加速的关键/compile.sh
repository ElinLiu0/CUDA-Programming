# nvcc以及g++编译器的编译参数
# -O:优化源代码
#   -O0：不做任何优化
#   -O1：使用默认的优化策略
#   -O2：使用除了10的优化以外，还做一些额外调整工作，例如指令调整等
#   -O3：包括循环展开和其他一些处理特性相关的优化工作
nvcc -O3 -arch=sm_75 add1cpu.cu -o add1cpu.out
# 另外，在add1cpu.cu中，我们使用了#ifdef宏，如果该宏在参数被激活
# 那么所有real类型都将指向double类型，因此计算时间会比float慢一倍