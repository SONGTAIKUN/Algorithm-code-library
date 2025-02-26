#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

// CUDA 头文件
#include <cuda_runtime.h>

const int SIZE = 16000; // 矩阵规模
const int TILE_SIZE = 32; // CUDA 线程块大小


/*
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
这两行代码的作用是 计算当前线程要处理的矩阵位置 (row, col)。

(1) 线程索引
CUDA 采用 网格 (Grid) + 线程块 (Block) + 线程 (Thread) 三级结构：
	•	线程索引 threadIdx.x & threadIdx.y:
	•	代表当前线程在 线程块 (Block) 内部 的索引。
	•	块索引 blockIdx.x & blockIdx.y:
	•	代表当前线程块在 网格 (Grid) 内部 的索引。
	•	块大小 blockDim.x & blockDim.y:
	•	代表 每个线程块 内有多少个线程。

(2) 计算全局索引

每个线程块中的线程在整个矩阵中都有一个 唯一的全局索引 (row, col)，计算方式如下：
	•	row = blockIdx.y * blockDim.y + threadIdx.y
	•	计算当前线程 所在的行索引：
	•	blockIdx.y * blockDim.y：找到当前线程块的起始行索引。
	•	+ threadIdx.y：在当前线程块内找到具体的线程位置。
	•	col = blockIdx.x * blockDim.x + threadIdx.x
	•	计算当前线程 所在的列索引：
	•	blockIdx.x * blockDim.x：找到当前线程块的起始列索引。
	•	+ threadIdx.x：在当前线程块内找到具体的线程位置。

示例
假设：
	•	N = 16000
	•	blockDim = (32, 32)（即每个线程块包含 32x32 个线程）
	•	Grid = (500, 500)（即网格包含 500x500 个线程块）

那么：
	•	blockIdx.y = 2, threadIdx.y = 5
	•	blockIdx.x = 3, threadIdx.x = 7

计算：
	•	row = 2 * 32 + 5 = 69
	•	col = 3 * 32 + 7 = 103

结论：这个线程计算的是 C[69][103] 的值。
*/
// CUDA 核函数：矩阵乘法
__global__ void matrixMulCUDA(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 生成随机矩阵
void generateMatrix(int* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = rand() % 10; // 生成0-9之间的随机数
        }
    }
}

// 打印矩阵（仅打印部分数据避免输出过大）
void printMatrix(int* matrix, int N) {
    for (int i = 0; i < std::min(10, N); i++) { // 仅打印前10行
        for (int j = 0; j < std::min(10, N); j++) { // 仅打印前10列
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now(); // 记录开始时间

    srand(time(0)); // 初始化随机种子

    int N = SIZE;

    // 分配主机内存
    int* h_A = new int[N * N];
    int* h_B = new int[N * N];
    int* h_C = new int[N * N];

    // 生成随机矩阵
    generateMatrix(h_A, N);
    generateMatrix(h_B, N);

    // 分配设备内存
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * N * sizeof(int));
    cudaMalloc((void**)&d_C, N * N * sizeof(int));
    /*
    1. cudaMalloc 的函数原型
    cudaError_t cudaMalloc(void** devPtr, size_t size);
    参数 devPtr：
    是一个指向指针的指针（void**），用于接收设备内存的地址。
    函数内部会修改 *devPtr 的值，使其指向分配的 GPU 内存首地址。
    参数 size：
    要分配的字节数（例如 N*N*sizeof(int)）。

    各部分的详细作用:
    (1) void** 的作用
    泛型设计：
    void** 是通用指针的指针，允许 cudaMalloc 处理任意类型的指针（如 int*, float* 等）。
    类型安全：
    强制转换为 void** 告诉编译器：“我们知道类型不严格匹配，但此处是安全的”。

    (2) &d_A 的作用
    传递指针的地址：
    d_A 是主机端定义的指针变量（如 int* d_A）。
    &d_A 获取它的地址（即 int**），使 cudaMalloc 能修改 d_A 的值，使其指向 GPU 内存。
    */

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // 定义 CUDA 线程块和网格大小
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    /*
    作用：定义一个二维网格，计算需要多少个线程块才能覆盖整个矩阵。
    关键公式：
    向上取整的整数除法
    blocksPerDim = (N + blockSize - 1) / blockSize
    例如：
    若 N = 16000，blockSize = 32，则 blocksPerDim = 16000 / 32 = 500。
    若 N = 16001，blocksPerDim = (16001 + 31) / 32 = 16032 / 32 = 501。

    1. dim3的用途
    dim3是CUDA API提供的一个三维向量类型，主要用于：
    定义线程块的维度：指定每个线程块（Block）中线程的分布（如1D、2D或3D）。
    定义网格的维度：指定整个网格（Grid）中包含的线程块数量及其排列方式。
    这种多维设计使得线程的索引计算更直观，尤其适用于处理矩阵、图像、物理场等多维数据结构。

    2. dim3的构造函数
    dim3的构造函数支持灵活的参数传递：
    一维初始化：dim3 threadsPerBlock(256)
    等效于dim3(256, 1, 1)，即每个线程块包含256个线程，按一维排列。
    二维初始化：dim3 threadsPerBlock(32, 32)
    等效于dim3(32, 32, 1)，定义一个32x32的二维线程块。
    三维初始化：dim3 threadsPerBlock(16, 16, 4)
    定义一个16x16x4的三维线程块。
    */
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 启动 CUDA 核函数
    matrixMulCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    /*
    在 CUDA 中，核函数（Kernel Function）的调用语法 <<<blocksPerGrid, threadsPerBlock>>> 
    是 NVIDIA 扩展的 C/C++ 语法，用于指定 GPU 并行执行的配置。以下是对该语法的详细解析：
    kernelFunction<<<执行配置>>>(参数列表);
    <<< ... >>>：CUDA 的执行配置符号，用于定义核函数在 GPU 上的并行执行方式。

    执行配置：包含两个核心参数：
    网格维度（blocksPerGrid）：定义网格中包含多少个线程块（Block）。
    线程块维度（threadsPerBlock）：定义每个线程块中包含多少个线程（Thread）。
    参数列表：传递给核函数的参数，与普通函数调用类似，但需注意参数必须位于设备内存（如通过 cudaMalloc 分配的指针）。
    */

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Matrix A (first 10x10):" << std::endl;
    printMatrix(h_A, N);

    std::cout << "\nMatrix B (first 10x10):" << std::endl;
    printMatrix(h_B, N);

    std::cout << "\nResult (first 10x10):" << std::endl;
    printMatrix(h_C, N);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    auto end = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double> duration = end - start;
    std::cout << "\nExecution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}


/*
CUDA 的三级结构：网格 (Grid) + 线程块 (Block) + 线程 (Thread)

CUDA 采用一种层次化的并行计算模型，将并行任务分配给 GPU 上的多个 线程，并将它们组织成 线程块 (Block)，这些线程块
又组成 网格 (Grid)。这种分层结构允许程序在处理大规模并行计算时更加高效，且易于管理。
下面是 网格 (Grid)、线程块 (Block) 和 线程 (Thread) 的详细介绍。


1. 线程 (Thread)

线程是 CUDA 中的最小计算单元，每个线程负责执行一部分任务。
线程的特性：
	•	每个线程有一个唯一的索引，可以通过 threadIdx 获取它在其所属线程块中的位置。线程的索引可以是一维、二维或三维。
	•	线程被分配到一个线程块中，线程块中的所有线程可以共享一些资源（例如共享内存）。
	•	线程之间是独立执行的，但它们可以通过同步机制（例如 __syncthreads()）来协作。

线程索引：
每个线程在其所属的线程块中都有一个 局部索引。例如，如果你定义一个 二维线程块，每个线程都会有 threadIdx.x 和 threadIdx.y 来指示它在块中的位置。
	•	线程索引可以通过 threadIdx.x，threadIdx.y 和 threadIdx.z 来访问，分别对应 一维、二维和三维线程块。
	•	可以通过 计算全局索引 来确定当前线程在整个网格中的位置。

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

•	threadIdx.x：当前线程在块中的索引。
•	blockIdx.x：当前线程块在网格中的索引。
•	blockDim.x：每个线程块的线程数。

2. 线程块 (Block)

线程块是多个线程组成的集合。一个线程块中的线程会共同工作，且在同一个线程块内的线程可以访问共享内存。
线程块的特性：
	•	每个线程块也有一个索引，可以通过 blockIdx 来获取。线程块的索引在整个网格中是唯一的。
	•	线程块内部的线程可以通过 共享内存 来协作，彼此之间的通信比全局内存快得多。
	•	线程块内的线程可以通过 同步操作 来保证它们按照一定的顺序执行。
	•	线程块有大小限制，通常的限制是每个线程块最多包含 1024 个线程（具体值依赖于硬件）。

线程块的维度：
	•	可以是 一维、二维 或 三维 的。通常来说，二维和三维的线程块适用于矩阵、图像等二维或三维数据的处理。

dim3 threadsPerBlock(16, 16);  // 每个线程块 16x16 = 256 个线程
dim3 blocksPerGrid(32, 32);    // 网格大小 32x32 个线程块

•	threadsPerBlock 表示每个线程块中的线程数目，这里是 16x16 的二维线程块，共 256 个线程。
•	blocksPerGrid 表示网格中线程块的数量，这里是 32x32 的二维网格。
    
3. 网格 (Grid)

网格是由多个线程块组成的集合。网格可以包含 一维、二维 或 三维 的线程块。
网格的特性：
	•	网格由多个线程块组成，每个线程块执行相同的代码，但可以处理不同的数据。
	•	网格的规模和维度由用户定义，通常会根据问题的规模来设置网格的维度。
	•	不同线程块之间是独立执行的，线程块间不共享内存，也不能直接通信。但是，它们可以通过全局内存来交换数据。

dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

•	blocksPerGrid 表示网格中 线程块的数量。这个值由矩阵的大小 N 和每个线程块的大小 threadsPerBlock 决定。
*/