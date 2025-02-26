#pragma comment(lib, "cublas.lib")
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cublas_v2.h>

const int SIZE = 16000; // 矩阵规模

// 生成随机矩阵
void generateMatrix(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = static_cast<float>(rand() % 10); // 生成0-9之间的随机数
        }
    }
}

// 打印矩阵（仅打印部分数据避免输出过大）
void printMatrix(float* matrix, int N) {
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
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    // 生成随机矩阵
    generateMatrix(h_A, N);
    generateMatrix(h_B, N);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // 创建 cuBLAS 句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    /*
    cublasHandle_t handle;
	•	声明一个 cuBLAS 句柄，类似于 cublas 上下文（context），用于管理 cuBLAS API 的执行状态、设备资源和计算操作。
	•	handle 会在 整个 cuBLAS 计算过程中 传递给 cublasSgemm()。
	cublasCreate(&handle);
	•	初始化 cuBLAS 句柄，分配 GPU 资源，准备进行矩阵计算。
	•	这个句柄 必须在所有 cuBLAS 操作之前创建，并在使用完毕后释放（cublasDestroy(handle);）。
    */

    // 使用 cuBLAS 进行矩阵乘法
    float alpha = 1.0f;
    float beta = 0.0f;
    /*
    作用
	alpha 和 beta 是矩阵乘法的 缩放系数，用于计算以下公式：
        C = alpha * times A * times B + beta * times C
	•	在你的代码中：
	•	alpha = 1.0f → 不缩放 A × B，即 A × B 直接乘法。
	•	beta = 0.0f → 忽略 C 的初始值，即 C = A × B。
    如果 beta 设为 1.0f，则原来的 C 值不会被覆盖，而是加上 A × B。
    */

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    /*
    cublasStatus_t cublasSgemm(cublasHandle_t handle, 
                           cublasOperation_t transA, cublasOperation_t transB,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc);
    
    参数	值	                作用
    handle	handle	        传入 cuBLAS 句柄，用于管理计算。
    transA	CUBLAS_OP_N	    矩阵 A 不转置（N = No Transpose）。
    transB	CUBLAS_OP_N	    矩阵 B 不转置。
    m	    N	            结果矩阵 C 的 行数（即 A 的行数）。
    n	    N	            结果矩阵 C 的 列数（即 B 的列数）。
    k	    N	            A 的列数 / B 的行数（用于 A × B ）。
    alpha	&alpha (1.0f)	缩放因子，这里是 1.0f，即 C = A × B。
    A	    d_A	            输入矩阵 A（存储在 GPU 设备内存）。
    lda	    N	            A 在存储器中的主维度（leading dimension），通常为 N。
    B	    d_B	            输入矩阵 B（存储在 GPU 设备内存）。
    ldb	    N	            B 的 主维度，通常为 N。
    beta	&beta (0.0f)	缩放因子，这里是 0.0f，即 C 不包含原始值。
    C	    d_C	            结果矩阵 C，存储到 GPU 设备内存。
    ldc	    N	            C 的 主维度，通常为 N。
    */

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

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

    // 销毁 cuBLAS 句柄
    cublasDestroy(handle);
    /*
    •	释放 cuBLAS 资源，避免内存泄漏。
	•	这一步必须在所有 cuBLAS 操作完成后执行。
    */

    auto end = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double> duration = end - start;
    std::cout << "\nExecution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}