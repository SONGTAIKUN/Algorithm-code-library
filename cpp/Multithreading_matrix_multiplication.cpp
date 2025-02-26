#include <iostream>

//用于随机数生成 (rand() 和 srand(time(0)))。
#include <cstdlib>
#include <ctime>

#include <vector>
#include <thread>       //提供 C++ 多线程支持。
#include <queue>        //提供任务队列 (std::queue)，用于存储任务（行索引）。

//用于线程同步，确保多线程安全访问任务队列。
#include <mutex>
#include <condition_variable>


#include <chrono>         //用于测量计算执行时间


//多核心（动态负载均衡示例代码）


const int SIZE = 1800; // 增大矩阵规模
const int THREAD_COUNT = 8;

std::queue<std::pair<int, int>> taskQueue;          //taskQueue：任务队列，用于存储需要计算的矩阵行范围。
std::mutex queueMutex;                              //queueMutex：互斥锁，用于确保多个线程安全访问 taskQueue。
std::condition_variable queueCondition;             //queueCondition：条件变量，用于线程同步，使线程等待任务到来。
bool stopThreads = false;                           //stopThreads = false：终止标志，用于指示工作线程何时停止。

// 生成随机矩阵
void generateMatrix(int** matrix) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = rand() % 10; // 生成0-9之间的随机数
        }
    }
}

// 打印矩阵（仅打印部分数据避免输出过大）
void printMatrix(int** matrix) {
    for (int i = 0; i < std::min(10, SIZE); i++) { // 仅打印前10行
        for (int j = 0; j < std::min(10, SIZE); j++) { // 仅打印前10列
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// 线程执行的矩阵乘法任务
void multiplyPart(int** A, int** B, int** result, int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 线程工作函数
/** 
    线程 不断从 taskQueue 取出任务，然后调用 multiplyPart() 计算相应部分的矩阵乘法。
    queueCondition.wait(lock, [] { return !taskQueue.empty() || stopThreads; });
    如果任务队列为空，则线程进入等待状态，避免 CPU 过载。
    如果 stopThreads 变为 true 并且 taskQueue 为空，线程退出。
**/
void workerThread(int** A, int** B, int** result) {
    while (true) {
        std::pair<int, int> task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCondition.wait(lock, [] { return !taskQueue.empty() || stopThreads; });

            if (stopThreads && taskQueue.empty()) return;
            
            task = taskQueue.front();
            taskQueue.pop();
        }
        multiplyPart(A, B, result, task.first, task.second);
    }
}

// 任务管理函数
/** 
    创建 10 个线程 处理任务。
    任务分配：
    每 10 行作为一个任务，加入 taskQueue。
    queueCondition.notify_one(); 通知一个等待的线程 开始执行任务。
    终止线程：
    stopThreads = true; 设置终止标志。
    queueCondition.notify_all(); 唤醒所有线程，防止它们被永远阻塞。
**/
void multiplyMatrices(int** A, int** B, int** result) {
    std::vector<std::thread> threads;
    
    for (int i = 0; i < THREAD_COUNT; i++) {
        threads.emplace_back(workerThread, A, B, result);
    }

    for (int startRow = 0; startRow < SIZE; startRow += 10) {
        int endRow = std::min(SIZE, startRow + 10);
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            taskQueue.emplace(startRow, endRow);
        }
        queueCondition.notify_one();
    }

    {
        std::lock_guard<std::mutex> lock(queueMutex);
        stopThreads = true;
    }
    queueCondition.notify_all();

    for (auto& thread : threads) {
        thread.join();
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now(); // 记录开始时间
    
    srand(time(0)); // 初始化随机种子
    
    int** matrixA = new int*[SIZE];
    int** matrixB = new int*[SIZE];
    int** result = new int*[SIZE];
    for (int i = 0; i < SIZE; i++) {
        matrixA[i] = new int[SIZE];
        matrixB[i] = new int[SIZE];
        result[i] = new int[SIZE];
    }
    
    generateMatrix(matrixA);
    generateMatrix(matrixB);
    
    std::cout << "Computing matrix multiplication...\n";
    multiplyMatrices(matrixA, matrixB, result);
    
    std::cout << "Matrix A (first 10x10):" << std::endl;
    printMatrix(matrixA);
    
    std::cout << "\nMatrix B (first 10x10):" << std::endl;
    printMatrix(matrixB);
    
    std::cout << "\nResult (first 10x10):" << std::endl;
    printMatrix(result);
    
    for (int i = 0; i < SIZE; i++) {
        delete[] matrixA[i];
        delete[] matrixB[i];
        delete[] result[i];
    }
    delete[] matrixA;
    delete[] matrixB;
    delete[] result;
    
    auto end = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double> duration = end - start;
    std::cout << "\nExecution time: " << duration.count() << " seconds" << std::endl;
    
    return 0;
}
