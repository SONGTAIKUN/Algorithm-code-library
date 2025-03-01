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


const int SIZE = 2000; // 增大矩阵规模
const int THREAD_COUNT = 24;

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
    /*
    工作线程在一个无限循环中运行，直到所有任务完成并被通知退出。

    每次循环中，线程尝试从任务队列中获取一个任务并执行。
    */
    while (true) {
        std::pair<int, int> task;
        {
            /*使用 std::unique_lock<std::mutex> 对任务队列的访问进行加锁。

            queueMutex 是一个互斥锁，用于保护任务队列 taskQueue 的访问。

            queueCondition.wait(lock, predicate) 是一个条件变量等待函数：

            如果任务队列为空且 stopThreads 为 false，线程会释放锁并进入等待状态。

            当主线程调用 queueCondition.notify_one() 或 queueCondition.notify_all() 时，线程会被唤醒。

            唤醒后，线程会重新检查条件（!taskQueue.empty() || stopThreads），如果条件满足，则继续执行；否则继续等待。
            */

            /*
            std::unique_lock 是一个模板类，用于管理互斥锁 (std::mutex) 的锁定和解锁操作。

            与 std::lock_guard 类似，std::unique_lock 也会在构造时自动锁定互斥锁，并在析构时自动解锁。
            但与 std::lock_guard 不同的是，std::unique_lock 提供了更灵活的控制，允许手动锁定和解锁，
            并且可以与条件变量一起使用。
            */
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCondition.wait(lock, [] { return !taskQueue.empty() || stopThreads; });
            /*
            queueCondition.wait(lock, predicate):
            queueCondition 是一个条件变量 (std::condition_variable)，用于线程间的同步。
            wait 是条件变量的成员函数，用于使当前线程进入等待状态，直到某个条件满足。
            wait 函数有两个参数：
            lock: 这是一个 std::unique_lock<std::mutex> 对象，表示当前线程持有的互斥锁。在调用 wait 时，wait 会自动释放 lock 所持有的互斥锁，使其他线程可以访问共享资源。
            predicate: 这是一个可调用对象（通常是 lambda 表达式），用于检查某个条件是否满足。wait 函数会在每次被唤醒时调用这个 predicate，如果 predicate 返回 true，则线程继续执行；如果返回 false，则线程继续等待。
            [] { return !taskQueue.empty() || stopThreads; }:
            这是一个 lambda 表达式，作为 wait 的第二个参数，用于检查条件是否满足。
            这个 lambda 表达式返回一个布尔值，表示任务队列 taskQueue 是否非空，或者 stopThreads 是否为 true。
            如果 taskQueue 不为空（即有任务需要处理），或者 stopThreads 为 true（表示所有任务已完成，线程可以退出），则 wait 函数会返回，线程继续执行。
            如果 taskQueue 为空且 stopThreads 为 false，则线程会继续等待，直到其他线程调用 queueCondition.notify_one() 或 queueCondition.notify_all() 唤醒它。
            */

            //如果 stopThreads 为 true（表示所有任务已完成）且任务队列为空，线程退出。这是线程退出的唯一条件。
            if (stopThreads && taskQueue.empty()) return;
            



            /*从任务队列中获取一个任务（task），任务是一个 std::pair<int, int>，表示行的范围（startRow 到 endRow）。

            使用 taskQueue.front() 获取队列中的第一个任务。
            使用 taskQueue.pop() 将队伍中第一个任务从队列中移除。
            */
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
    
    /*
    使用 emplace_back 创建 THREAD_COUNT 个线程，每个线程执行 workerThread 函数。

    workerThread 是实际执行矩阵乘法任务的函数，A、B 和 result 作为参数传递给线程。
    */

    //emplace_back向vector末尾添加元素时，直接在容器的内存位置构造该元素，而不是先构造一个元素，然后复制或移动到vector中，这样做可以显著提高性能。
    for (int i = 0; i < THREAD_COUNT; i++) {
        threads.emplace_back(workerThread, A, B, result);
    }

    /*
    将矩阵按行分块，每块包含 10 行（startRow 到 endRow）。

    使用 taskQueue.emplace(startRow, endRow) 将每个任务块（即行的范围）加入任务队列。

    使用 lock_guard<mutex> 保护任务队列的访问，避免多线程竞争。

    每次加入任务后，调用 queueCondition.notify_one() 通知一个等待的线程去处理任务。
     */
    for (int startRow = 0; startRow < SIZE; startRow += 10) {
        int endRow = std::min(SIZE, startRow + 10);
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            taskQueue.emplace(startRow, endRow);
        }
        queueCondition.notify_one();
    }

    /*
    设置 stopThreads 标志为 true，表示所有任务已经分配完毕。

    使用 queueCondition.notify_all() 通知所有线程检查 stopThreads 标志并退出。

    1. std::lock_guard
    作用：
    std::lock_guard 是一个模板类，用于自动管理互斥锁的生命周期。
    它在构造时加锁，在析构时自动释放锁，确保锁的正确释放，即使发生异常。

    2. std::mutex
    作用：
    std::mutex 是 C++ 标准库提供的互斥锁类，用于保护共享资源的访问。
    它提供了 lock() 和 unlock() 方法，分别用于加锁和释放锁。

    3. queueMutex
    作用：
    queueMutex 是一个 std::mutex 对象，用于保护任务队列 taskQueue 或标志变量 stopThreads 的访问。
    它是一个共享的互斥锁，多个线程通过它来同步对共享资源的访问。

    4. lock
    作用：
    lock 是 std::lock_guard<std::mutex> 的一个实例（对象）。
    它的名字可以是任意的，但通常命名为 lock 以表明其用途。

    5. 整体形式 std::lock_guard<std::mutex> lock(queueMutex);
    语法解析：
    std::lock_guard<std::mutex>：模板类实例化，指定锁的类型为 std::mutex。
    lock(queueMutex)：构造函数调用，传入需要管理的互斥锁 queueMutex。
    lock：实例名称，表示一个 std::lock_guard 对象。
    执行过程：
    构造 lock 对象时，自动调用 queueMutex.lock()，加锁互斥锁。
    在 lock 对象的作用域内，共享资源（如任务队列或标志变量）的访问是线程安全的。
    当 lock 对象离开作用域时（例如，作用域结束或发生异常），自动调用 queueMutex.unlock()，释放互斥锁。
    */
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        stopThreads = true;
    }
    queueCondition.notify_all();

    //使用 join() 等待所有线程完成任务并退出。
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
