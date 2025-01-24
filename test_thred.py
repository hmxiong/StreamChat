import threading
import time
import gc

# 示例 GPU 资源释放函数（根据实际使用的框架进行调整）
def release_gpu_resources():
    # 假设使用的是 PyTorch
    import torch
    torch.cuda.empty_cache()

# 创建一个全局事件对象
stop_event = threading.Event()

# 定义线程A的函数
def thread_a_function():
    # for i in range(5):
    i = 0
    while i < 5:
        print("线程A正在运行...", i)
        time.sleep(1)
        i += 1
    print("线程A已结束")
    # 线程A结束时设置事件
    stop_event.set()

# 定义线程B的函数
def thread_b_function():
    import torch
    # 在GPU上分配一些变量
    tensor = torch.randn(1000, 1000).cuda()
    while not stop_event.is_set():
        print("线程B正在运行...")
        time.sleep(1)
    print("线程B已结束")
    # 清理GPU上的变量
    del tensor
    torch.cuda.empty_cache()

# 主循环中重复启动线程A和线程B
for _ in range(4):
    # 重置事件
    stop_event.clear()
    
    # 创建并启动线程A
    thread_a = threading.Thread(target=thread_a_function)

    
    # 创建并启动线程B
    thread_b = threading.Thread(target=thread_b_function)
    
    thread_a.start()    
    thread_b.start()
    
    # 等待线程A结束
    thread_a.join()
    
    # 等待线程B结束
    thread_b.join()
    
    # 强制进行垃圾回收以确保资源被释放
    gc.collect()
    
    # 清理GPU资源
    release_gpu_resources()
    
    print("主循环已结束一次")

print("所有线程已结束")