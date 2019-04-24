import tensorflow as tf
import numpy as np
import threading
import time


def MyLoop(coord, worder_id):
    # 线程不停从coord那儿检查should_stop是否为True：即停止
    while not coord.should_stop():
        # 随机停止所有线程
        if np.random.rand() < 0.1:
            print("Stopping from id: %d\n" % worder_id)
            # 一旦某个线程调用request_stop函数，所有线程的should_stop函数返回True
            coord.request_stop()
        else:
            print("Working on id: %d\n" % worder_id)
        # 休息1s
        time.sleep(1)


# Coordinator类用来协同多个线程(一起停止)
coord = tf.train.Coordinator()
# 创建5个线程
threads = [threading.Thread(target=MyLoop, args=(coord, i,)) for i in range(5)]
for t in threads:
    t.start()
# 等待所有线程退出
coord.join(threads)
