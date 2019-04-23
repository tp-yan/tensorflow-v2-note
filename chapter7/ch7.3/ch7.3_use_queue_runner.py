import tensorflow as tf

'''
 QueueRunner:主要用于启动多个线程来操作同一个队列
'''
# 最多100个元素的先进先出队列
queue = tf.FIFOQueue(100, "float")
# 定义入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])
# [enqueue_op] * 5:启动5个线程，都是完成enqueue_op操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
# 将QueueRunner加入默认的 GraphKeys.QUEUE_RUNNERS 集合
tf.train.add_queue_runner(qr)
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # coord协同启动与停止线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner定义的所有线程，默认启动 GraphKeys.QUEUE_RUNNERS 集合中的线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(3):
        print(sess.run(out_tensor)[0])
    # 停止coord管理的所有线程
    coord.request_stop()
    coord.join(threads)
