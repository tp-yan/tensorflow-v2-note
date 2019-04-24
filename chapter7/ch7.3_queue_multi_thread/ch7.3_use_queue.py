import tensorflow as tf

'''
 队列和变量类似，都是计算图上有状态的节点，其他计算节点可修改它们的状态。
 在TF中，队列不仅是一种数据结构，它更提供了多线程机制
'''
# 创建一个最多保存2个元素的先进先出队列，并指定类型
q = tf.FIFOQueue(2, "int32")
# 和变量一样，队列需要初始化：enqueue_many
init = q.enqueue_many(([0, 10],))
# 取出队列第一个元素并赋值给变量x
x = q.dequeue()
y = x + 1
# 入列
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        # 打印出队元素
        print(v)
