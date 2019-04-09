import tensorflow as tf

'''
Tensor的数据类型定了就不能再改变，而维度(shape)可以再改变
'''

# 上下文管理器只是可以方便资源管理，但并没有将此sess设置为默认的会话！！！
with tf.Session() as sess:
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1), name="w1")
    w2 = tf.Variable(tf.random_normal([2, 2], stddev=1), name="w2")
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("w1:\n", w1.eval(session=sess))
    print("w2:\n", w2.eval(session=sess))
    sess.run(tf.assign(w1, w2, validate_shape=False))   # 设置参数validate_shape=False，使张量维度在程序运行中也可以改变！
    # tf.assign(w1,w2)  # 这句代码会报错
    print("after assign, w1:\n", w1.eval(session=sess)) # w1变成 2x2的矩阵
