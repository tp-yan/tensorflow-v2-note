import tensorflow as tf

'''
 滑动平均模型：使模型在测试数据上更健壮。    原理：将每一轮迭代得到的模型综合起来，从而更robust
 前提：采用随机梯度下降算法训练NN时，使用滑动平均模型，可提高在测试数据上的表现
 ExponentialMovingAverage会对每一个变量维护一个影子变量/滑动平均，每次变量更新时，影子变量也需要进行更新
'''

# 定义一个变量用于计算其 滑动平均(即影子变量)。注：需要计算滑动平均的变量必须是实数型
v1 = tf.Variable(0,dtype=tf.float32)
# 迭代次数，用于动态控制衰减率
step = tf.Variable(0,trainable=False)

# 定义一个滑动平均类(class)，初始化时需给定衰减率(0.99)和控制衰减率的变量(step)
ema = tf.train.ExponentialMovingAverage(0.99,step)
# 定义更新变量的影子变量/滑动平均的操作。需要给定一个列表，每次执行此操作时，列表中变量对应的滑动平均都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 通过ema.average(v1)获得v1的滑动平均。初始化后v1及其滑动平均都为0
    print(sess.run([v1, ema.average(v1)]))
    # 更新变量v1
    sess.run(tf.assign(v1,5))
    # 相应的其滑动平均也需要手动更新
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新step
    sess.run(tf.assign(step,10000))     # 使得衰减率与前面不同
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))
    # 再次更新滑动平均(在上次滑动平均与当前v1的值上更新)
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))
