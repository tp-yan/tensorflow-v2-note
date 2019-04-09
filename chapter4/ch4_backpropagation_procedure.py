import tensorflow as tf

'''
 P81-84
 梯度下降算法：主要用于优化单个参数的取值。 核心思想：沿着梯度的反方向让参数朝着总损失更小的方向更新。 缺点：无法保证全局最优，除非损失函数为凸函数
 反向传播算法给出了一个高效的方式在所有参数上使用梯度下降算法
 随机梯度下降：因为梯度下降算法要保证在所有训练样本上保证损失最小，耗时太长，为了加速训练，随机梯度下降算法在每一轮迭代中，随机优化某一条训练数据上的损失函数
            缺点：因为一条样本上损失最小，不代表所有样本上损失最小，有时甚至无法达到局部最优
 实际应用：梯度+随机梯度...算法的折中，在batch上使得总损失最小！即 使用batch的随机梯度下降算法
'''

# 使用TF实现 反向传播算法的一般流程
batch_size = n

# 每次读取一小批数据来执行反向传播算法
x = tf.placeholder(tf.float32, shape=(batch_size, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y-input')

# 定义损失函数，选择某个优化器
loss = ...
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 训练NN
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    STEPS = 10000
    # 迭代执行BP算法
    for i in range(STEPS):
        # 准备batch_size个训练样本。一般打乱后随机选取效果更好
        currentX, currentY = ...
        sess.run(train_step, feed_dict={x: currentX, y_: currentY})
