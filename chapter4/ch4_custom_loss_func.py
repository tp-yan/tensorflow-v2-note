import tensorflow as tf
from numpy.random import RandomState

''''
 单层感知机(即没有隐含层)即使使用非线性激活函数，任然无法解决异或问题。解决：需要加入隐含层，将输入数据映射到更高维，解决更复杂的问题
 多层感知机即使隐含层再多，若不使用非线性激活函数，也无法解决 异或问题(非线性问题)。若不去线性化，再多层的感知机与单层感知机一样，只能解决线性问题。
    解决：采用非线性激活函数！
 根据实际需求实现自定义损失函数
'''

batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# 回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义单层神经网络向前传播过程， 这里是简单的加权求和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义 预测多了和预测少了的成本/代价
loss_less = 10
loss_more = 1
loss = tf.reduce_mean(tf.where(tf.greater(y, y_), loss_more * (y - y_), loss_less * (y_ - y)))  # 自定义损失函数
# AdamOptimizer也是采用的梯度下降算法，同时综合了RMSPro与Momentum两种自适应方法
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 设置回归正确值为2个输入的和加上一个随机量。一般噪声均值为0的小量，这里设置-0.05~0.05的随机数
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
    print(sess.run(w1))
