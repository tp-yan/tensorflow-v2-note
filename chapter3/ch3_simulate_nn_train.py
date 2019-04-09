import tensorflow as tf
from numpy.random import RandomState

'''
 模拟训练神经网络，解决二分类问题
 使用numpy工具包的RandomState 生成模拟数据集
'''

batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

'''
 在shape的一个维度上使用None，可以适应不同大小的batch
 比如在训练时batch一般较小，而在测试时可以一次性使用全部数据
'''
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

y = tf.sigmoid(y)
# y代表预测是正样本的概率，1-y代表预测是负样本的概率。因为这里没有使用 softmax 层，所以考虑2个类别(正/负)的概率
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# x1+x2 < 1的为正样本1，其他为负样本0
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("\ninitial weights:\n")
    print(sess.run(w1))
    print(sess.run(w2))

    # 训练轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次取batch个样本来训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            # 每隔一段时间，计算所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X[start:end], y_: Y[start:end]})
            print("After %d training step(s), cross entropy on all data is %g" % (i,total_cross_entropy))
    print("\nAfter training:\n")
    print(sess.run(w1))
    print(sess.run(w2))