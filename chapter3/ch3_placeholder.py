import tensorflow as tf

'''
 placeholder机制用于提供输入数据到计算图（神经网络）中，其相当于定义了一个位置。
 placeholder数据的类型必须指定，而数据的维度信息可以不指定，通过传入的实际数据推导出
'''

# 通过placeholder实现向前传播算法
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义placeholder作为存放输入数据的地方
x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

# 在执行运算节点时，再通过 feed_dict 传入数据
print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))  # [[3.957578]]

sess.close()

'''
 提供一个batch的训练样例
'''
x1 = tf.placeholder(tf.float32, shape=(3, 2), name="input1")    #  可以将设置 shape=(None, 2)：适应不同大小的batch
a1 = tf.matmul(x1, w1)
y1 = tf.matmul(a1, w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(y1, feed_dict={x1: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))  # 提供一个batch的训练数据
'''
[[3.957578 ]
 [1.1537654]
 [3.1674924]]
'''
# 使用TF实现损失函数+反向传播算法
# tf.sigmoid:将y1转换为0~1之间的数值。转换后 y1代表预测是正样本的概率，1-y1代表预测是负样本的概率
y1 = tf.sigmoid(y1)
y_ = tf.constant([[1], [0], [1]])    # 自己手动设置标签
# 定义损失函数
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y1, 1e-10, 1.0) + (1 - y_) * tf.log(tf.clip_by_value(1 - y1, 1e-10, 1.0))))
learning_rate = 0.001
# 定义反向传播算法训练/优化NN的参数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
''' 
其他常用优化器(不同的优化算法)：
    tf.train.GradientDescentOptimizer
    tf.train.AdamOptimizer
    tf.train.MomentumOptimizer
'''
sess.close()
