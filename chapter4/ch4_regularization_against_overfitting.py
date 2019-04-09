import tensorflow as tf

'''
 避免过拟合问题的常用方法：正则化
 正则化：在损失函数中加入刻画模型复杂度的指标R(w)，w：是模型的所有权重参数
 优化目标变成： J(θ)+λR(w)，λ：是模型复杂度在总损失中所占比例
 刻画模型复杂度的函数R(w)的2种实现：
 1. L1正则化，R(w)=||w||₁ = ∑|wi|
 2. L2正则化, R(w)=||w||₂ = ∑|wi|²
 正则化的基本思想：通过限制权重的大小，使模型不能任意拟合训练数据中的随机噪声
'''

'''
# 一种带L2正则项的损失函数
x = ...
y_ = ...
lambda_ = ...
w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w)
loss = tf.reduce_mean(tf.square(y - y_)) + tf.contrib.layers.l2_regularizer(lambda_)(w)
# tf.contrib.layers.l2_regularizer(lambda_)：返回一个函数，该函数计算给定参数的L2正则化项的值
# tf.contrib.layers.l1_regularizer
'''

weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
    # 输出L1正则化的值：5.0
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    # 输出L2正则化的值：7.5 ， TF实现L2正则化时，会除以2，方便求导简洁
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))