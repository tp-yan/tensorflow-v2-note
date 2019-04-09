import tensorflow as tf


# 为了建模方便，TF会将常量转换为一种永远输出固定值的运算
g = tf.Graph()
with g.as_default():
    a = tf.constant([1.0, 2.0], name='a')
    b = tf.constant([3.0, 4.0], name='b')

# 指定加法计算，在0号gpu上执行
with g.device('/gpu:0'):
    result = a + b

with tf.Session(graph=g) as sess:
    print(result.eval())