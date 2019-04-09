import tensorflow as tf

sess = tf.InteractiveSession()

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(tf.reduce_mean(v).eval())     # reduce_mean:对整个矩阵求均值



'''
交叉熵与softmax回归一起使用：  p 78
tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_)
tf.nn.sparse_softmax_cross_entropy_with_logits  : 若只有一个类别正确性，这个函数可以加速计算过程
实现均方误差函数MSE：
mse = tf.reduce_mean(tf.square(y_ - y)) # P79 

'''

sess.close()