import tensorflow as tf

"""
 在tensorflow代码中 * / + - 都是元素级的操作
 这里比较 矩阵元素间乘法与 矩阵乘法
"""

sess = tf.InteractiveSession()
v1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
v2 = tf.constant([[6.0, 7.0], [8.0, 9.0]])

print((v1*v2).eval())   # 矩阵元素间相乘
print(tf.matmul(v1,v2).eval())  # 矩阵相乘

sess.close()