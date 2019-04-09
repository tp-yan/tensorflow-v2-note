import tensorflow as tf

sess = tf.InteractiveSession()

v1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
v2 = tf.constant([5.0, 4.0, 3.0, 2.0, 1.0])

# tf.greater、tf.where都是元素级的操作
# tf.where：相当于条件表达式 ? :
print(tf.greater(v1, v2).eval())

print(tf.where(tf.greater(v1, v2), v1, v2).eval())

sess.close()