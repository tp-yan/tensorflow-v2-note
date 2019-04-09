import tensorflow as tf

sess = tf.InteractiveSession()

v = tf.constant([1.0, 2.0, 3.0])
print(tf.log(v).eval())     # 求每个元素的对数值

sess.close()