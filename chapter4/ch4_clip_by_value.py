import tensorflow as tf


sess = tf.InteractiveSession()

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(tf.clip_by_value(v,2.5,4.5).eval())   # clip_by_value:限定元素范围，截断值
# [[2.5 2.5 3. ]
#  [4.  4.5 4.5]]

sess.close()