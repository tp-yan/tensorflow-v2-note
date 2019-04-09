import tensorflow as tf

'''
 恢复modelA -- 方式一：所有重新定义计算图上的所有运算后，再加载模型变量的值
 即重新构建原来的模型结构，只是将保存的变量值，重新恢复
'''
# 定义计算图中的运算：必须与原模型一模一样
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()
# 只加载或者保存部分变量
# saver = tf.train.Saver([v1])  # 只加载v1变量


with tf.Session() as sess:
    # 加载保存的变量值
    saver.restore(sess,"model/model.ckpt")
    print(sess.run(result))
