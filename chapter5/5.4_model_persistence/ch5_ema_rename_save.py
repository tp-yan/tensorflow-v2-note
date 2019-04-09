import tensorflow as tf

'''
 重命名变量的主要目的之一：方便使用变量的滑动平均(影子变量)
 加载模型时将影子变量映射到变量本身，使用模型时就不需要调用函数取变量的滑动平均
'''

v = tf.Variable(0, dtype=tf.float32, name="v")
# 在没有声明滑动平均模型时，只有一个变量v
for variables in tf.global_variables():
    print(variables)

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.global_variables())
# 声明滑动平均模型后，TF会自动生成一个影子变量
for variables in tf.global_variables():
    print(variables)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))
    sess.run(maintain_average_op)
    saver.save(sess, "model/model_ema.ckpt")  # TF会将v和v的影子变量一起保存
    print(sess.run([v, ema.average(v)]))  # [10.0, 0.099999905]
