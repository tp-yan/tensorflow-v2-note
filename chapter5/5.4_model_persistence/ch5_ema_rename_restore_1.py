import tensorflow as tf

''''
 通过变量重命名直接读取变量的滑动平均值--方式一
'''

v = tf.Variable(0, dtype=tf.float32, name="v")
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "model/model_ema.ckpt")
    print(sess.run(v))  # 影子变量的值：0.099999905
