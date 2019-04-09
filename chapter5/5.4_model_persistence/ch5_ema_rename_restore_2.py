import tensorflow as tf

''''
 通过变量重命名直接读取变量的滑动平均值--方式二
 tf.train.ExponentialMovingAverage提供的variables_to_restore函数方便加载时重命名滑动平均模型变量
'''

v = tf.Variable(0, dtype=tf.float32, name="v")  # 变量名必须与原模型一样
ema = tf.train.ExponentialMovingAverage(0.99)
# ema.variables_to_restore()可以直接生成字典；{"v/ExponentialMovingAverage": v}
# <tf.Variable 'v:0' shape=() dtype=float32_ref>：代表了变量v
print(ema.variables_to_restore())

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, "model/model_ema.ckpt")
    print(sess.run(v))  # 影子变量的值：0.099999905
