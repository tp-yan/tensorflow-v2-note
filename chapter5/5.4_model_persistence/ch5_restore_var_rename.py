import tensorflow as tf

'''
 在恢复模型中将变量重命令
'''

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")  # 变量名与原模型不一致
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
result = v1 + v2
# 使用字典来重命名变量，将保存的变量加载到指定变量
saver = tf.train.Saver({"v1": v1, "v2": v2})

with tf.Session() as sess:
    # 加载保存的变量值
    saver.restore(sess, "model/model.ckpt")
    print(sess.run(result))
