import tensorflow as tf

'''
 恢复modelA -- 方式二：直接加载已持久化的图，不需要重复定义图上的运算。
 同时恢复图结构与变量
'''

# 直接加载持久化的图
saver = tf.train.import_meta_graph("model/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "model/model.ckpt")
    # 通过张量的名字来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))  # 即对应result节点
