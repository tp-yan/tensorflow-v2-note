import tensorflow as tf

'''
 将二进制形式的 model.ckpt.meta 导出为json格式的可读文件
'''

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()
# export_meta_graph：导出TF计算图的元图数据，以json格式保存
saver.export_meta_graph("model/model.ckpt.meta.json", as_text=True)
