import tensorflow as tf

'''
 保存一个简单的TF计算图模型modelA
'''

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
# tf.train.Saver()类用于持久化模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)  # 先对变量初始化
    # 将模型保存到 model/model.ckpt 文件   .ckpt：checkpoint,TF模型文件
    saver.save(sess, "model/model.ckpt")  # 与当前文件的同级目录
    # 会在目录下生成4个文件：
    # 1. model.ckpt.meta ：计算图结构，也就是神经网络结构
    # 2. model.ckpt.data: SSTable格式存储的，(key,value)列表形式
    # 3. model.ckpt.index：与 model.ckpt.data 一起保存了TF程序中的所有变量取值
    # 4. checkpoint文件：固定名字，由Saver类自动生成与维护。保存了一个目录下最新的模型文件列表
