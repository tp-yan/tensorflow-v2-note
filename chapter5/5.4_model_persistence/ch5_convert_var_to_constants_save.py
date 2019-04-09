import tensorflow as tf
from tensorflow.python.framework import graph_util

'''
 tf.train.Saver会保存TF程序中的全部信息
 有时只需要知道如何从输入层通过前向传播计算得到输出层即可，而不需要其他节点信息（变量初始化、模型保存节点）
 convert_variables_to_constants函数将变量以常量的方式持久化到计算图中，在一个文件保存计算图结构和变量值
'''

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 导出计算图的GraphDef部分，只需这部分就可完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    # 将变量转为常量保存，同时将图中不必要节点去掉
    # ['add']：给出了需要保存的节点名称，而不是张量add:0，所以与节点add相关的计算节点也会被保存
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    # 保存导出的模型
    with tf.gfile.GFile("model/combined_model.pb", "wb") as f:  # 只生成一个文件
        f.write(output_graph_def.SerializeToString())
