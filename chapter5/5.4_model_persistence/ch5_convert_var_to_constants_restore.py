import tensorflow as tf
from tensorflow.python.platform import gfile

'''
 从model/combined_model.pb中恢复模型(计算节点)
'''

with tf.Session() as sess:
    model_filename = "model/combined_model.pb"
    # 读取模型文件，并解析成对应的 GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 将graph_def中保存的图加载到当前的图中。
    # return_elements=['add:0']：给出了返回的张量的名称
    result = tf.import_graph_def(graph_def, return_elements=['add:0'])
    print(sess.run(result))
