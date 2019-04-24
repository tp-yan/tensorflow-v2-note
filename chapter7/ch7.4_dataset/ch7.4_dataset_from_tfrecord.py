import tensorflow as tf
import parse_txt_tfrecord as tfrp

'''
 每一个TFRecord都有自己不同的Feature格式，在读取TFRecord时，需提供一个parser函数来解析所读取的TFRecord数据
'''

input_files = ["./data.tfrecords-00000-of-00002", "./data.tfrecords-00001-of-00002"]
# TFRecordDataset读取的数据是二进制的
dataset = tf.data.TFRecordDataset(input_files)
# map函数：将数据集中的每一条数据作为传入方法的参数，经过处理后的数据被包装成新的数据集返回
dataset = dataset.map(tfrp.parser)
# 使用one_shot_iterator时，数据集的所有参数必须已确定，且one_shot_iterator无需初始化
iterator = dataset.make_one_shot_iterator()
i, j = iterator.get_next()

with tf.Session() as sess:
    for _ in range(2):
        f1, f2 = sess.run([i, j])
        print(f1, f2)
