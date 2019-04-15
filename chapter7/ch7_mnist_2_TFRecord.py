import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
 TFRecord格式可以统一不同的原始数据，TF提供的一种统一格式来存储数据。可以有效记录输入数据的更多许多信息
'''


# 实现将mnist原始数据转为TFRecord格式存储

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成(字节)字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets("../datasets/MNIST_data", dtype=tf.uint8, one_hot=True)
images = mnist.train.images
# 一维数据：[0,1,..,0],将作为一个属性保存在TFRecord中
labels = mnist.train.labels
# 训练数据的图像分辨率，也作为Example中的一个属性
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出TFRecord文件路径
filename = "./mnist_output.tfrecords"
# 创建一个writer来写 TFRecord文件
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    # 将图像矩阵转为一个字符串
    image_raw = images[index].tostring()    # b'\x00\x00\x00\x00\x00\x00\...
    # if index % 10000 == 0:
    #     print(index, ":\n", image_raw, "\n")
    # 将一个样例转为Example Protocol Buffer ，并将所有的信息写入这个数据结构
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)
    }))

    # 将一个Example写入TFRecord文件中
    writer.write(example.SerializeToString())
writer.close()
