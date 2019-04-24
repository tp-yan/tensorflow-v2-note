import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 模拟海量数据下写入多个文件
# TFRecord文件数
num_shards = 2
# 每个TFRecord存的训练样本数
instance_per_shard = 2
for i in range(num_shards):
    # 将文件后缀命名为 0000n-of-0000m 格式区分，m:数据被存在几个文件中，n:当前文件的编号。这样方便通过正则表达式获取文件列表以及更多文件信息
    filename = ("./data.tfrecords-%.5d-of-%.5d" % (i, num_shards))
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instance_per_shard):
        example = tf.train.Example(features=tf.train.Features(feature={
            "i": _int64_feature(i),
            "j": _int64_feature(j)
        }))
        writer.write(example.SerializeToString())
    writer.close()
