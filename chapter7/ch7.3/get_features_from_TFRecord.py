import tensorflow as tf


def get_features(path_regx):
    # 符合正则表达式的文件列表
    files = tf.train.match_filenames_once(path_regx)
    # shuffle=True:文件在加入队列前打乱顺序
    # string_input_producer生成的输入队列可同时被多个文件读取线程操作，输入队列会将文件均匀地分给不同的线程
    # 当输入队列中的所有文件都会处理完后，会重新在家初始化时提供的文件列表全部重新加入队列
    # num_epochs参数：限制加载初始文件列表的最大轮数。
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        "i": tf.FixedLenFeature([], tf.int64),
        "j": tf.FixedLenFeature([], tf.int64),
    })
    return features