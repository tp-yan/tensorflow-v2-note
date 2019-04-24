import tensorflow as tf
import parse_txt_tfrecord as tfrp

'''
 若使用placeholder来初始化数据集，需要用到 initializable_iterator：动态初始化数据集
 另外还有：
 reinitializable_iterator:可以多次initialize用于遍历不同的数据源
 feedable_iterator:动态指定运行哪个iterator
'''
# 从TFRecord文件创建数据集，具体路径由placeholder占位符代替
input_files = tf.placeholder(tf.string)
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(tfrp.parser)

iterator = dataset.make_initializable_iterator()
feat1, feat2 = iterator.get_next()

with tf.Session() as sess:
    # 初始化Iterator
    sess.run(iterator.initializer,
             feed_dict={input_files: ["./data.tfrecords-00000-of-00002", "./data.tfrecords-00001-of-00002"]})

    # 一般动态输入的数据量大小未知，故常用这种方式读取数据
    # 遍历所有数据一个Epoch，遍历结束后 抛出 OutOfRangeError
    while True:
        try:
            f1, f2 = sess.run([feat1, feat2])
            print([f1, f2])
        except tf.errors.OutOfRangeError:   # 抛出OutOfRangeError说明所有数据都遍历一遍了
            break
