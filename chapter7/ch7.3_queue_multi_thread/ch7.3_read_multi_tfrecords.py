import tensorflow as tf

# 符合正则表达式的文件列表
files = tf.train.match_filenames_once("./data.tfrecords-*")
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
# 这两行是自己添加的代码
# qr = tf.train.QueueRunner(filename_queue, [serialized_example] * 4)
# tf.train.add_queue_runner(qr)

with tf.Session() as sess:
    # 使用 tf.train.match_filenames_once 函数时需要初始化一些变量
    tf.local_variables_initializer().run()
    print(sess.run(files))

    coord = tf.train.Coordinator()
    # 启动默认集合中的queue_runners，默认2个qr：一个负责随机打乱文件顺序以及加入输入队列，一个负责读数据
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(len(threads))  # 具体由几个线程，由其自己决定

    for i in range(10):
        print(sess.run([features["i"], features["j"]]))

    coord.request_stop()
    coord.join(threads)
