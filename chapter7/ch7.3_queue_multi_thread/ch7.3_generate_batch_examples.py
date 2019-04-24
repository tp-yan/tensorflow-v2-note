import tensorflow as tf
import get_features_from_TFRecord as gft


'''
 将从TFRecord中读取的单个样例组成一个batch的样例集
'''
path_regx = "./data.tfrecords-*"
features = gft.get_features(path_regx)

# 模拟单个样本的 特征向量example，标签label
example, label = features["i"], features["j"]

batch_size = 3
# 组合样例的队列中最多可存样例个数。太大：占内存资源，太小：出队时可能因没有数据而阻塞。一般设置为与 batch_size相关的数
capacity = 1000 + 3 * batch_size
# tf.train.shuffle_batch与tf.train.batch都是将单个样例组织成batch形式的输出，都会生成一个队列，队列的入队操作就是 生成单个样例的方法
# 出队时若队列中元素不够min_after_dequeue，则会等入队元素足够后才完成
# tf.train.shuffle_batch:会将出队元素顺序随机打乱，需要指定其特有的参数min_after_dequeue保证随机打乱效果，而string_input_producer是
# 将文件顺序打乱
example_batch, label_batch = tf.train.shuffle_batch(
    [example, label],  # 不断调用生成单个样例的方法将样本入队，即入队操作
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=30  # 限制出队时队列中元素的最少个数
)

with tf.Session() as sess:
    # 必须要有这行代码，不然无法得到 string_input_producer 函数生成的文件列表，则读不到样本
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)

    coord.request_stop()
    coord.join(threads)

# tf.train.shuffle_batch 与 batch 还提供了并行化处理输入数据的方法，num_threads 参数指定多个线程同时执行入队操作
# 入队操作：就是数据读取以及预处理的过程，若num_threads > 1，则多个线程同时读取一个文件中的不同样例并进行预处理
# tf.train.shuffle_batch_join（tf.train.batch_join）函数：从输入文件队列中获取不同的文件分配给不同的线程，一般输入文件队列由
# string_input_producer函数生成
# tf.train.shuffle_batch：读取前尽量将同一个TFRecord文件中的样例随机打乱
# tf.train.shuffle_batch_join：若线程数多于文件数，则存在多个线程读取同一文件相同数据
