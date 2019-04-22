import tensorflow as tf

'''
 从之前保存的TFRecord中读取数据。
 !!TF中与IO相关的操作不用在计算图中执行!!
'''

# 创建一个reader来读取TFRecord文件中的样例
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表:因为数据过大，可以存在多个TFRecord文件中
filename_queue = tf.train.string_input_producer(["./mnist_output.tfrecords"])

# 从文件中读取一个样例。一次性读取多个样例： read_up_to函数
_, serialized_example = reader.read(filename_queue)
# 解析读入的一个样例Example。解析多个样例： parse_example函数
features = tf.parse_single_example(serialized_example, features={
    # TF提供2种不同的属性解析方法：
    # 1. tf.FixedLenFeature：解析结果为一个Tensor
    # 2. tf.VarLenFeature：解析结果为SparseTensor，用于处理稀疏数据
    # 解析数据的格式必须与写入时一致
    'image_raw': tf.FixedLenFeature([], tf.string),
    'pixels': tf.FixedLenFeature([], tf.int64),
    'label': tf.FixedLenFeature([], tf.int64)
})

# tf.decode_raw：将字节字符串解析成图像对应的像素数组
image = tf.decode_raw(features['image_raw'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# 启动多线程处理输入数据
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 每次运行读取TFRecord文件中的一个样例。所有样例读完后，会重头再读
for i in range(10):
    print(sess.run([image, label, pixels]))
