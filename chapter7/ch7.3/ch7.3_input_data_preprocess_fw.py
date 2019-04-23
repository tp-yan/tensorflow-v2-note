import tensorflow as tf
import ch7_full_img_preprocess as fip

'''
 这里完成 输入数据处理框架 并非可执行代码
'''


def inference(image_batch):
    return None


def calc_loss(logit, label_batch):
    return None


file_path = "./file_pattern-*"  # 假设都是使用TFRecord格式存储的数据
# 匹配符合的文件列表
files = tf.train.match_filenames_once(file_path)
# 生成输入文件队列
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
# 依次从输入文件队列的文件中逐个读出 样本
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                   features={
                                       "image": tf.FixedLenFeature([], tf.string),
                                       "label": tf.FixedLenFeature([], tf.int64),
                                       "height": tf.FixedLenFeature([], tf.int64),
                                       "width": tf.FixedLenFeature([], tf.int64),
                                       "channels": tf.FixedLenFeature([], tf.int64),
                                   })
image, label = features["image"], features["label"]
height, width = features["height"], features["width"]
channels = features["channels"]

# 将原始数据解析为像素矩阵
decoded_img = tf.decode_raw(image, tf.uint8)
# 还原图像尺寸
decoded_img.set_shape([height, width, channels])
# 输入层图像大小
image_size = 299

# 图像预处理
distorted_img = fip.preprocess_for_train(decoded_img, image_size, image_size, None)

min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
# 由shuffle_batch决定是否多线程读取 [image, label]
image_batch, label_batch = tf.train.shuffle_batch([distorted_img, label], batch_size=batch_size, capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue)

#
learning_rate = 0.01
logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    TRAINING_ROUNDS = 5000
    for i in range(TRAINING_ROUNDS):
        sess.run(train_op)

    coord.request_stop()
    coord.join(threads)
