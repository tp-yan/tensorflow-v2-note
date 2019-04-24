import tensorflow as tf
import parse_img_tfrecord as img_parser
import ch7_full_img_preprocess as img_reprocessor

'''
 使用Dataset框架完成 ch7.3中的数据输入流程：
 1.从文件中读取原始数据 --> 2.进行预处理、shuffle、batch等操作 --> 3.通过repeat训练多个epoch --> 4.读取测试数据集
'''


def inference(image_batch):
    return None


def calc_loss(logit, label_batch):
    return None


train_files_path = "./train_file-*"
test_files_path = "./test_file-*"
train_files = tf.train.match_filenames_once(train_files_path)
test_files = tf.train.match_filenames_once(test_files_path)

# 输入层图像大小
image_size = 299
batch_size = 100
shuffle_buffer = 10000

dataset = tf.data.TFRecordDataset(train_files)
# 解析二进制数据，map返回的每条数据是(decoded_img,label)两个张量
dataset = dataset.map(img_parser.parser)

dataset = dataset.map(
    lambda image, label:  # image, label:是上一个map返回的结果，作为 lambda表达式的参数，返回经过预处理的image及其label
    (img_reprocessor.preprocess_for_train(image, image_size, image_size, None), label))
dataset = dataset.shuffle(shuffle_buffer).batch(batch_size=batch_size)

NUM_EPOCHS = 10
dataset = dataset.repeat(NUM_EPOCHS)

# 因为tf.train.match_filenames_once得到的结果与 placeholder类似，故也许初始化，也需要使用initializable_iterator
iterator = dataset.make_initializable_iterator()
image_batch, label_batch = iterator.get_next()

learning_rate = 0.01
logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 测试数据的Dataset不需要预处理、打乱顺序、重复多个epoch，只需调整图像大小，直接进行batch操作
test_dataset = tf.data.TFRecordDataset(test_files)
test_dataset = test_dataset.map(img_parser.parser).map(
    lambda image, label: (tf.image.resize_images(image, [image_size, image_size]), label))
test_dataset = test_dataset.batch(batch_size)

test_iterator = test_dataset.make_initializable_iterator()
test_image_batch, test_label_batch = test_iterator.get_next()
test_logit = inference(test_image_batch)
predictions = tf.argmax(test_logit, axis=-1, output_type=tf.int32)

with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    sess.run(iterator.initializer)

    while True:
        try:
            sess.run(train_step)
        except tf.errors.OutOfRangeError:
            break

    # 测试部分
    sess.run(test_iterator.initializer)
    # 每个元素对应一个预测/正确标签
    test_results = []
    test_labels = []
    while True:
        try:
            pred, label = sess.run([predictions, test_label_batch])
            test_results.extend(pred)
            test_labels.extend(label)
        except tf.errors.OutOfRangeError:
            break

    # 计算准确率
    # zip:将可迭代参数的元素依次组成一个tuple返回
    correct = [float(y == y_) for (y, y_) in zip(test_results, test_labels)]
    accuracy = sum(correct) / len(correct)
    print("Test accuracy is:", accuracy)
