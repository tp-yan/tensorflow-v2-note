import tensorflow as tf
import parse_txt_tfrecord as tfrp
import ch7_full_img_preprocess as fip

input_files = ["./data.tfrecords-00000-of-00002", "./data.tfrecords-00001-of-00002"]
# TFRecordDataset读取的数据是二进制的
dataset = tf.data.TFRecordDataset(input_files)
# map函数：将数据集中的每一条数据作为传入方法的参数，经过处理后的数据被包装成新的数据集返回
dataset = dataset.map(tfrp.parser)

# 在数据集框架中，无论是 预处理、shuffle、batch等所有操作都是在数据集上进行处理，而在队列框架上预处理是在图像张量上执行，
# shuffle、batch是在队列上完成
# 在数据集上使用map函数 完成 图像预处理工作
image_size = 299
# preprocess_for_train(decoded_img, image_size, image_size, None)
# lambda表达式将 原函数preprocess_for_train的四个参数变为一个参数，其中 x 就是 decoded_img
# 此行代码的意思：将数据集中的每个样本作为参数x，调用 preprocess_for_train函数后，再返回预处理过后的图像
# image_size是变量有具体取值，由上文给出
dataset = dataset.map(lambda x: fip.preprocess_for_train(x, image_size, image_size, None))

# 接着利用数据集实现 shuffle和batch
buffer_size = 30  # 等效于 min_after_dequeue 参数
# shuffle算法：在内部使用一个缓冲区保存 buffer_size 条数据，每读入一条新数据，则从缓冲区中随机选择一条输出
dataset = dataset.shuffle(buffer_size=buffer_size)
batch_size = 100
# 若数据集中每个数据包含多个张量(iterator.get_next()的返回值)，如[image,label]，则batch操作将对每一个张量分开进行
# 假设 image:[300,300],label:[],batch_size = 128. 那么 batch之后，数据集的每一个输出维度分别是 [128,300,300]、[128] 的张量
dataset = dataset.batch(batch_size=batch_size)

# repeat：将数据集中的数据复制N份(每一份成为一个epoch).若在repeat之前已进行shuffle操作，则输出的每一个epoch中随机shuffle结果并不相同
# repeat、map、shuffle、batch都是一个计算节点，其他操作：
# concatenate():将2个数据集顺序连接起来
# take(N):从数据集读取前N项
# skip(N):从数据集中跳过前N项
# flap_map():从多个数据集中轮流读取数据
N = 10
dataset = dataset.repeat(N)