import tensorflow as tf

'''
 在Dataset框架中，每一个数据源被抽象为一个“数据集”，数据集可看成一个基本对象，可对其进行batching、shuffle等操作。
 Dataset是比 队列 更高层的数据处理框架，且是TF推荐的输入数据的首选框架。
 
 每一个数据集的来源可以是：一个张量、一个TFRecord文件、一个文本文件或者经过sharding的一系列文件等等。
 往往训练集太大无法全部读入内存，从数据集中读取数据时需要使用一个迭代器按顺序读取与 队列的dequeue()类似
 同时，数据集也是计算图上的一个节点。
 
 利用数据集读取数据的三个基本步骤：
 1.定义数据集的构造方法
 2.定义遍历器
 3.使用 get_next()方法从遍历器中读取数据张量，作为计算图其他部分的输入
'''

# 下面从 一个张量创建一个数据集
input_data = [1, 2, 3, 5, 8]
dataset = tf.data.Dataset.from_tensor_slices(input_data)

# 定义iterator遍历数据集，因为不是placeholder，故使用最简单的one_shot_iterator
iterator = dataset.make_one_shot_iterator()
# 返回代表一个输入数据的张量，类似于队列的 dequeue
x = iterator.get_next()
y = x * x
with tf.Session() as sess:
    for i in range(len(input_data)):
        print(sess.run(y))
