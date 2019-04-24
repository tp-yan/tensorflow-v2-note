import tensorflow as tf

# 从文本文件创建数据集，可以是多个文件
input_file = ["./input_file1.txt", "./input_file2.txt"]
# 文本的每行字符串作为一个训练样本
dataset = tf.data.TextLineDataset(input_file)

iterator = dataset.make_one_shot_iterator()
# 返回一个字符串类型的张量，代表文件中的一行
x = iterator.get_next()
with tf.Session() as sess:
    for i in range(13):
        print(sess.run(x))
