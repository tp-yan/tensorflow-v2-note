import numpy as np

import tensorflow as tf
import os
import mnist_inference

'''
 修改第五章中，全连接网络训练MNIST数据集的代码，这里用卷积神经网络(类似LeNet-5模型)实现
'''
# 训练神经网络需要用到的参数
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
# LEARNING_RATE_BASE = 0.8  # 初始学习率
# change
LEARNING_RATE_BASE = 0.01  # 初始学习率。收敛更快
LEARNING_RATE_DECAY = 0.99  # 学习率衰减系数
REGULARIZATION_RATE = 0.0001  # 正则化项的比重，即lambda
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

MODEL_SAVE_PATH = "model/"  # 模型保存路径
MODE_NAME = "model_mnist_LeNet5.ckpt"


def train(mnist):
    # 损失函数的计算和反向传播过程的实现都复用之前的原代码
    # 这里需要修改卷积神经网络的输入数据尺寸：CNN输入是一个三维矩阵，而非之前的一维向量
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,  # 一个batch的样例个数 ，这里没使用None，是因为后面的 tf.reshape时需要具体batch的大小
        # 后三维代表了一个样例的数据
        mnist_inference.IMAGE_SIZE,
        mnist_inference.IMAGE_SIZE,  # 输入图片尺寸
        mnist_inference.NUM_CHANNELS],  # 图片深度，即颜色通道个数
                       name='x-input')
    # change
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')  # None or BATCH_SIZE 没有影响
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # change
    y = mnist_inference.inference(x, False, regularizer)  # 使用mnist_inference中定义的前向传播过程，使用了dropout反而收敛变慢，损失更大？？？
    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类。传入动态控制衰减率的变量 global_step
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在NN的所有网络参数(weights+biases)上使用滑动平均
    # tf.trainable_variables：返回 TRAINABLE_VARIABLES集合中的元素，该集合自动保存没有指定 trainable=False的Variable
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算当前batch中所有样例的平均交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,  # 训练完所有样本需要的次数
        # change
        LEARNING_RATE_DECAY, staircase=True)  # staircase=True（效果更优） or False 对结果影响不大
    # 在minimize中传入 global_step 变量(迭代次数)，global_step将自动更新
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # train_op = tf.group(train_step,variables_averages_op) 等效如下：
    with tf.control_dependencies([train_step, variables_averages_op]):  # 将train_step、variables_averages_op两个操作整合到一起
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 同理输入的训练数据格式也要调整为四维矩阵,再传入sess.run过程
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前模型在当前batch上的损失函数大小以估计训练效果。在验证数据集上的正确率信息由另一个单独程序生成
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # global_step使模型文件名后面如： model.ckpt-1000 添加训练的轮数
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODE_NAME), global_step=global_step)


# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时，自动下载数据
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)


# TF提供的一个主程序入口
if __name__ == '__main__':
    tf.app.run()  # 会调用上面定义的main函数

'''
 训练记录：
After 1 training step(s), loss on training batch is 4.5159.
After 1001 training step(s), loss on training batch is 0.7963.
After 2001 training step(s), loss on training batch is 0.675641.
After 3001 training step(s), loss on training batch is 0.681144.
After 4001 training step(s), loss on training batch is 0.765745.
After 5001 training step(s), loss on training batch is 0.634894.
After 6001 training step(s), loss on training batch is 0.667482.
After 7001 training step(s), loss on training batch is 0.709603.
After 8001 training step(s), loss on training batch is 0.622255.
After 9001 training step(s), loss on training batch is 0.639394.
After 10001 training step(s), loss on training batch is 0.64507.
After 11001 training step(s), loss on training batch is 0.631707.
After 12001 training step(s), loss on training batch is 0.620483.
After 13001 training step(s), loss on training batch is 0.614534.
After 14001 training step(s), loss on training batch is 0.633172.
After 15001 training step(s), loss on training batch is 0.652686.
After 16001 training step(s), loss on training batch is 0.648175.
After 17001 training step(s), loss on training batch is 0.611909.
After 18001 training step(s), loss on training batch is 0.661383.
After 19001 training step(s), loss on training batch is 0.609987.
After 20001 training step(s), loss on training batch is 0.60562.
After 21001 training step(s), loss on training batch is 0.799953.
After 22001 training step(s), loss on training batch is 0.614792.
After 23001 training step(s), loss on training batch is 0.60752.
After 24001 training step(s), loss on training batch is 0.627114.
After 25001 training step(s), loss on training batch is 0.609718.
After 26001 training step(s), loss on training batch is 0.601168.
After 27001 training step(s), loss on training batch is 0.606314.
After 28001 training step(s), loss on training batch is 0.599224.
After 29001 training step(s), loss on training batch is 0.605827.
'''
