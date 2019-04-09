import tensorflow as tf
import os
import mnist_inference

# 训练神经网络需要用到的参数
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减系数
REGULARIZATION_RATE = 0.0001  # 正则化项的比重，即lambda
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
# 模型保存路径
MODEL_SAVE_PATH = "model/"
MODE_NAME = "model_mnist.ckpt"


def train(mnist):  # 传入封装好的mnist，进行训练
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)  # 使用mnist_inference中定义的前向传播过程
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
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,  # 训练完所有样本需要的次数
                                               LEARNING_RATE_DECAY)
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
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

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
