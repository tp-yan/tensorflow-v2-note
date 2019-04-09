import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
 在MNIST数据集上实现第4章所提到的功能：指数级衰减学习率、正则化避免过拟合、滑动平均模型增加Robust
 TF提供的input_data工具类将自动下载并转化为MNIST数据格式，将数据从原始的数据包中解析成输入NN的数据格式
 TF提供的minst类对MNIST数据集进行了封装
'''

# MNIST数据集相关的常数
INPUT_NODE = 784  # 输入节点个数。将图片二维像素点矩阵转为一维数组(可看作特征向量)
OUTPUT_NODE = 10

# 配置NN的参数
LAYER1_NODE = 500  # 这里只使用一个隐含层
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减系数
REGULARIZATION_RATE = 0.0001  # 正则化项的比重，即lambda
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 辅助函数，给定NN的输入和所有参数，计算NN的前向传播结果
# 这里定义了3层FNN（一层隐含层）
# 可传入计算参数平均值的类(滑动平均类)：便于在测试时使用滑动平均模型. avg_class:比如之前使用到的ema
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 若没有提供滑动平均类，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐含层前向传播结果，使用ReLU函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 计算输出层的前向传播结果。
        # 因为计算损失函数时会一并计算softmax函数，故这里不需要加入激活函数？？？？
        # 预测时使用的是类别节点相对大小，故可以不适用softmax层
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 使用avg_class.average函数计算变量的滑动平均，来代替变量，再计算前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE), name='y-input')

    # 生成隐含层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下NN前向传播结果
    # 不使用滑动平均类
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 此变量无需计算滑动平均，故指定为不可训练的变量
    # 在训练NN时，一般将训练轮数变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类。传入动态控制衰减率的变量 global_step
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在NN的所有网络参数(weights+biases)上使用滑动平均
    # tf.trainable_variables：返回 TRAINABLE_VARIABLES集合中的元素，该集合自动保存没有指定 trainable=False的Variable
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 使用交叉熵作为损失函数 当只有一个类别为正确答案时，sparse_softmax_cross_entropy_with_logits加快计算过程
    # tf.argmax(y_, 1):得到正确答案对应的类别编号
    # 第一个参数logits=y，是没有softmax层的前向传播结果
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算当前batch中所有样例的平均交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失 = 交叉熵 + 正则项损失
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,  # 训练完所有样本需要的次数
                                               LEARNING_RATE_DECAY)
    # 使用GradientDescentOptimizer优化损失函数。
    # 在minimize中传入 global_step 变量(迭代次数)，global_step将自动更新
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 每过一遍数据，既需反向传播更新模型参数，又需要更新每个参数的滑动平均
    # 为了一次完成多个操作，TF提供了:
    # tf.control_dependencies 与 tf.group机制
    # train_op = tf.group(train_step,variables_averages_op) 等效如下：
    with tf.control_dependencies([train_step, variables_averages_op]):  # 将train_step、variables_averages_op两个操作整合到一起
        train_op = tf.no_op(name='train')

    # 检验使用滑动模型的前向传播结果是否正确
    # average_y、y_： batch*10的二维数组
    # tf.argmax:返回 长度为batch的一维数组。第二个参数1：取最大值的操作仅在第一个维度进行，即只在每一行选取最大值对应的下标
    # tf.equal：判断两个张量的每一维是否相等，元素级操作
    correction_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # tf.cast(correction_prediction,tf.float32):布尔型转为实数型。True->1.0,False->0.0
    accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))  # 这个均值就是模型在这组数据上的正确率

    # 开始训练过程
    with tf.Session() as sess:
        # 初始化所有变量
        tf.global_variables_initializer().run()

        # 准备验证数据。一般NN在训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代训练NN
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据上的结果。因为MNIST数据集小，故这咯没有对验证数据划分为更小的batch
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                # %g：浮点数字，自动根据值的大小采用%e(科学计数法)或%f格式输出
                # 对比验证集和测试集上的正确率，证明：验证集上的测试效果能代表测试集上的效果
                # 选取验证集的关键在于 验证集的分布要尽可能接近测试集分布
                print(
                    "After %d training step(s), validation accuracy using average model is %g, "
                    "test accuracy using average model is %g" % (i, validate_acc, test_acc))
            # 产生这一轮使用的batch数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 训练结束后，在测试数据上检测NN的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g " % (TRAINING_STEPS, test_acc))


# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时，自动下载数据
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)


# TF提供的一个主程序入口
if __name__ == '__main__':
    tf.app.run()  # 会调用上面定义的main函数
