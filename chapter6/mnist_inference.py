import tensorflow as tf

'''
 使用CNN重写前向传播过程，模型类似LeNet-5模型(7c层)，但这里只有6层，以及其他微小差异
'''

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28  # 输入原始图像尺寸
NUM_CHANNELS = 1  # LeNet-5中使用的是3通道RGB，而TF封装的MNIST是黑白1通道的
NUM_LABELS = 10

# 第一层卷积层的尺寸和个数(深度)
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第一层卷积层的尺寸和个数(深度)
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层节点个数
FC_SIZE = 512


# 定义CNN的前向传播过程
def inference(input_tensor, train, regularizer):
    """
    :param input_tensor: batchx28x28x1 的MNIST原始像素，四维矩阵
    :param train: 用于区分训练还是测试过程
    :param regularizer: 计算权重正则项损失的函数
    :return:  CNN前向传播结果
    """
    # 声明第一层卷积层的变量，并实现前向传播
    # 卷积层输入为28x28x1的MNIST原始像素，因为使用全0填充，输出28x28x32的矩阵
    with tf.variable_scope("layer1-conv1"):
        # 先声明卷积层变量
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用5x5x32的过滤器，步长为1，全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层的最大池化层。输入为上层的输出：28x28x32，输出：14x14x32
    # 滤波器尺寸2x2，全0填充，步长2
    with tf.variable_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 第三层卷积层,输入：14x14x32，输出：14x14x64
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weights", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层最大池化层的前向传播过程,输入：14x14x64，输出：7x7x64。
    with tf.variable_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 将第四层输出转换为第五层全连接层的输入格式：将三维矩阵拉长一维向量
        # 第四层输出：7x7x64=3136 --> 第五层输入：一维向量
        # 注：每一层NN的输入输出都为一个batch的矩阵，故pool2.get_shape()得到的维度也包含batch中数据个数
        pool_shape = pool2.get_shape().as_list()

        # 计算第五层输入层的节点个数，一维向量长度 = 矩阵长宽以及深度的乘积
        # pool_shape[0]：为一个batch中数据个数
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

        # 使用tf.reshape函数将第四层输出变为一个batch的向量
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # dropout一般只在全连接层而不是卷积层和池化层使用
    # 声明第五层全连接层的变量并实现前向传播过程
    with tf.variable_scope("layer5-fc1"):
        fc1_weights = tf.get_variable("weights", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc1_weights))
        fc1_biases = tf.get_variable("biases", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # 只有在训练时才dropout
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明第六层全连接层的变量并实现前向传播过程
    with tf.variable_scope("layer6-fc2"):
        fc2_weights = tf.get_variable("weights", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weights))
        fc2_biases = tf.get_variable("biases", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit  # 返回第六层输出，没有softmax层
