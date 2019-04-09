import tensorflow as tf

'''
 此部分负责生成NN参数以及定义网络结构，即前向传播过程
'''

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def generate_weight_variable(shape, regularizer):
    """
    生成权重变量，并正态随机初始化
    :param shape: 
    :param regularizer: 计算正则项的函数 
    :return: 权重变量
    """
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    """
    定义NN前向传播过程
    :param input_tensor: 输入张量
    :param regularizer: 计算权重正则项损失的函数
    :return: 前向传播结果
    """
    with tf.variable_scope("layer1"):
        weights = generate_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer2"):
        weights = generate_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
