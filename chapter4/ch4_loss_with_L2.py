import tensorflow as tf

'''
 在损失函数中加入L2正则项，避免过拟合
 这里将定义网络结构的部分和计算损失函数的部分分开
'''


# 实现5层全连接网络，且损失函数加入L2正则项

def generate_weight(shape, lambda_):
    '''
    生成一层神经网络边上的权重，并将其对应的L2正则化损失项，加入名为'losses'的集合中
    :param shape: 权重参数的shape，由其关联的两层节点个数决定
    :param lambda_: L2正则化损失项在总损失函数中所占比例
    :return: 返回生成的权重变量
    '''
    weight = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # 将权重的L2正则项损失加入集合'losses'中
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda_)(weight))
    return weight


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# 每层网络中的节点个数
layer_dimension = [2, 10, 10, 10, 1]
# NN的层数
n_layers = len(layer_dimension)
# 向前传播时的张量，开始为输入层
cur_layer = x
# 当前层节点个数
in_dimension = layer_dimension[0]

# 循环生成5层FNN
for i in range(1, n_layers):
    # layer_dimension[i]为下一层节点个数
    out_dimension = layer_dimension[i]
    weight = generate_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)  # 下一层的输入
    # 进入下一层前将下一层的节点个数更新为当前层节点个数
    in_dimension = layer_dimension[i]

# 在训练数据上的损失函数
mse_loss = tf.reduce_mean(tf.square(cur_layer - y_))  # cur_layer：最后为输出层张量
# 将mse_loss也加入‘losses’集合
tf.add_to_collection('losses',mse_loss)
# tf.get_collection：返回一个包含集合元素的list
loss = tf.add_n(tf.get_collection('losses'))    # tf.add_n：将损失函数的不同部分加起来，得到总损失函数
