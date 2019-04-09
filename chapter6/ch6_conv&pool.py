import tensorflow as tf

'''
 使用TF实现简单的卷积层
'''
input = ...  # 4维数据
# 卷积层参数个数只与卷积核尺寸、深度以及当前层节点矩阵深度有关
# 4维矩阵变量。前2维(5x5)代表卷积核尺寸，第三维3代表当前层的深度，第四维16表示过滤器的深度即个数
filter_weights = tf.get_variable("weights", [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
# shape=[16]，与filter个数是一致的
biases = tf.get_variable("biases", [16], initializer=tf.constant_initializer(0.1))
# conv2d：实现卷积层的前向传播算法。
# 第1个输入参数：当前层的节点矩阵。一个4维矩阵，后三维代表一个节点矩阵，第一维对应一个输入batch，如input[0,:,:,:]代表第一张输入图片
# 第2个输入参数：卷积层所有权重参数
# 第3个输入参数：不同维度上的步长。4维数组，第1和最后一维必须是1，因为卷积层步长只对矩阵的长和宽有效
# 第4个输入参数：填充方式。SAME 或者 VALID(不填充)
conv = tf.nn.conv2d(input, filter_weights, strides=[1, 1, 1, 1], padding="SAME")
# bias_add：给每一个节点加上偏执项，即在过滤器与其对应的3维矩阵节点的点积结果上添加偏置
# 这里不能直接使用加法，因为偏置只有一个数，而点积结果一般是 nxn大小的矩阵
bias = tf.nn.bias_add(conv, biases)
# 使用ReLU激活函数去线性化，卷积层输出
actived_conv = tf.nn.relu(bias)

# TF实现最大池化层：缩小矩阵的尺寸(不改变深度)，减少网络参数，加快计算过程！
# ksize:四维数组。滤波器尺寸，第一个和最后一个必须为1，中间2个数为尺寸长宽，因为池化层的滤波器是二维的，只减小矩阵的长和宽，不会改变矩阵的深度
# strides：同卷积层过滤器一样，第一个和最后一个必须为1，只能在长宽上做步长迁移
# padding同卷积层
pool = tf.nn.max_pool(actived_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
