import tensorflow as tf

'''
 Inception结构：不同于LeNet5模型(每个卷积层使用的过滤器尺寸一样)，在Inception结构中，每一层卷积层使用不同尺寸的过滤器，
 然后再将不同过滤器的卷积结果在矩阵深度维度上进行拼接
 使用tensorflow-slim工具实现 Inception模块
 tensorflow-slim库：一行代码实现卷积层。因为卷积层操作都是一样的，故可以抽取出公共代码部分，传入指定参数即可
 这里只实现Inception-v3的最后一个Inception模块
'''

# 加载Slim库
slim = tf.contrib.slim
# slim.arg_scope:用于设置函数的默认参数取值，第一个参数是需要设置默认参数值的函数列表，后面的参数，在调用列表中的函数时，自动添加到函数中
# 在调用函数若再设置这些参数的值，则默认值被覆盖
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding="VALID"):
    ...  # 此处省略了Inception-v3从输入层到最后一个Inception模块（第11个Inception模块）之前的所有前向传播过程
    # 这里只实现Inception-v3的最后一个Inception模块
    net = ...  # 假设是最后一个Inception模块的上层输出节点矩阵
    # 为一个Inception模块声明一个统一的变量名命令空间
    with tf.variable_scope("Mixed_7c"):
        # 给Inception模块的每一条路径声明一个命令空间
        with tf.variable_scope("Branch_0"):
            # slim.conv2d(输入节点矩阵net，过滤器深度/个数320，过滤器尺寸[1,1]，scope=变量的命名空间)
            branch_0 = slim.conv2d(net, 320, [1, 1], scope="Conv2d_0a_1x1")
        # 第二条路径，其本身也是一个Inception结构
        with tf.variable_scope("Branch_1"):
            branch_1 = slim.conv2d(net, 384, [1, 1], scope="Conv2d_0a_1x1")
            # tf.concat:将多个矩阵拼接起来，3:指明拼接的维度是矩阵的深度
            branch_1 = tf.concat(3, [
                slim.conv2d(branch_1, 384, [1, 3], scope="Conv2d_0b_1x3"),  # 注意这里的输入是 branch_1而不是net
                slim.conv2d(branch_1, 384, [3, 1], scope="Conv2d_0c_3x1")
            ])
        # 第三条路径,本身也是一个Inception结构
        with tf.variable_scope("Branch_2"):
            branch_2 = slim.conv2d(net, 448, [1, 1], scope="Conv2d_0a_1x1")
            branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope="Conv2d_0b_3x3")
            branch_2 = tf.concat(3, [
                slim.conv2d(branch_2, 384, [1, 3], scope="Con2d_0c_1x3"),
                slim.conv2d(branch_2, 384, [3, 1], scope="Con2d_0d_3x1")
            ])
        # 第四条路径
        with tf.variable_scope("Branch_3"):
            branch_3 = slim.avg_pool2d(net, [3, 3], scope="AvgPool_0a_3x3")
            branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope="Conv2d_0b_1x1")

        # 当前Inception模块的最后输出由上面4个计算结果拼接得到
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
