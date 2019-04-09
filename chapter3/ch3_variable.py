import tensorflow as tf

'''
在TF中，变量tf.Variable用于保存和更新网络参数
变量声明函数(也是一个运算，返回Tensor)tf.Variable需要给出初始化该变量的方法，初始化值可以是：1. 随机数 2. 常数 3. 由其他变量的初始值计算得到 
变量是一种特殊的Tensor，张量属性：name:一个张量的唯一标识符,shape,type
'''

weights = tf.Variable(tf.random_normal([2, 3], stddev=2))  # 矩阵2x3 均值为0，标准差为2的正态分布随机数
biases = tf.Variable(tf.zeros([3]))  # 长度为3，值为0的一维数组 .在NN中，偏置项bias一般用常数来设置初始值
xw2 = tf.Variable(weights.initialized_value())
xw3 = tf.Variable(weights.initialized_value() * 2.0)  # 由其他变量的 初始值 计算得到

'''
以下实现通过变量实现NN的参数并实现向前传播的过程
'''

# 声明w1,w2两个变量.seed设置随机种子，保证每次运行结果一致
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))    # 底层由 Assign节点/操作实现w1的初始化赋值
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 将输入的特征向量定义为一个常量，1x2的矩阵
x = tf.constant([[0.7, 0.9]])

# 实现向前传播过程，由矩阵乘法实现
a = tf.matmul(x, w1)    # 底层由 read节点/操作读取w1的取，然后输入到matmul节点/操作
y = tf.matmul(a, w2)

with tf.Session() as sess:  # 上下文管理器只是可以方便资源管理，但并没有将此sess设置为默认的会话！！！
    '''
    在执行运算之前，需要将所有涉及的变量必须都进行初始化.
    变量初始化必须显示调用！！
    '''
    sess.run(w1.initializer)  # 分别执行w1w2的初始化过程
    sess.run(w2.initializer)
    # 一种快捷方法tf.global_variables_initializer():自动实现初始化所有变量的过程，会自动处理变量间的依赖关系
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(y))
