import tensorflow as tf

'''
 tf.get_variable与tf.Variable 创建变量功能是一样的
 get_variable：创建变量时，必须指定名称。首先根据名字去创建变量，若已存在同名变量，则报错
 Variable：名字可选
'''
# tf.constant_initializer与tf.constant常数生成函数基本是一一对应的
# 不使用tf.variable_scope函数，tf.get_variable创建的变量，属于默认的空命名空间
v1 = tf.get_variable("v1", shape=[1], initializer=tf.constant_initializer(1.0))
v2 = tf.Variable(tf.constant(1.0, shape=[1]), name='v2')
print(v1.name)  # v1:0 ：命令空间为空. 0表示生产变量这个运算的第一个结果
print(v2.name)  # v2:0

'''
 tf.variable_scope：变量命名空间管理器，搭配 tf.get_variable 来创建和获取变量
 1. 只能获取tf.get_variable创建的变量，不能获取tf.Variable创建的变量
 2. reuse=True：指定 tf.get_variable 为获取已有变量，若不存在则报错
 3. reuse=False：指定 tf.get_variable 为创建新变量，若存在则报错
 4. tf.variable_scope内创建的变量，属于指定的命名空间，变量全名前缀带命名空间
'''
with tf.variable_scope(""):
    vv = tf.get_variable("vv", shape=[1], initializer=tf.zeros_initializer)
    print(vv.name)

with tf.variable_scope("", reuse=True):
    vy = tf.get_variable("v1", shape=[1])
    # vx = tf.get_variable("v2",shape=[1])    # 不能获取tf.Variable创建的变量,报错
    print(v1 == vy)  # True . 说明 v1默认属于空命名空间

# 在foo命名空间内创建名为‘v’的变量
with tf.variable_scope("foo"):  # reuse默认为False
    vt = tf.Variable(tf.constant(1.0, shape=[1]), name='vt')
    print(vt.name)  # foo/vt:0 , 但是仍然无法 通过 variable_scope 获取 Variable创建的变量
    f_v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))

with tf.variable_scope("foo", reuse=True):
    f_v1 = tf.get_variable("v", [1])
    print(f_v1 == f_v)  # True
    print(f_v1.name)

# tf.variable_scope 嵌套使用
with tf.variable_scope("root"):
    print(tf.get_variable_scope())  # tensorflow.python.ops.variable_scope.VariableScope
    print(tf.get_variable_scope().reuse)  # 获取上下文管理器的reuse参数值
    with tf.variable_scope("foo", reuse=True):
        print(tf.get_variable_scope().reuse)
        with tf.variable_scope("bar"):  # 当嵌套的 variable_scope 没有指定reuse时，其值为外层的reuse值
            print(tf.get_variable_scope().reuse)
    print(tf.get_variable_scope().reuse)

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print(v3.name)

    v4 = tf.get_variable("v1", [1])
    print(v4.name)

with tf.variable_scope("", reuse=True):
    v5 = tf.get_variable("foo/bar/v", [1])  # 直接通过带命名空间的变量名获取其他命名空间的变量
    print(v5 == v3)  # <tf.Variable 'foo/bar/v:0' shape=(1,) dtype=float32_ref>
    print(v5)
    v6 = tf.get_variable("foo/v1", [1])
    print(v6 == v4)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(v1))
    print(sess.run(v2))

INPUT_NODE = 2
LAYER1_NODE = 10
OUTPUT_NODE = 1


# 使用 tf.variable_scope tf.get_variable 管理变量，去除变量参数的传递
def inference(input_tensor, reuse=False):
    # 定义第一层的网络变量与前向传播结果
    with tf.variable_scope("layer1", reuse=reuse):
        weights = tf.get_variable("weights", shape=[INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 定义第二层的网络变量与前向传播结果
    with tf.variable_scope("layer2", reuse=reuse):
        weights = tf.get_variable("weights", shape=[LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2


x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name='x-input')
y = inference(x)  # 第一次调用inference，则创建NN结构和变量参数

# 使用训练好的NN进行推导，直接调用inference(new_x,True)
new_x = ...
new_y = inference(new_x, True)
