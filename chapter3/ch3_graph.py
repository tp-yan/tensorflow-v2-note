import tensorflow as tf

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([3.0, 4.0], name='b')

# a.graph 获取张量所属的计算图
# tf.get_default_graph()：获取TF系统自动维护的默认计算图
# 张量默认被添加到默认计算图
print(a.graph is tf.get_default_graph())

'''
 不同计算图上的张量和运算不会共享
'''

g1 = tf.Graph()     # 生成新的计算图
with g1.as_default():
    # 在计算图g1中定义变量"v"，并初值为0
    v = tf.get_variable("v",shape=[1],initializer=tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量"v"，并初值为1
    v = tf.get_variable("v",shape=[1],initializer=tf.ones_initializer)

# 在计算图g1中读取变量“v”的值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        # 在计算图g1中读取变量“v”的值应该为0
        print(sess.run(tf.get_variable("v")))

# 在计算图g2中读取变量“v”的值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        # 在计算图g1中读取变量“v”的值应该为1
        print(sess.run(tf.get_variable("v")))