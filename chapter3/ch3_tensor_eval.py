import tensorflow as tf

'''
 执行运算，都必须在某个会话中完成（会话需要手动指定）！！
 当默认的会话指定后，tensor可通过 tf.Tensor.eval函数计算张量的取值
'''

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([3.0, 4.0], name='b')
result = a + b

# 取Tensor的值
sess = tf.Session()
with sess.as_default():
    '''指定默认会话'''
    print(result.eval())

# 等效代码
print(sess.run(result))
print(result.eval(session=sess))    # 没有指定默认会话而执行运算时，需要显示指定默认会话
sess.close()

# 交互式环境下直接构建默认会话函数tf.InteractiveSession()：自动将生产的会话注册为默认会话
sess = tf.InteractiveSession()
# 在这之间的代码都不用再指定默认会话
print(result.eval())
sess.close()
