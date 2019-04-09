import tensorflow as tf

'''
 会话session: 
    1. 执行运算
    2. 拥有并管理TF程序运行时的所有资源
    3. 计算完成后，关闭session以帮助系统回收资源，否则出现资源泄露
'''

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([3.0, 4.0], name='b')
result = a + b

# 创建会话的第一种方式：显示创建和关闭会话
sess = tf.Session()
sess.run(result)  # 执行result运算，获取计算结果
sess.close()    # 关闭会话，释放资源。 必须显示关闭会话！！！

# 使用Python的上下文管理器：不用担心异常退出，而未关闭会话造成资源泄露
with tf.Session() as sess:
    sess.run(result)

