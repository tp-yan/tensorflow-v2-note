import tensorflow as tf

'''
使用ConfigProto Protocol Buffer配置会话：配置并行的线程数、GPU分配策略、运算超时时间等参数 
allow_soft_placement：True:若某些运算无法在GPU上执行（如运算输入包含对CPU计算结果的引用），则自动调整到CPU上
log_device_placement：True:日志中将会记录每个节点被安排在哪个设备上以方便调试，生成环境中一般为False：减少日志量 
'''

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# 无论哪种会话生成方式，都可以指定config参数
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
