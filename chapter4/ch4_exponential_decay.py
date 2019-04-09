import tensorflow as tf

'''
 使用指数衰减学习率方法，使前期加快模型收敛过程，后期模型学习更稳定
 指数衰减学习率实现：
 decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
 decayed_learning_rate：每一轮优化时，使用的学习率
 learning_rate：初始学习率
 decay_rate：衰减系数。一般是固定值，小于1
 global_step：一个变量，即当前训练的轮数
 decay_steps：衰减速度。一般是固定值
'''

# 使用TF实现 指数衰减学习率
global_step = tf.Variable(0)

# 通过tf.train.exponential_decay函数实现指数衰减学习率
learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=100, decay_rate=0.96,
                                           staircase=True)
# 若staircase=True，则global_step/decay_steps转为整数，学习率成为一个阶梯函数。 在这样的设置下，... --P86
# 若staircase=False，则学习率表现为一个连续下降函数

# 使用指数衰减学习率
# 在minimize函数中传入global_step将自动更新，从而使学习率也相应更新
my_loss = ...
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(my_loss, global_step=global_step)
