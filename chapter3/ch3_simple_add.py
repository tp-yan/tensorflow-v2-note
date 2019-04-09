import tensorflow as tf

# 为了建模方便，TF会将常量转换为一种永远输出固定值的运算
a = tf.constant([1.0, 2.0], name='a')   # tf.constant也是一个计算. a是一个Tensor，其保存了计算过程，是对计算结果的引用
b = tf.constant([3.0, 4.0], name='b')
result = a + b
print(a)    # Tensor("a:0", shape=(2,), dtype=float32)
print(result)  # Tensor("add:0", shape=(2,), dtype=float32)

sess = tf.InteractiveSession()

print(result.eval())  # 执行计算节点，并取该节点上的运算结果

sess.close()
