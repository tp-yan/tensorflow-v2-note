import tensorflow as tf

'''
 通过tf.train.NewCheckpointReader类来查看保存的变量信息
'''

# 可以省略 .data .index
reader = tf.train.NewCheckpointReader("model/model_ema.ckpt")
# 获取所有变量列表。 返回一个 (变量名,变量维度) 的字典
global_variables = reader.get_variable_to_shape_map()
for variable_name in global_variables:
    print(variable_name, global_variables[variable_name])   # global_variables[variable_name]：变量维度
# 获取变量值
print("value for variable v1 is ", reader.get_tensor("v"))
print("value for variable EMA is ", reader.get_tensor("v/ExponentialMovingAverage"))
