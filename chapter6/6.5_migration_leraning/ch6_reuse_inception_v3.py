import tensorflow as tf
import os.path
import glob
import numpy as np
import tensorflow.contrib.slim as slim
# 加载通过TensorFlow-Slim定义好的inception_v3模型。
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

'''
 迁移已训练好的Inception-v3模型，重新训练全连接层的参数，所有其他参数都复用已训练好的模型参数
'''

# 处理好的图像数据文件
INPUT_DATA = "./flower_processed_data.npy"
# 保存重新训练好的模型
# TRAIN_FILE = "./save_model/model"
TRAIN_FILE = "./train_model/model"
# Google提供的训练好的模型文件
CKPT_FILE = "./inception_v3.ckpt"  # 只有模型参数，而无网络结构

# 定义训练时使用的参数
LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASSES = 5  # 物种类别

# 指明不需要从训练好的inception-v3模型恢复的那些变量，即全连接层所涉及的所有变量，也是我们需要重新训练的变量
# 这里给出的是那些变量的前缀，即变量的命名空间
CHECKPOINT_EXCLUDE_SCOPES = "InceptionV3/Logits,InceptionV3/AuxLogits"
# 需要训练的参数名称，即上面的全连接层参数，给出的也是变量前缀
TRAINABLE_SCOPES = "InceptionV3/Logits,InceptionV3/AuxLogits"


# 获取所有需要从google训练好的模型中加载的参数
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(",")]
    variables_to_restore = []
    # 枚举inception-v3模型中的所有参数，然后判断是否需要从加载列表中移除
    # slim.get_model_variables():通过调用默认图，获取其上的 ops.GraphKeys.MODEL_VARIABLES ==> key 集合里面的所有变量：包括所有模型参数
    # 最终调用的是：get_collection(key, scope=None): return get_default_graph().get_collection(key, scope)
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True  # 不加载该变量
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


# 获取所有需要训练的变量列表
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(",")]
    variables_to_train = []
    for scope in scopes:
        # tf.get_collection最终调用的是：get_collection(key, scope=None): return get_default_graph().get_collection(key, scope)
        # scope:使用 re.match 筛选集合中与scope匹配的那些变量
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main(argv=None):
    # 加载预处理好的数据
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_examples = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    '''
    # 因为list无法进行多维切片，故需要先将其转为 numpy的array进行多维切片
    tmp = np.array(validation_images)
    validation_images = (tmp[:BATCH, :, :, :]).tolist()
    tmp = np.array(validation_labels)
    validation_labels = (tmp[:BATCH]).tolist()
    tmp = np.array(testing_images)
    testing_images = (tmp[:BATCH, :, :, :]).tolist()
    tmp = np.array(testing_labels)
    testing_labels = (tmp[:BATCH]).tolist()
    '''
    print("%d training examples, %d validation examples and %d testing examples." %
          (n_training_examples, len(validation_labels), len(testing_labels)))

    # 定义inception-v3的输入
    images = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, shape=[None], name='labels')
    # 定义inception-v3模型
    # 即实现inception-v3前向传播过程，因为google给的训练好的模型文件只有参数取值
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES, is_training=True)

    trainable_variables = get_trainable_variables()

    # 定义交叉熵损失。模型在定义时已经将正则化损失加入到损失集合了
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
    # 定义训练过程。minimize的过程中指定了需要优化的变量集合
    # tf.losses.get_total_loss()：返回ops.GraphKeys.LOSSES(名字为losses)集合中变量(损失)与正则项损失的总和，
    total_loss = tf.losses.get_total_loss()
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(total_loss)

    # 计算正确率
    with tf.name_scope("evaluation"):
        correction_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

    # 定义加载模型的函数
    load_fn = slim.assign_from_checkpoint_fn(
        CKPT_FILE,
        # 因为前面重新定义了inception-v3模型以及前向传播过程，所以那些计算节点已经被默认加入到默认的计算图，
        # 故et_tuned_variables函数调用的get_collection函数就能获得默认图，从而获得所有模型变量
        get_tuned_variables(),
        ignore_missing_vars=True)

    # 保存新训练的模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 初始化没有加载进来的变量。此过程一定要在模型加载之前，因为模型加载之后，相应的变量值被覆盖
        init = tf.global_variables_initializer()
        sess.run(init)

        # 加载google训练好的模型(只加载指定的不需要再训练的变量值，不加载全连接层的变量值)
        print("Loading tuned variables from %s " % CKPT_FILE)
        load_fn(sess)  # 类似于 saver.restore函数，覆盖不用训练的变量值

        start = 0
        end = BATCH
        for i in range(STEPS):
            # 这里只会更新指定的部分参数，而不是全部参数???如何实现的，难道是因为 loss集合里面只有 全连接层变量的损失项？？？
            _, loss = sess.run([train_step, total_loss],
                               feed_dict={
                                   images: training_images[start:end],
                                   labels: training_labels[start:end]})
            # 输出日志
            if i % 30 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=i)
                validation_accuracy = sess.run(evaluation_step,
                                               feed_dict={
                                                   images: validation_images,
                                                   labels: validation_labels})
                print("Step %d: Training loss is %.1f Validation accuracy = %.1f%% " %
                      (i, loss, validation_accuracy * 100.0))

            # 在生成训练数据时，已打乱数据，故这里只需顺序使用
            start = end
            if start == n_training_examples:
                start = 0
            end = start + BATCH
            if end > n_training_examples:
                end = n_training_examples

        # 最后在测试集上测试正确率
        test_accuracy = sess.run(evaluation_step, feed_dict={images: testing_images, labels: testing_labels})
        print("Final test accuracy = %.1f%% " % (test_accuracy * 100.0))


if __name__ == "__main__":
    tf.app.run()
