import numpy as np
import time
import tensorflow as tf

# 使用mnist_train与mnist_inference中定义的常量与函数
import mnist_train
import mnist_inference

# 每10s加载一次最新的模型，并在测试集上测试最新模型的正确率
from tensorflow.examples.tutorials.mnist import input_data

'''
 此程序应该与训练程序同时进行，因为训练程序每保存一次模型，就会更新checkpoint文件，而此程序才能读到最新的模型
 但往往训练程序不可能也是10s就保存一次模型，故此程序会对同一模型测试多次(因为checkpoint文件内容未更新)
 此程序是在以保存的模型上进行测试的
'''

EVAL_INTERVAL_SECS = 10
BATCH_SIZE = 5000  # 设为验证集样例个数，一次性将验证集全部输入进行正确率计算


def evaluate(mnist):
    with tf.Graph().as_default() as g:  # 生成一个新的图作为默认计算图
        # 与训练时一样，都需要重构模型网络结构
        # 定义输入输出格式
        x = tf.placeholder(tf.float32, shape=[
            BATCH_SIZE,  # 这里没使用None，是因为后面的 tf.reshape时需要具体batch的大小
            mnist_inference.IMAGE_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y-input')
        print(mnist.validation.images.shape)  # (5000, 784)
        reshaped_xs = np.reshape(mnist.validation.images, (
            BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
        print(reshaped_xs.shape)  # (5000, 28, 28, 1)
        validate_feed = {x: reshaped_xs, y_: mnist.validation.labels}  # 使用验证集完成测试

        # 测试时不需要正则项损失，故这里不使用正则化函数
        y = mnist_inference.inference(x, False, None)
        # tf.argmax(y,1)：得到输入样例的预测类别
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名方式加载模型，这样在前向传播过程中就不需要调用求滑动平均的函数了，完全共用 mnist_inference定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # get_checkpoint_state通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:  # model_checkpoint_path：model_mnist.ckpt-29001
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    evaluate(mnist)


if __name__ == "__main__":
    tf.app.run()
