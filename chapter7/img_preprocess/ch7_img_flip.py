import tensorflow as tf
import matplotlib.pyplot as plt

'''
 许多图像识别问题中，图像的翻转不应该影响识别结果。在训练模型时，可随机地翻转训练图像，使模型可识别不同角度的实体。
 随机地翻转训练图像是一种常见的数据增强方式
'''

img_path = "../cat.jpg"
img_raw_data = tf.gfile.FastGFile(img_path, "rb").read()
# print(img_raw_data)

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(img_raw_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # 上下翻转
    flipped_ud = tf.image.flip_up_down(img_data)
    # 左右翻转
    flipped_lr = tf.image.flip_left_right(img_data)
    # 沿对角线翻转
    transposed = tf.image.transpose_image(img_data)

    # 图像随机翻转 50%概率
    flipped = tf.image.random_flip_up_down(img_data)
    flipped = tf.image.random_flip_left_right(img_data)

    plt.figure()  # 创建一个新图，并注册为默认图
    plt.subplot(2, 2, 1)  # 给当前图添加子图
    plt.imshow(img_data.eval())
    plt.subplot(2, 2, 2)
    plt.imshow(flipped_ud.eval())
    plt.subplot(2, 2, 3)
    plt.imshow(flipped_lr.eval())
    plt.subplot(2, 2, 4)
    plt.imshow(transposed.eval())
    plt.show()
