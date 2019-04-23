import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
 完整的图像预处理样例，输入是原始图像，输出则是指定尺寸的神经网络输入层输入图像
'''


# 随机调整图像的色彩：对比度，亮度、饱和度等，不同的顺序会导致不同的结果，这里随机选择一种顺序
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    else:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image, 0.0, 1.0)


# 输入原始图像，输出则是神经网络输入层图像。按给定尺寸和标注框范围对图像进行预处理。这一步只需在训练时进行，预测时直接输入原图像
def preprocess_for_train(image, height, width, bbox):
    # 如果没有标注框，则认为整个图像是需要关注的部分
    if bbox == None:
        # shape=[1, 1, 4]：将[0.0, 0.0, 1.0, 1.0]重塑为 1x1x4的矩阵:[[[0. 0. 1. 1.]]]
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        print("bbox:", bbox.eval())

    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机截取图像，减小 需要关注的物体小大 对图像识别算法的影响
    bbox_begin, bbox_size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)

    distort_img = tf.slice(image, bbox_begin, bbox_size)
    # 将随机截取图像调整为NN输入层的大小。大小调整算法随机选择
    distort_img = tf.image.resize_images(distort_img, [height, width], method=np.random.randint(4))
    # 随机左右翻转图像
    distort_img = tf.image.random_flip_left_right(distort_img)
    # 随机调整图像色彩
    distort_img = distort_color(distort_img, np.random.randint(2))
    return distort_img


img_path = "../cat.jpg"
image_raw_data = tf.gfile.FastGFile(img_path, "rb").read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    for i in range(4):
        result = preprocess_for_train(img_data, 299, 299, boxes)
        plt.imshow(result.eval())
        plt.show()
