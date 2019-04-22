import tensorflow as tf
import matplotlib.pyplot as plt

'''
 调整图像的亮度、对比度、饱和度和色相都不会影响识别结果，可以随机调整训练图像的这些属性，使训练得到的模型尽可能小地受到无关因素的影响
'''

img_path = "../cat.jpg"
img_raw_data = tf.gfile.FastGFile(img_path, "rb").read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(img_raw_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    # 调整亮度
    # 将亮度 -0.5
    adjusted_sub_bri = tf.image.adjust_brightness(img_data, -0.5)
    # 色调调整可能导致像素值超出0-1范围，可视化前需要截断在0-1范围内，否则无法正常显示
    adjusted_sub_bri = tf.clip_by_value(adjusted_sub_bri, 0.0, 1.0)
    adjusted_plus_bri = tf.image.adjust_brightness(img_data, 0.5)
    adjusted_plus_bri = tf.clip_by_value(adjusted_plus_bri, 0.0, 1.0)
    # [-max_delta,max_delta]内随机调整亮度
    adjusted_bri = tf.image.random_brightness(img_data, max_delta=0.5)
    adjusted_bri = tf.clip_by_value(adjusted_bri, 0.0, 1.0)

    # 调整对比度
    # 对比度减少0.5倍
    adjusted_sub_con = tf.image.adjust_contrast(img_data, 0.5)
    adjusted_sub_con = tf.clip_by_value(adjusted_sub_con, 0.0, 1.0)
    # 增加5倍
    adjusted_plus_con = tf.image.adjust_contrast(img_data, 5)
    adjusted_plus_con = tf.clip_by_value(adjusted_plus_con, 0.0, 1.0)
    # 在[lower,upper]范围内随机调整对比度
    adjusted_con = tf.image.random_contrast(img_data, lower=0.5, upper=5)
    adjusted_con = tf.clip_by_value(adjusted_con, 0.0, 1.0)

    # 调整色调Hue
    adjusted_hue_1 = tf.image.adjust_hue(img_data, 0.1)
    adjusted_hue_3 = tf.image.adjust_hue(img_data, 0.3)
    adjusted_hue_6 = tf.image.adjust_hue(img_data, 0.6)
    adjusted_hue_9 = tf.image.adjust_hue(img_data, 0.9)
    # 在[-max_delta, max_delta]范围内随机调整色调
    # max_delta取值：0-0.5
    adjusted_hue = tf.image.random_hue(img_data, max_delta=0.4)

    # 饱和度
    # 饱和度-5
    adjusted_sat_sub = tf.image.adjust_saturation(img_data, -5)
    adjusted_sat_sub = tf.clip_by_value(adjusted_sat_sub, 0.0, 1.0)
    # 饱和度+5
    adjusted_sat_plus = tf.image.adjust_saturation(img_data, 5)
    adjusted_sat_plus = tf.clip_by_value(adjusted_sat_plus, 0.0, 1.0)
    # 在[lower,upper]范围内随机调整饱和度
    adjusted_sat = tf.image.random_saturation(img_data, lower=2, upper=10)
    adjusted_sat = tf.clip_by_value(adjusted_sat, 0.0, 1.0)

    # 图像标准化：图像的亮度信息均值变为0，方差变为1
    adjusted_stand = tf.image.per_image_standardization(img_data)

    plt.figure()
    plt.subplot(6, 3, 2)
    plt.imshow(img_data.eval())

    plt.subplot(6, 3, 4)
    plt.imshow(adjusted_sub_bri.eval())
    plt.subplot(6, 3, 5)
    plt.imshow(adjusted_plus_bri.eval())
    plt.subplot(6, 3, 6)
    plt.imshow(adjusted_bri.eval())

    plt.subplot(6, 3, 7)
    plt.imshow(adjusted_sub_con.eval())
    plt.subplot(6, 3, 8)
    plt.imshow(adjusted_plus_con.eval())
    plt.subplot(6, 3, 9)
    plt.imshow(adjusted_con.eval())

    plt.subplot(6, 3, 10)
    plt.imshow(adjusted_hue_3.eval())
    plt.subplot(6, 3, 11)
    plt.imshow(adjusted_hue_6.eval())
    plt.subplot(6, 3, 12)
    plt.imshow(adjusted_hue_9.eval())

    plt.subplot(6, 3, 13)
    plt.imshow(adjusted_sat_sub.eval())
    plt.subplot(6, 3, 14)
    plt.imshow(adjusted_sat_plus.eval())
    plt.subplot(6, 3, 15)
    plt.imshow(adjusted_sat.eval())

    plt.subplot(6, 3, 17)
    plt.imshow(adjusted_stand.eval())

    plt.show()