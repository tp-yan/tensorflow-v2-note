import tensorflow as tf
import matplotlib.pyplot as plt

'''
 因为神经网络的输入层节点个数固定，而图片大小是不固定的，故需要将图像大小统一。
 方式1： tf.image.resize_images函数，通过算法使原始图像上的所有信息尽量保存
 方式2： resize_image_with_crop_or_pad对图像进行裁剪或填充
'''

image_raw_data = tf.gfile.FastGFile("../cat.jpg", "rb").read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    # print(img_data.eval())
    # 这里将图片数据转为实数类型，将0-255的像素值转为0.0-1.0范围内的实数
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)  # 自动转换后缩放到0~1区间
    # print(img_data.eval())
    # 如果输入是uint8格式，那么输出将是0~255之内的实数，不方便后续处理
    # 故在调整图像大小之前先转为实数类型
    # method：0：双线性差值 1：最近领法 2：双三次差值法 3：面积插值法
    resized1 = tf.image.resize_images(img_data, [300, 300], method=0)
    resized2 = tf.image.resize_images(img_data, [300, 300], method=1)
    resized3 = tf.image.resize_images(img_data, [300, 300], method=2)
    # resize_image_with_crop_or_pad 函数对图像进行裁剪（截取图像居中部分）或填充
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)  # 目标尺寸比原尺寸小，裁剪
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)  # 目标尺寸比原尺寸大，四周用0填充
    # 按比例裁剪图像，比例必须是0-1之间的数
    central_cropped = tf.image.central_crop(img_data, 0.5)
    # 裁剪或填充指定区域的图像
    # tf.image.crop_to_bounding_box:
    # tf.image.pad_to_bounding_box

    plt.figure()

    plt.subplot(3, 3, 1)
    plt.imshow(resized1.eval())
    plt.subplot(3, 3, 2)
    plt.imshow(resized2.eval())
    plt.subplot(3, 3, 3)
    plt.imshow(resized3.eval())

    plt.subplot(3, 3, 4)
    plt.imshow(croped.eval())
    plt.subplot(3, 3, 5)
    plt.imshow(padded.eval())
    plt.subplot(3, 3, 6)
    plt.imshow(central_cropped.eval())

    plt.show()
