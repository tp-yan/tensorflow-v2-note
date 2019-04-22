import tensorflow as tf
import matplotlib.pyplot as plt  # python画图工具

# 读取图像原始数据
image_raw_data = tf.gfile.FastGFile("cat.jpg", "rb").read()

with tf.Session() as sess:
    # 对图像的JPEG格式解码得到三维矩阵--一个张量
    # tf.image.decode_png:解码png格式
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())

    # 使用pyplot可视化图像
    plt.imshow(img_data.eval())
    plt.show()

    # 将三维矩阵按照JPEG格式编码输出图像并保存
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("./output.jpg", "wb") as f:
        f.write(encoded_image.eval())
