import tensorflow as tf
import matplotlib.pyplot as plt

'''
 给图像添加标注框
'''

img_path = "../cat.jpg"
img_raw_data = tf.gfile.FastGFile(img_path, "rb").read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(img_raw_data)
    # 将图像缩小
    img_data = tf.image.resize_images(img_data, [180, 267], method=1)
    # draw_bounding_boxes：输入是一个batch的数据，即多张图片组成的四维数据，故需要增加一维。同时要求输入是实数
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, dtype=tf.float32), 0)
    # 给出每张图片的所有标注框：[Ymin,Xmin,Ymax,Xmax]，都是相对位置
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    # 带有标注框的图片
    result = tf.image.draw_bounding_boxes(batched, boxes)  # 四维数组

    # 随机截取图像上有信息含量的部分也是一种提高模型鲁棒性的方式
    # min_object_covered：随机截取部分至少包含boxes的40%的内容
    # bbox_for_draw：标注框位置
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(img_data), bounding_boxes=boxes,
                                                                        min_object_covered=0.4)
    print("begin:\n", begin.eval())
    print("size:\n", size.eval())
    print("bbox_for_draw:\n", bbox_for_draw.eval())
    # 带标注框的图片
    img_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    # 随机截取的图片
    distorted_img = tf.slice(img_data, begin, size)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(result.eval()[0])
    plt.subplot(1, 3, 2)
    plt.imshow(img_with_box.eval()[0])
    plt.subplot(1, 3, 3)
    plt.imshow(distorted_img.eval())
    plt.show()
