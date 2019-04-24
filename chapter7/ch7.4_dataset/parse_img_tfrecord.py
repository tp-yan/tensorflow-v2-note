import tensorflow as tf


def parser(record):
    '''
    从 TFRecord中解析并还原图像
    :param record: 从TFRecord中读出的二进制Example
    :return: 原始图像，标签
    '''
    features = tf.parse_single_example(record, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64)
    })
    decoded_img = tf.decode_raw(features["image"], tf.uint8)
    decoded_img.set_shape([features['height'], features['width'], features['channels']])
    label = features['label']
    return decoded_img, label
