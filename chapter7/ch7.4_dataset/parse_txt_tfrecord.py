import tensorflow as tf

def parser(record):
    features = tf.parse_single_example(record, features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)
    })
    return features['i'], features['j']