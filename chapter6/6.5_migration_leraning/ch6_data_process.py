import tensorflow as tf
import glob
import os.path
import numpy as np
from tensorflow.python.platform import gfile

'''
 此程序完成数据预处理工作，主要包括：
 将原始图像数据(5个目录存放5个不同类别花的图片)，经过整理以numpy的格式保存
'''
# 原始输入数据目录:包含5个子目录
INPUT_DATA = "../../datasets/flower_photos"
OUTPUT_FILE = "./flower_processed_data.npy"

# 验证和测试集所占比例
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


# 读取数据并将数据分割为训练、验证、测试数据集
def create_image_lists(sess, testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  #
    is_root_dir = True

    # 初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0  # 以数字代表花类别

    # 读取所有子目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False  # 第一个元素是根目录路径
            continue
        # 获取一个子目录中的所有图片
        # extensions = ['jpg', 'jpeg', 'JPG', 'JPEG'] # 在Linux下通配符区分大小写
        extensions = ['jpg', 'jpeg']  # 在Windows下，通配符不区分大小写
        file_list = []
        dir_name = os.path.basename(sub_dir)  # 目录路径中的最后一个目录名
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
            # glob.glob(file_glob):按正则表达式去匹配图片文件，并以list返回所有匹配文件的相对路径
            # ！！注：！！glob.glob()里面的通配符，在Windows下是不区分大小写的，而在Linux下是区分大小写的，所以在Linux下没这个问题。
            file_list.extend(glob.glob(file_glob))  # 将一个list追加到另一个list的末尾，相当于list拼接
        if not file_list:
            continue

        print("\n======================", dir_name, "======================\n")
        i = 0
        # 处理图片数据
        for file_name in file_list:
            i += 1
            # 解析图片，并将图片转化为299x299，适合inception-v3模型输入的格式
            image_raw_data = gfile.FastGFile(file_name, "rb").read()
            image = tf.image.decode_jpeg(image_raw_data)  # jpeg域解析到原始像素域
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            """
            tf.image.resize_images：TF中的一个计算节点
            输入：4-D Tensor of shape `[batch, height, width, channels] or 3-D Tensor of shape `[height, width, channels]
            Returns:
            If `images` was 4-D, a 4-D float Tensor of shape
            `[batch, new_height, new_width, channels]`.
            If `images` was 3-D, a 3-D float Tensor of shape
            `[new_height, new_width, channels]`.
            """
            # image:was 3-D, a 3-D float Tensor of shape `[new_height, new_width, channels]`
            image = tf.image.resize_images(image, [299, 299])  # 原始图片的尺寸和大小并不一致！！！
            image_value = sess.run(image)  # image_value: numpy.ndarray 类型
            # print(image_value.shape)  # (299, 299, 3)

            # 随机划分数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
            if i % 200 == 0:
                print(i, " images processed.")
        current_label += 1

    # 将训练数据随机打乱已获得更好的训练效果
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)  # 保证两次shuffle次序一样
    np.random.shuffle(training_labels)

    return np.asarray(
        [training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels])


def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        # 通过numpy格式（.npy）保存处理后的数据
        np.save(OUTPUT_FILE, processed_data)


if __name__ == "__main__":
    main()
