import os

'''
 遍历文件夹的2种方式
'''


# 方法一：os.listdir
def gci(filepath):
    # 遍历filepath下所有文件，包括子目录
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            gci(fi_d)
        else:
            print(os.path.join(filepath, fi_d))


# 递归遍历当前目录下所有文件
# pwd = os.getcwd()
# print(pwd)
# gci(pwd)

#  方法二：os.walk：返回三元tupple(dirpath, dirnames, filenames)
# dirpath是一个string，代表目录的路径
# dirnames是一个list，包含了dirpath下所有子目录的名字
# filenames是一个list，包含了非目录文件的名字
# 注：这些名字不包含路径信息,如果需要得到全路径,需要使用 os.path.join(dirpath, name).
for fpathe, dirs, fs in os.walk('../'):  # './'：当前文件所在目录 '../':父目录
    print(fpathe)  # ../
    print(
        dirs)  # ['.git', '.idea', '201806-github代码数据打包', 'chapter3', 'chapter4', 'chapter5', 'chapter6', 'datasets', 'my_utils']
    print(fs)  # ['test.py']
    '''
        for f in fs:
            pass
            # print(os.path.join(fpathe, f))
    '''
    break



INPUT_DATA = "./datasets/flower_photos"
# 依次遍历指定路径下每个目录
for x, y, z in os.walk(INPUT_DATA):
    # x:当前目录对应的相对路径，字符串
    # y：当前目录下的所有子目录名列表，list
    # z:当前目录下的所有文件列表，list
    print(x, "\t||\t", y, "\t||\t", z)

sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
print(sub_dirs)