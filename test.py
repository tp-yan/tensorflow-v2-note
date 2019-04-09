import glob
import os.path

INPUT_DATA = "./datasets/flower_photos"

# 依次遍历指定路径下每个目录
for x, y, z in os.walk(INPUT_DATA):
    # x:当前目录对应的路径，字符串
    # y：当前目录下的所有子目录名列表，list
    # z:当前目录下的所有文件列表，list
    print(x, "\t||\t", y, "\t||\t", z)

sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
print(sub_dirs)
sub_dir = "./datasets/flower_photos\daisy"
dir_name = os.path.basename(sub_dir)
print(dir_name)
extension = "jpg"
file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
print("\nfile_glob:\n", file_glob)
jpgs = glob.glob(file_glob)
print(jpgs)
