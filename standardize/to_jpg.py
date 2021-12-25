# 将其转为JPG
from PIL import Image
import os
import numpy as np


def is_jpg(file_path):
    file_list = os.listdir(file_path)
    for file in file_list:
        if '.' not in file:
            pic_list = os.listdir(os.path.join(file_path,file))
            for pic in pic_list:
                try:
                    pic_content = Image.open(os.path.join(file_path, file, pic))
                    temp = np.array(pic_content).shape
                    try:
                        temp_1 = temp[2]
                    except Exception:
                        print("sdasd")
                        os.remove(os.path.join(file_path, file, pic))
                    if pic_content.format == 'JPEG':
                        pass
                except IOError:
                    print('害群之马')
                    print(os.path.join(file_path,file,pic))
                    os.remove(os.path.join(file_path,file,pic))


def size_of_pic(file_path):
    pic_list = os.listdir(file_path)
    for pic in pic_list:
        pic_content = Image.open(os.path.join(file_path, pic))
        temp = np.array(pic_content).shape
        try:
            temp[2] != 3
        except Exception:
            print('nONE')
            os.remove(os.path.join(file_path, pic))



is_jpg("D:/fruit_recognization/get_information/西瓜测试")