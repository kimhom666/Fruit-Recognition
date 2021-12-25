import tensorflow as tf
import numpy as np
import os
img_width = 104
img_height = 104


def get_files(file_dir):
    pic_list = []
    label_list = []
    file_list = os.listdir(file_dir)
    for index in range(len(file_list)):
        pic_path_list = os.listdir(file_dir+file_list[index])
        for pic in pic_path_list:
            pic_list.append(file_dir+file_list[index]+'/'+pic)
            label_list.append(index)
    temp = np.array([pic_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    print(label_list)
    print(image_list)
    return image_list, label_list

def one_hot(labels, N_classes):
    one_hot_labels = []
    for label in labels:
        onehot = np.zeros(N_classes)
        onehot[label] = 1
        one_hot_labels.append(onehot)
    return one_hot_labels


def get_Batch(data, label, batch_size):
    print(data.shape, label.shape)
    input_queue = tf.train.slice_input_producer([data, label], num_epochs=1, shuffle=True, capacity=32)
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
    return x_batch, y_batch



def get_batch(image, label, image_W, image_H, batch_size):
    capacity = 32
    # 转换数据为 ts 能识别的格式
    image = tf.cast(image, tf.string)
    # 将image 和 label 放倒队列里
    input_queue = tf.train.slice_input_producer([image, label])
    print(input_queue[0])
    label = input_queue[1]
    print("______________")
    label = tf.one_hot(input_queue[1], 5)
    # 读取图片的全部信息
    image_contents = tf.read_file(input_queue[0])
    # 把图片解码，channels ＝3 为彩色图片, r，g ，b  黑白图片为 1 ，也可以理解为图片的厚度
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 对数据进行标准化,标准化，就是减去它的均值，除以他的方差
    image = tf.image.per_image_standardization(image)
    # 生成批次  num_threads 有多少个线程根据电脑配置设置  capacity 队列中 最多容纳图片的个数  tf.train.shuffle_batch 打乱顺序，
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)

    # 重新定义下 label_batch 的形状
    print('shape of label_batch:  ', end='')
    print(label_batch.shape)
    # 转化图片
    image_batch = tf.cast(image_batch, tf.float32)
    print(image_batch.shape)
    return image_batch, label_batch


train_images, train_labels = get_files("D:/fruit_recognization/get_information/西瓜测试/")
train_images_batch, train_label_batch = get_batch(train_images,
                                                      train_labels,
                                                      104,
                                                      104,
                                                      32,)
