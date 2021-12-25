import tensorflow as tf
from PIL import Image
import numpy as np
import os


def get_one_image(train):
    files = os.listdir(train)
    labels = []
    pics = []
    for file in files:
        temp = file.split('.')
        img_dir = os.path.join(train, file)
        image = Image.open(img_dir)
        image = image.resize([104, 104])
        image = np.array(image)
        pics.append(image)
        if temp[0][0] == 'm':
            labels.append(0)
        elif temp[0][0] == 's':
            labels.append(1)
        elif temp[0][0] == 'c':
            labels.append(2)
        elif temp[0][0] == 'o':
            labels.append(3)
        else:
            labels.append(4)

    temp = np.array([pics, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    pics = list(temp[:, 0])
    labels = list(temp[:, 1])
    labels = [int(float(i)) for i in labels]
    return pics, labels


def test_pd():
    images, labels = get_one_image('C:/Users/Administrator/Desktop/watermelon_test/')
    count = 0
    for index in range(len(images)):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open('logs/5kinds-11000.pb', "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                # x_test = x_test.reshape(1, 28 * 28)
                input_x = sess.graph.get_tensor_by_name("x:0")
                output = sess.graph.get_tensor_by_name("output:0")
                # 对图片进行测试
                image = tf.image.per_image_standardization(images[index])
                tf.global_variables_initializer().run()
                pre_num = sess.run(output, feed_dict={input_x: images[index]})  # 利用训练好的模型预测结果
                print(pre_num[0], labels[index])
                if int(pre_num[0]) == int(labels[index]):
                    count += 1
    print(float(count/len(images)))

test_pd()