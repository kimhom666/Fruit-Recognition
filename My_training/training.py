import tensorflow as tf
from My_training import input_data
from numpy import arange
N_Classes = 5
Img_w = 104
Img_h = 104
channel_num = 3
learning_rate = 10e-5
BATCH_SIZE = 32
CAPACITY = 256

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


train_images, train_labels = input_data.get_files("D:/fruit_recognization/get_information/西瓜测试/")
train_images_batch, train_label_batch = input_data.get_batch(train_images,
                                                      train_labels,
                                                      Img_w,
                                                      Img_h,
                                                      BATCH_SIZE)

images = tf.placeholder(dtype=tf.float32, shape=[None, Img_h, Img_w, channel_num])
labels = tf.placeholder(dtype=tf.float32, shape=[None, N_Classes])
W_conv1 = weight_variable([3, 3, 3, 16])
b_conv1 = bias_variable([16])

x_image = tf.reshape(images,[-1,Img_w, Img_h, channel_num])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 16, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

print('shape of it ')
print(h_pool2.shape)
batch_size = 32
h_pool2_flat = tf.reshape(h_pool2, shape=[batch_size, -1])
W_fc1 = weight_variable([26*26*16, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable(shape=[1024, 1024])
b_fc2 = bias_variable([1024])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


W_fc3 = weight_variable([1024, N_Classes])
b_fc3 = bias_variable([N_Classes])

y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

cross_entropy = -tf.reduce_sum(labels*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    # 开启协调器
    coord = tf.train.Coordinator()
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners(sess, coord)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    try:
        for i in range(20000):
            if coord.should_stop():
                break
            data, label = sess.run([train_images_batch, train_label_batch])
            sess.run(train_step, feed_dict={images: data, labels: label, keep_prob: 0.5})
            train_accuracy = accuracy.eval({images: data, labels: label, keep_prob: 0.5})
            if i%100 == 0:
                print("%d step,, Testing accuracy %g" % (i,train_accuracy))
    except tf.errors.OutOfRangeError:  # num_epochs 次数用完会抛出此异常
        print("---Train end---")
    finally:
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
        print('---Programm end---')
    coord.join(threads)

