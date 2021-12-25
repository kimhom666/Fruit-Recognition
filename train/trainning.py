import numpy as np
import tensorflow as tf
from train import model, input_data

N_CLASSES = 5  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
IMG_W = 104  # 重新定义图片的大小，图片如果过大则训练比较慢
IMG_H = 104
BATCH_SIZE = 32  # 每批数据的大小
CAPACITY = 256
MAX_STEP = 12000  # 训练的步数，应当 >= 10000
learning_rate = 0.00001  # 学习率，建议刚开始的 learning_rate <= 0.0001


def run_training():
    # 数据集
    logs_train_dir = "logs/"

    train, train_label = input_data.get_files("D:/fruit_recognization/get_information/西瓜测试/")
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    keep_prob = 0.5
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES, keep_prob)
    train_loss = model.losses(train_logits, train_label)
    train_op = model.trainning(train_loss, learning_rate)
    train_acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    # 移植模型需要的接口
    tf.get_variable_scope().reuse_variables()
    x = tf.placeholder(tf.float32, shape=[104, 104, 3], name='x')
    image = tf.reshape(x, [1, 104, 104, 3])
    logit = model.inference(image, 1, N_CLASSES, keep_prob)
    logit = tf.nn.softmax(logit, name='logit')
    pre_num = tf.argmax(logit, 1, output_type='int32', name="output")

    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

            if step % 100 == 0:
                print("Step %d, train loss = %.2f, train accuracy = %.2f" % (step, tra_loss, tra_acc))
                # summary_str = sess.run(summary_op)
                # train_writer.add_summary(summary_str, step)
            if step % 1000 == 0 or (step + 1) == MAX_STEP:
                # checkpoint_path = os.path.join(logs_train_dir, "model.ckpt")
                # saver.save(sess, checkpoint_path, global_step=step)
                # 保存pb文件
                # 保存pb文件,用于android移植
                # 保存训练好的模型
                # 形参output_node_names用于指定输出的节点名称,output_node_names=['output']对应pre_num=tf.argmax(y,1,name="output"),
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                                output_node_names=['output'])
                with tf.gfile.FastGFile('logs/5kinds-' + str(step) + '.pb',
                                        mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
                    f.write(output_graph_def.SerializeToString())
    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached.")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


# train
run_training()