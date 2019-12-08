import os
import numpy as np 
import tensorflow as tf 
import natsort
import time

import data_process


DATADIR = './captcha/'
TRAINDATA = 5000
BATCH_SIZE = 32
MAX_EPOCH = 200
LEARNING_RATE = 0.001
KEEP_PROB = 0.7


def get_batches(batch_size, fqueue):
    with tf.name_scope(None, 'read_data', [fqueue]):
        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(fqueue)
        image, label = data_process.read_tfrecord(example_serialized)
        min_after_dequeue = 1000
        num_threads = 1
        capacity = min_after_dequeue + 30 * batch_size

        image = tf.cast(tf.reshape(image, [60, 160, 1]), dtype=tf.float32)
        pack_these = [image, label]
        pack_name = ['image', 'label']
        all_batched = tf.train.shuffle_batch(
            pack_these,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=False,
            name='input_batch_train')

        batch_dict = {}
        for name, batch in zip(pack_name, all_batched):
            batch_dict[name] = batch

        return batch_dict

def get_weight(shape, is_training=True):

    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(initial_value=init, trainable=is_training)

def get_bais(shape, is_training=True):

    init = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial_value=init, trainable=is_training)

def conv2d(x, w):
    '''
    :param x: input tensor of shape [batch, in_height, in_width, in_channels]
    :param w: filter / kernel of shape [filter_height, filter_wedth, in_channels, out_channels]
    :param : strides[0] = strides[3] = 1, strides[1]为x方向步长，strides[2]为y方向步长
    :param : padding: a string of "SAME", "VALID"

    '''
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):

    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def cnn(x, keep_prob):
    
    '''
        x format = NHWC
        x shape: [batch, height, width, channel]
    '''
    # batch_size, height, width = x.shape

    # CNN
    with tf.variable_scope('CNN') as sc:

        # 第一层
        w1 = get_weight([3,3,1,32]) # 3x3 size, channel 1, output 32
        b1 = get_bais([32])

        output_1 = tf.nn.relu(conv2d(x, w1) + b1)
        output_pool_1 = max_pool_2x2(output_1)
        output_pool_1 = tf.nn.dropout(output_pool_1, keep_prob=keep_prob)


        # 第二层
        w2 = get_weight([3,3,32,64]) # 3x3 size, channel 32, output 64
        b2 = get_bais([64])

        output_2 = tf.nn.relu(conv2d(output_pool_1, w2) + b2)
        output_pool_2 = max_pool_2x2(output_2)
        output_pool_2 = tf.nn.dropout(output_pool_2, keep_prob=keep_prob)

        # 第三层
        w3 = get_weight([3,3,64,64]) # 3x3 size, channel 64, output 64
        b3 = get_bais([64])

        output_3 = tf.nn.relu(conv2d(output_pool_2, w3) + b3)
        output_pool_3 = max_pool_2x2(output_3)
        output_pool_3 = tf.nn.dropout(output_pool_3, keep_prob=keep_prob)


        # 全连接层 1
        image_height = int(output_pool_3.shape[1])
        image_width = int(output_pool_3.shape[2])

        output_pool_3_flat = tf.reshape(output_pool_3, [-1, image_height*image_width*64])
        w_fc_1 = get_weight([image_height*image_width*64, 1024])
        b_fc_1 = get_bais([1024])
        output_fc_1 = tf.nn.relu(tf.matmul(output_pool_3_flat, w_fc_1) + b_fc_1)
        output_drop = tf.nn.dropout(output_fc_1, keep_prob=keep_prob)

        # 全连接层 2
        w_fc_2 = get_weight([1024, 40])
        b_fc_2 = get_bais([40])
        predict = tf.matmul(output_drop, w_fc_2) + b_fc_2

        return predict

def cal_loss(predict, y):
    with tf.name_scope('loss'):
    # 最后一层的激活函数在 sigmoid_cross_entropy_with_logits() 这个函数中，不用单独添加
    # loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y[:, 0:10], logits=predict[:, 0:10]))
    # loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y[:,10:20], logits=predict[:,10:20]))
    # loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y[:,20:30], logits=predict[:,20:30]))
    # loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y[:,30:40], logits=predict[:,30:40]))

    # return (loss0+loss1+loss2+loss3) / 4
    # y = tf.reshape(y, [-1, 4, 10])
    # predict = tf.reshape(predict, [-1, 4, 10])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predict))
        tf.summary.scalar('loss', loss)
        return loss

def cal_acc(predict, y):

    with tf.name_scope('accurary'):

        # accurary
        predict_4 = tf.reshape(predict, [-1, 4, 10])
        label_4 = tf.reshape(y, [-1, 4, 10])
        correct_prediction = tf.equal(tf.argmax(label_4,2),tf.argmax(predict_4,2))
        accurary = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accurary', accurary)

        return accurary

def train_one_epoch(sess, op, train_writer):

    num_batch = TRAINDATA // BATCH_SIZE

    loss_sum = 0
    for i in range(num_batch):
        start_time = time.time()
        merge, global_step, loss_val, _, accurary_val = sess.run([op['merge'], \
                                                                 op['step'], \
                                                                 op['loss'], \
                                                                 op['optimizer'], \
                                                                 op['accurary']])
        end_time = time.time()

        loss_sum += loss_val
        print('time:%.04f, step:%d, loss:%.5f, accurary:%.4f' % \
              ((end_time-start_time), global_step, loss_val, accurary_val))

        if global_step % 10 == 0:
            with open('./loss.txt', 'a+') as f:
                f.write('step:%d, loss: %06f \n' % (global_step, loss_val))
        train_writer.add_summary(merge, global_step)
    loss_avg = loss_sum / num_batch
    print("average loss in an epoch: %f ==================================="% (loss_avg))

def train():
    # with tf.Graph().as_default():
  
    filelist = os.listdir(DATADIR)
    data_dirs = natsort.natsorted(filelist)
    all_files = []
    for data_dir in data_dirs:
        if '.tfrecords' in data_dir:
            all_files.append(DATADIR + data_dir)

    do_shuffle = True
    fqueue = tf.train.string_input_producer(all_files, shuffle=do_shuffle, name="input")
    batch_dict = get_batches(BATCH_SIZE, fqueue)

    x = batch_dict['image']
    y = batch_dict['label']

    # model
    predict = cnn(x, KEEP_PROB)
    loss = cal_loss(predict, y)

    # accurary
    accurary = cal_acc(predict, y)

    # train and optimze
    step = tf.Variable(initial_value=0, trainable=False, name='gl_step')
    lr = tf.train.exponential_decay(learning_rate=LEARNING_RATE,\
                                    global_step=step, \
                                    decay_steps=2000, \
                                    decay_rate=0.9 )
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=step)

    # save model
    saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    coord = tf.train.Coordinator() 
    threads = tf.train.start_queue_runners(sess, coord)

    # merged = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'accuracy'), \
    #                            tf.get_collection(tf.GraphKeys.SUMMARIES,'loss')])
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/checkpoints', sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    op = {
        'merge': merged,
        'predict': predict,
        'loss': loss,
        'optimizer': optimizer,
        'step': step,
        'accurary': accurary,
    }

    for epoch in range(MAX_EPOCH):
        print('**** EPOCH %03d ****' % (epoch))

        train_one_epoch(sess, op, train_writer)


        if epoch % 5 == 0:
            savepath = saver.save(sess, os.path.join('./checkpoints', "model.ckpt"))

    train_writer.close()

    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
    
    train()