import os
import numpy as np 
import tensorflow as tf 
from PIL import Image

from train import cal_loss, cnn, cal_acc


def read_file_list(filelist):
    """
    Scan the image file and get the image paths and labels
    """
    with open(filelist) as f:
        lines = f.readlines()
        files = []
        for l in lines:
            items = l.split()
            files.append(items[0])
            
    return files

def label2vector(strings):
    label_num = [int(i) for i in strings]
    # label_num = np.array(label_num)
    label = np.zeros([4, 10])
    for i, item in enumerate(label_num):
        label[i, item] = 1.0

    return np.reshape(label, [1, 40])


def vector2label(vector):

    label = ''
    for i in range(4):
        val = np.argmax(vector[i*10:(i+1)*10], axis=-1)
        label += (str(val))

    return label


def test(image, label):
    x = tf.placeholder(tf.float32, [None, 60, 160, 1])
    y = tf.placeholder(tf.float32, [None, 40])

    predict = cnn(x, keep_prob=1)

    loss = cal_loss(predict, y)

    # accurary
    accurary = cal_acc(predict, y)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, './checkpoints/model.ckpt')


        fetch = {
            'predict': predict,
            'loss': loss,
            'accurary': accurary,
        }
        result = sess.run(fetch, feed_dict={x:image, y:label})

    return result


if __name__ == "__main__":
    
    filedir = './test.txt'
    filename = read_file_list(filelist=filedir)
    image = np.zeros([len(filename), 60, 160, 1])
    label = np.zeros([len(filename), 40])

    for i, line in enumerate(filename):
        x = Image.open(line)
        x = np.mean(np.array(x), axis=2) / 255.0
        image[i] = np.expand_dims(x, axis=3)
        # 获取labels
        name = line.split('/')[-1][0:4]
        label[i] = label2vector(name)

    result = test(image, label)
    for i in range(result['predict'].shape[0]):
        pre_text = vector2label(result['predict'][i])
        text = vector2label(label[i])
        
        print("验证码正确值:", text, ' 模型预测值:', pre_text)
    print('acc:', result['accurary'])
    print('loss:', result['loss'])
