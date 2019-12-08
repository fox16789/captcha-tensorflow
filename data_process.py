from PIL import Image
import random
import sys
import os
import math
import tensorflow as tf
import numpy as np

# 验证集数量
_NUM_TEST = 500

# 随机种子
_RANDOM_SEED = 0

# 数据集路径
DATASET_DIR = '.\\captcha\\images\\'

# tfrecord文件存放路径
TFRECORD_DIR = '.\\captcha\\'

# 判断tfrecord是否存在
def _dataset_exits(data_set):
    for split_name in ['train', 'test']:
        output_filename = os.path.join(data_set,split_name + '.tfrecord')
        if not tf.gfile.Exists(output_filename):
            return False
    return True

# 获取所有验证码图片
def _get_filename_and_class(data_set):
    photo_filename = [] #存储所有的图片的路径
    for filename in os.listdir(data_set):
        path = os.path.join(data_set,filename)
        photo_filename.append(path)
    return photo_filename

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values=[values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def byte_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, label0, label1, label2, label3):
    return tf.train.Example(features=tf.train.Features(feature={
        'image':byte_feature(image_data),
        'label0':int64_feature(label0),
        'label1':int64_feature(label1),
        'label2':int64_feature(label2),
        'label3':int64_feature(label3),
    }))

# 把数据转为tfrecord格式
def _convert_dataset(split_name, filenames, dataset_dir):
    assert split_name in ['train', 'test']

    with tf.Session() as sess:
        # 定义tfrecord文件的路径和名字
        output_filename = os.path.join(TFRECORD_DIR,split_name+'.tfrecords')

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,filename in enumerate(filenames):
                try:
                    sys.stdout.write('\r>>Converting images %d/%d' % (i+1,len(filenames)))
                    sys.stdout.flush()

                    # 读取图片
                    image_data = Image.open(filename)
                    # 根据模型rezsize
                    # print(np.array(image_data).shape)
                    # image_data = image_data.resize((224,224),Image.NEAREST)
                    # 灰度化
                    # image_data = np.array(image_data.convert('L'))
                    image_data = np.mean(np.array(image_data), axis=2, dtype=int)
                    # image_data = np.squeeze(image_data, axis=2)
                    print(image_data.shape)
                    exit()
                    image_data = image_data.tobytes()

                    # 获取labels
                    labels = filename.split('\\')[-1][0:4]
                    num_labels = []
                    for j in range(4):
                        num_labels.append(int(labels[j]))
                    
                    # 生成protocol数据类型
                    example = image_to_tfexample(image_data,num_labels[0],num_labels[1],num_labels[2],num_labels[3])
                    tfrecord_writer.write(example.SerializeToString())

                except IOError as e:
                    print("Couldn't read", filename)
                    print('Error:', e)
                    print('Skip it\n')
    sys.stdout.write('\n')
    sys.stdout.flush()

def read_tfrecord(example_serialized):
    feature_map = {
        'image': tf.io.FixedLenFeature([], dtype=tf.string),
        'label0': tf.io.FixedLenFeature([], dtype=tf.int64),
        'label1': tf.io.FixedLenFeature([], dtype=tf.int64),
        'label2': tf.io.FixedLenFeature([], dtype=tf.int64),
        'label3': tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    features = tf.io.parse_single_example(example_serialized, feature_map)

    image = tf.decode_raw(features['image'], tf.uint8)

    # 转为灰度图
    # image = tf.reduce_mean(image, axis=-1)

    # label转为one_hot vector
    # label = tf.zeros([1, 40])

    label0 = tf.one_hot(features['label0'], depth=10)
    label1 = tf.one_hot(features['label1'], depth=10)
    label2 = tf.one_hot(features['label2'], depth=10)
    label3 = tf.one_hot(features['label3'], depth=10)
    
    label = tf.concat([label0, label1, label2, label3], axis=0)
    # label = tf.expand_dims(label, axis=0)

    return image, label

if __name__ == "__main__":
    
    # 判断tfrecord文件是否存在
    if _dataset_exits(TFRECORD_DIR):
        print('tfrecord has existed')

    else:
        photo_filenames = _get_filename_and_class(DATASET_DIR)
        # 把数据分割为训练集和测试集
        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        training_filename = photo_filenames[_NUM_TEST:]
        testing_filename = photo_filenames[:_NUM_TEST]

        _convert_dataset('train',training_filename,DATASET_DIR)
        _convert_dataset('test',testing_filename,DATASET_DIR)
        print('生成tfrecord文件')