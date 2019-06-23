# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import argparse
from utils import GENRES, load_track, get_default_shape, load_track_with_aug
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -  %(filename)s - %(funcName)s: %(lineno)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

each_genres_num = 100
genres_num = len(GENRES)

def create_tfrecords_default(data_path, train_path, test_path, test_size=.2, aug=10):
    writer_train = tf.python_io.TFRecordWriter(train_path) #输出成tfrecord文件
    writer_test = tf.python_io.TFRecordWriter(test_path)  # 输出成tfrecord文件

    test_size_per_genres = int(each_genres_num * test_size)

    default_shape = get_default_shape(data_path) # (647, 128)
    total_data = genres_num * aug * each_genres_num
    with tqdm(desc='creating===>>', total=total_data) as pbar:
        for index, name in enumerate(GENRES):
            audio_list = [os.path.join(data_path, name + '/' + audio) for audio in os.listdir(os.path.join(data_path, name))
                          if audio.endswith('au')]

            np.random.shuffle(audio_list)
            train_data = audio_list[:-test_size_per_genres]
            test_data = audio_list[-test_size_per_genres:]

            # train data
            for _, audio in enumerate(train_data):

                au, _ = load_track(audio, default_shape)

                au_flatten = au.flatten()

                example = tf.train.Example(features=tf.train.Features(feature={
                    "genres": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'au_flattern': tf.train.Feature(float_list=tf.train.FloatList(value=au_flatten))
                }))
                writer_train.write(example.SerializeToString())  #序列化为字符串

                pbar.update(aug * each_genres_num * (1 - test_size) / len(train_data) / aug)

                # data augmentation
                if aug > 1:
                    for i in range(aug - 1):

                        au_aug, _ = load_track_with_aug(audio, default_shape)

                        au_flatten_aug = au_aug.flatten()

                        example_aug = tf.train.Example(features=tf.train.Features(feature={
                            "genres": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                            'au_flattern': tf.train.Feature(float_list=tf.train.FloatList(value=au_flatten_aug))
                        }))
                        writer_train.write(example_aug.SerializeToString())  # 序列化为字符串

                        pbar.update(aug * each_genres_num * (1 - test_size) / len(train_data) / aug)




            # test data
            for _, audio in enumerate(test_data):

                au, _ = load_track(audio, default_shape)

                au_flatten = au.flatten()

                example = tf.train.Example(features=tf.train.Features(feature={
                    "genres": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'au_flattern': tf.train.Feature(float_list=tf.train.FloatList(value=au_flatten))
                }))
                writer_test.write(example.SerializeToString())  #序列化为字符串

                pbar.update(aug * each_genres_num * test_size / len(test_data) / aug)

                # data augmentation
                if aug > 1:
                    for i in range(aug - 1):
                        au_aug, _ = load_track_with_aug(audio, default_shape)

                        au_flatten_aug = au_aug.flatten()

                        example_aug = tf.train.Example(features=tf.train.Features(feature={
                            "genres": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                            'au_flattern': tf.train.Feature(float_list=tf.train.FloatList(value=au_flatten_aug))
                        }))
                        writer_test.write(example_aug.SerializeToString())  # 序列化为字符串

                        pbar.update(aug * each_genres_num * test_size / len(test_data) / aug)

    writer_train.close()
    writer_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='GTZAN/genres',
        help='data_sets path.'
    )

    parser.add_argument(
        '--train_path',
        type=str,
        default='tfrecords/train.tfrecords',
        help='train tfrecords save path.'
    )

    parser.add_argument(
        '--test_path',
        type=str,
        default='tfrecords/test.tfrecords',
        help='test tfrecords save path.'
    )

    parser.add_argument(
        '--test_size',
        type=float,
        default=.2,
        help='Proportion of test data that between [0, 1]'
    )

    parser.add_argument(
        '--aug',
        type=int,
        default=10,
        help='the size of data sets up to (arg) times as original. 1 means not augmentation.'
    )
    args = parser.parse_args()
    test_size = args.test_size
    data_path = args.data_path
    train_path = args.train_path
    test_path = args.test_path
    aug = max(args.aug, 1)

    logger.info('\nThe following parameters will be applied for data creating:\n')
    logger.info("data_sets path: {}".format(data_path))
    logger.info("train tfrecords save path: {}".format(train_path))
    logger.info("test tfrecords save path: {}".format(test_path))
    logger.info("Proportion of test data: {}".format(test_size))
    logger.info("the size of data sets up to {} times as original.".format(aug))

    create_tfrecords_default(data_path, train_path, test_path, test_size, aug)
