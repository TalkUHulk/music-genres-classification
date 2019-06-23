import tensorflow as tf
import numpy as np
import cv2 as cv
from utils import get_record_dataset
slim = tf.contrib.slim


# for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
#     example = tf.train.Example()
#     example.ParseFromString(serialized_example)
#
#     image = example.features.feature['img_raw'].float_list.value
#     label = example.features.feature['label'].int64_list.value
#     # 可以做一些预处理之类的
#     print(image, label)

# def read_and_decode(filename):
#     # 根据文件名生成一个队列
#     filename_queue = tf.train.string_input_producer([filename])
#
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
#     features = tf.parse_single_example(serialized_example,
#                                      features={
#                                        'label': tf.FixedLenFeature([], tf.int64),
#                                        'img_raw': tf.VarLenFeature(tf.float32),
#                                      })
#
#     #img = tf.decode_raw(features['img_raw'], tf.float32)
#     img = features['img_raw']
#     img = tf.sparse_tensor_to_dense(img)
#     img = tf.reshape(img, [224, 224, 3])
#     #img = tf.cast(tf.reshape(img, [224, 224, 3]) * 255, tf.uint8)
#     img = tf.cast(img * 255, tf.uint8)
#     label = tf.cast(features['label'], tf.int32)
#
#     return img, label
#
#
#
# img, label = read_and_decode("train.tfrecords")
#
# #使用shuffle_batch可以随机打乱输入
# img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                 batch_size=1, capacity=10,
#                                                 min_after_dequeue=2)
# init = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     sess.run(init)
#     threads = tf.train.start_queue_runners(sess=sess)
#     for i in range(2):
#         val, l = sess.run([img_batch, label_batch])
#         val = np.reshape(val, (224, 224, 3))
#         cv.imshow('1', val)
#         cv.waitKey()
#         print(val.shape, l)


# def read_and_decode(filename):
#     # 根据文件名生成一个队列
#     filename_queue = tf.train.string_input_producer([filename])
#
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
#     features = tf.parse_single_example(serialized_example,
#                                      features={
#                                        'label': tf.FixedLenFeature([], tf.int64),
#                                        'img_raw': tf.VarLenFeature(tf.float32),
#                                      })
#
#     #img = tf.decode_raw(features['img_raw'], tf.float32)
#     img = features['img_raw']
#     img = tf.sparse_tensor_to_dense(img)
#     img = tf.reshape(img, [647,128])
#     #img = tf.cast(tf.reshape(img, [224, 224, 3]) * 255, tf.uint8)
#     #img = tf.cast(img * 255, tf.uint8)
#     label = tf.cast(features['label'], tf.int32)
#
#     return img, label
#
#
#
# img, label = read_and_decode("train.tfrecords")
#
# #使用shuffle_batch可以随机打乱输入
# img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                 batch_size=1, capacity=10,
#                                                 min_after_dequeue=2)
# init = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     sess.run(init)
#     threads = tf.train.start_queue_runners(sess=sess)
#     for i in range(2):
#         val, l = sess.run([img_batch, label_batch])
#         print(val, l)


dataset = get_record_dataset('tfrecords/data.tfrecords', num_samples=1000,
                             num_classes=10)
data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
label,image = data_provider.get(['genres', 'au_flattern'])

inputs, labels = tf.train.batch([image, label],
                                batch_size=1,
                                # capacity=5*FLAGS.batch_size,
                                allow_smaller_final_batch=True)

# 输出当前tensor的静态shape 和动态shape，与另一种读取方式进行对比

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(2):
        val, l = sess.run([inputs, labels])
        print(val, l)