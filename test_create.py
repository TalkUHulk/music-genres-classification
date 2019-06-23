import os
import tensorflow as tf
import cv2 as cv
from utils import load_track

# writer = tf.python_io.TFRecordWriter("train.tfrecords")
# path = '/Users/wangdong/Documents/DPED-master/dped/dped/blackberry/test_data/full_size_test_images/'
# for index in range(29):
#     img_name = path + '{}.jpg'.format(index)
#
#     img = cv.imread(img_name)
#     img = cv.resize(img, (224, 224))
#     img = img.flatten()
#
#     img = img / 256
#     print(img.shape)
#     #img_raw = img.tobytes()              #将图片转化为原生bytes
#     example = tf.train.Example(features=tf.train.Features(feature={
#             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#             'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=img))
#         }))
#     writer.write(example.SerializeToString())  #序列化为字符串
# writer.close()

writer = tf.python_io.TFRecordWriter("train.tfrecords")
tmp_features, _ = load_track(os.path.join('GTZAN/genres', 'blues/blues.00000.au'))
print(tmp_features.shape)
data = tmp_features.flatten()

print(data)
#img_raw = img.tobytes()              #将图片转化为原生bytes
example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[12])),
        'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=data))
    }))
writer.write(example.SerializeToString())  #序列化为字符串
writer.close()