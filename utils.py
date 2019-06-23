import numpy as np
import librosa as lbr
import tensorflow as tf
import os

slim = tf.contrib.slim

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
        'pop', 'reggae', 'rock']

WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}

def get_default_shape(dataset_path):
    tmp_features, _ = load_track(os.path.join(dataset_path,
        'blues/blues.00000.au'))
    return tmp_features.shape


def load_track(filename, enforce_shape=None):
    new_input, sample_rate = lbr.load(filename, mono=True)
    features = lbr.feature.melspectrogram(new_input, **MEL_KWARGS).T

    if enforce_shape is not None:
        if features.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - features.shape[0],
                    enforce_shape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0)
        elif features.shape[0] > enforce_shape[0]:
            features = features[: enforce_shape[0], :]


    features[features == 0] = 1e-6
    return (np.log(features), float(new_input.shape[0]) / sample_rate)



def read_and_decode(filename): # read train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'genres': tf.FixedLenFeature((1,), tf.int64),
                                           'au_flattern': tf.VarLenFeature(tf.float32),
                                       })#return image and label


    labels = tf.cast(features['genres'], tf.int32) #throw label tensor
    au_flattern = features['au_flattern']
    au_flattern = tf.sparse_tensor_to_dense(au_flattern)
    au = tf.reshape(au_flattern, get_default_shape('GTZAN/genres'))
    au = tf.expand_dims(au, axis=2)
    return au, labels



def audio_augmention(data, sr):
    # Adding white noise
    wn = np.random.randn(len(data))
    data_wn = data + 0.005 * wn

    # Shifting the sound
    steps = np.random.randint(-10, 10)
    data_sf = lbr.effects.pitch_shift(data_wn, sr, n_steps=steps)

    # Changing volume
    volume = np.random.uniform(.5, 2)
    data_sf *= volume

    return data_sf


def load_track_with_aug(filename, enforce_shape=None):
    new_input, sample_rate = lbr.load(filename, mono=True)
    new_input_with_aug = audio_augmention(new_input, sample_rate)
    features = lbr.feature.melspectrogram(new_input_with_aug, **MEL_KWARGS).T

    if enforce_shape is not None:
        if features.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - features.shape[0],
                    enforce_shape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0)
        elif features.shape[0] > enforce_shape[0]:
            features = features[: enforce_shape[0], :]

    features[features == 0] = 1e-6
    return (np.log(features), float(new_input.shape[0]) / sample_rate)



def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "evaluate/ArgMax"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        # for op in graph.get_operations():
        #     print(op.name, op.values())

#freeze_graph('model/inception_resnet_v2_iteration_9599.ckpt', 'model/test.pb')

def print_node():
    from tensorflow.python import pywrap_tensorflow
    import os
    checkpoint_path=os.path.join('model/inception_resnet_v2_iteration_9599.ckpt')
    reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map=reader.get_variable_to_shape_map()
    # b = [b for b in var_to_shape_map if b.startswith('generator/b')]
    # b.sort()
    # w = [w for w in var_to_shape_map if w.startswith('generator/W')]
    # w.sort()
    # v = [v for v in var_to_shape_map if v.startswith('generator/V')]
    # v.sort()
    #
    # print(w)
    # print(b)
    # print(v)
    for key in var_to_shape_map:
        print('tensor_name: ', key)







