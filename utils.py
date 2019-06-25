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











