from model import inception_resnet_v2
import tensorflow as tf
from utils import  read_and_decode
import numpy as np
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -  %(filename)s - %(funcName)s: %(lineno)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



model_path = './models/'
test_data = './tfrecords/test.tfrecords'

tf.reset_default_graph()

au_test, label_test = read_and_decode(test_data)

au_test_batch, label_test_batch = tf.train.shuffle_batch([au_test, label_test],
                                                            batch_size=1,
                                                            num_threads=16,
                                                            capacity=800 + 3,
                                                            min_after_dequeue=800,
                                                            )

input_ = tf.placeholder(tf.float32, [None, 647, 128, 1])

logits_, _ = inception_resnet_v2(input_, is_training=False, dropout_keep_prob=1, create_aux_logits=False)

with tf.Session() as sess:

    saver = tf.train.Saver()

    saver.restore(sess, "models/inception_resnet_v2_iteration_9999.ckpt")

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(coord=coord)

    N = 1000
    top1 = 0
    top3 = 0
    for i in tqdm(range(N)):
        data, labels = sess.run([au_test_batch, label_test_batch])

        logits = sess.run(logits_, feed_dict={input_: data}).ravel()

        max_index = np.argsort(-logits)

        predict = np.argmax(logits)
        if predict == int(labels):
            top1 += 1
        if int(labels) in max_index[:3]:
            top3 += 1

    logging.info("top1: {:.2f}%".format(top1 / N * 100))
    logging.info("top3: {:.2f}%".format(top3 / N * 100))

    coord.request_stop()
    coord.join(threads)







