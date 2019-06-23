import tensorflow as tf
from utils import read_and_decode
from model import inception_resnet_v2
import argparse
from tqdm import tqdm
import os
import logging

slim = tf.contrib.slim
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -  %(filename)s - %(funcName)s: %(lineno)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train(train_data_,
          decay_rate_,
          global_steps_,
          decay_steps_,
          batch_size_,
          learning_rate_,
          eval_step_,
          model_path_,
          summary_path_,
          load_model_):

    if not os.path.exists(model_path_):
        os.mkdir(model_path_)

    if not os.path.exists(summary_path_):
        os.mkdir(summary_path_)

    graph = tf.Graph()

    with graph.as_default():

        au_train, label_train = read_and_decode(train_data_)

        min_fraction_of_examples_in_queue = 0.4
        test_size = .2
        aug = 10
        total_examples = 1000 * aug
        min_queue_examples_train = int(total_examples * (1 - test_size) * min_fraction_of_examples_in_queue)

        au_train_batch, label_train_batch = tf.train.shuffle_batch([au_train, label_train],
                                                        batch_size=batch_size_,
                                                        num_threads=16,
                                                        capacity=min_queue_examples_train + 3 * batch_size_,
                                                        min_after_dequeue=min_queue_examples_train,
                                                        )

        label_train_batch_ = tf.one_hot(tf.squeeze(label_train_batch), 10, 1, 0)

        logits, end_points = inception_resnet_v2(au_train_batch)

        if 'AuxLogits' in end_points:
            slim.losses.softmax_cross_entropy(
                end_points['AuxLogits'], label_train_batch_, weights=0.4, scope='aux_loss')

        slim.losses.softmax_cross_entropy(
            logits, label_train_batch_, weights=1.0, scope='base_loss')

        total_loss = slim.losses.get_total_loss()

        tf.summary.scalar('loss', total_loss)

        global_ = tf.Variable(tf.constant(0), trainable=False)

        lr = tf.train.exponential_decay(learning_rate_, global_, decay_steps_, decay_rate_, staircase=True)

        tf.summary.scalar('lr', lr)

        optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label_train_batch_, 1))

        accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', accuracy_)

        saver = tf.train.Saver(max_to_keep=5)

        merged = tf.summary.merge_all()

        writer = tf.summary.FileWriter(summary_path_, graph=graph)

        init = tf.global_variables_initializer()

        with tf.Session() as sess, open('log/train_log.log', 'w') as log:

            if load_model_:
                checkpoint = tf.train.get_checkpoint_state(model_path_)

                meta_graph_path = checkpoint.model_checkpoint_path + ".meta"

                restore = tf.train.import_meta_graph(meta_graph_path)

                restore.restore(sess, tf.train.latest_checkpoint(model_path_))

                step = int(meta_graph_path.split("_")[-1].split(".")[0])
            else:
                sess.run(init)
                step = 0

            coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(coord=coord)
            try:
                for i in tqdm(range(step, global_steps_)):

                    acc, loss, train_summary, _ = sess.run([accuracy_, total_loss, merged, optimizer],
                                                           feed_dict={global_: i})
                    print("steps:{} train loss :{:.2f}, accuracy: {:.2f}".format(i, loss, acc), file=log)
                    log.flush()

                    if (i + 1) % eval_step_ == 0:

                        saver.save(sess, '{}/inception_resnet_v2_iteration_{}.ckpt'.format(model_path_, i))
                        writer.add_summary(train_summary, i)

            except KeyboardInterrupt:
                logger.exception('Interrupted')
                coord.request_stop()
            except Exception as e:
                logger.exception(e)
                coord.request_stop(e)
            finally:
                logger.info("Model saved in file: %s" % model_path_)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data',
        type=str,
        default='./tfrecords/train.tfrecords',
        help='train_data path.'
    )

    parser.add_argument(
        '--decay_rate',
        type=float,
        default=0.9,
        help='learning rate decay rate.'
    )

    parser.add_argument(
        '--global_steps',
        type=int,
        default=10000,
        help='global steps'
    )

    parser.add_argument(
        '--decay_steps',
        type=int,
        default=100,
        help='learning rate decay steps.'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='learning rate.'
    )
    parser.add_argument(
        '--eval_step',
        type=int,
        default=100,
        help='evaluation steps.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='batch size.'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='models/',
        help='tensorflow model path.'
    )

    parser.add_argument(
        '--summary_path',
        type=str,
        default='summary/',
        help='tensorflow summary path.'
    )

    parser.add_argument(
        '--load_model',
        type=bool,
        default=False,
        help='whether you wish to continue training.'
    )

    args = parser.parse_args()

    train_data = args.train_data
    decay_rate = args.decay_rate
    global_steps = args.global_steps  # 总的迭代次数
    decay_steps = args.decay_steps  # 衰减次数
    learning_rate = args.learning_rate
    eval_step = args.eval_step
    summary_path = args.summary_path
    model_path = args.model_path
    load_model = args.load_model
    batch_size = args.batch_size

    logger.info('\nThe following parameters will be applied for data creating:\n')
    logger.info('train_data path: {}.'.format(train_data))
    logger.info("learning rate decay rate: {}".format(decay_rate))
    logger.info("global steps: {}".format(global_steps))
    logger.info("learning rate decay steps: {} .".format(decay_steps))
    logger.info('batch size: {}.'.format(batch_size))
    logger.info('learning rate: {}.'.format(learning_rate))
    logger.info('evaluation steps: {}.'.format(eval_step))
    logger.info('tensorflow model path: {}.'.format(model_path))
    logger.info('tensorflow summary path: {}.'.format(summary_path))
    logger.info('whether you wish to continue training: {}.'.format(load_model))

    train(train_data,
          decay_rate,
          global_steps,
          decay_steps,
          batch_size,
          learning_rate,
          eval_step,
          model_path,
          summary_path,
          load_model)

