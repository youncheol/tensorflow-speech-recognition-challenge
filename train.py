import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
import argparse
import logging
from tqdm import tqdm
from data_processor import TFRecord
from model import DenseNet, CnnLstm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfr_fname", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--save_fname", required=True)
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs (defalut: 5)")
    parser.add_argument("--densenet", default=False, action="store_true",
                        help="Train DenseNet model (default: False)")

    return parser.parse_args()


def get_logger(logdir):
    try:
        os.mkdir(logdir)
    except FileExistsError:
        pass

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(logdir + "/" + "log.txt")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def main():
    args = get_args()
    logger = get_logger(args.logdir)

    logger.info(vars(args))

    with tf.device("/gpu:0"):
        if args.densenet:
            model = DenseNet()
        else:
            model = CnnLstm()

    tfrecord = TFRecord()
    tfrecord.make_iterator(args.tfr_fname)

    total = sum(1 for _ in tf.python_io.tf_record_iterator(args.tfr_fname)) // tfrecord.batch_size

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(args.num_epochs):
            logger.info(f"Epoch {epoch+1}")

            sess.run(tfrecord.init_op)
            loss_list = []

            progress_bar = tqdm(total=total, desc="[TRAIN] Loss: 0", unit="batch", leave=False)

            while True:
                try:
                    step = sess.run(model.global_step)

                    spec, label = tfrecord.load(sess, training=True)

                    _, loss, merged = model.train(sess, spec, label)

                    progress_bar.update(1)
                    progress_bar.set_description(f"[TRAIN] Batch Loss: {loss:.4f}")

                    loss_list.append(loss)

                    writer.add_summary(summary=merged, global_step=step)

                except tf.errors.OutOfRangeError:
                    break

            progress_bar.close()

            mean_loss = np.mean(loss_list)
            logger.info(f"  -  [TRAIN] Mean Loss: {mean_loss:.4f}")

            saver.save(sess, args.logdir + "/" + args.save_fname + ".ckpt", global_step=sess.run(model.global_step))

if __name__ == "__main__":
    main()




