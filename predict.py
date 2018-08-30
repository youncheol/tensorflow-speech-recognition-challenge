import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import logging
import csv
from tqdm import tqdm
from model import DenseNet, CnnLstm
from data_processor import TFRecord


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_fname", required=True,
                        help="file name of saved model to use")
    parser.add_argument("--tfr_fname", default="test.tfrecord",
                        help="TFRecord file name where the data to predict is stored (default: test.tfrecord)")
    parser.add_argument("--sample_fname", default="sample_submission.csv",
                        help="Kaggle sample submission file name (default: sample_submission.csv)")
    parser.add_argument("--output_fname", default="submission.csv",
                        help="output file name (default: submission.csv)")
    parser.add_argument("--densenet", default=False, action="store_true",
                        help="use DenseNet model (default: False)")
    parser.add_argument("--proba", default=False, action="store_true",
                        help="predict probabilities (default: False)")

    return parser.parse_args()


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger


def make_submission(predict, sample_fname, sub_fname, proba):
    class_names = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'silence', 'stop', 'unknown', 'up', 'yes']
    df = pd.read_csv(sample_fname)
    files = df["fname"]

    if proba:
        pp = pd.DataFrame(predict, index=files)
        pp.to_csv(sub_fname, index=False)
    else:
        with open(sub_fname, "w") as f:
            fieldnames = ["fname", "label"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(predict)):
                writer.writerow({"fname": files[i], "label": class_names[predict[i]]})


def main():
    args = get_args()
    logger = get_logger()

    logger.info(vars(args))

    with tf.device("/gpu:0"):
        if args.densenet == True:
            model = DenseNet()
        else:
            model = CnnLstm()

    tfrecord = TFRecord()
    tfrecord.make_iterator(args.tfr_fname, training=False)

    total = sum(1 for _ in tf.python_io.tf_record_iterator(args.tfr_fname)) // tfrecord.batch_size

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, args.model_fname)

        sess.run(tfrecord.init_op)

        spec = tfrecord.load(sess, training=False)
        predict = model.predict(sess, spec, args.proba)

        progress_bar = tqdm(total=total, desc="[PREDICT]", unit="batch", leave=False)

        while True:
            try:
                spec = tfrecord.load(sess, training=False)

                if args.proba:
                    predict = np.vstack([predict, model.predict(sess, spec, args.proba)])
                else:
                    predict = np.hstack([predict, model.predict(sess, spec, args.proba)])

                progress_bar.update(1)

            except tf.errors.OutOfRangeError:
                break

    make_submission(predict, args.sample_fname, args.output_fname, args.proba)
    logger.info(f"{args.output_fname} is created.")


if __name__ == "__main__":
    main()
