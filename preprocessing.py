from data_processor import PreProcessing
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--tfr_fname", default="train.tfrecord",
                        help="file name of TFRecord to be created (default: train.tfrecord)")
    parser.add_argument("--train", default=False, action="store_true",
                        help="use training data (default: False)")

    return parser.parse_args()


def main():
    args = get_args()

    prep = PreProcessing(args.data_path)
    prep.processing(args.tfr_fname, args.train)


if __name__ == "__main__":
    main()
