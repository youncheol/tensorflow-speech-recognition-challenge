from data_processor import PreProcessing
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_fname", required=True)
    parser.add_argument("--training", default=False, action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    prep = PreProcessing(args.data_path)
    prep.processing(args.output_fname, args.training)


if __name__ == "__main__":
    main()

