import argparse

from src.train import setup as train_setup
from src.eval_Mag import setup as eval_Mag_setup
from src.eval_Complex import setup as eval_Complex_setup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    train_setup(subparsers.add_parser("train"))
    eval_Mag_setup(subparsers.add_parser("eval_Mag"))
    eval_Complex_setup(subparsers.add_parser("eval_Complex"))
    args = parser.parse_args()
    args.func(args)
