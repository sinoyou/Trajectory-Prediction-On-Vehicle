import argparse
import sys

sys.path.append('../')
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo not recommended

from script.tools import Recorder
from model.runner import Tester

# args parser
parser = argparse.ArgumentParser()


def test():
    # model
    parser.add_argument('--model', type=str)
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--obs_len', default=10, type=int)
    parser.add_argument('--pred_len', default=5, type=int)

    # dataset
    parser.add_argument('--test_dataset', type=str)

    # save
    parser.add_argument('--export_path', default=None, type=str)

    args = parser.parse_args()
    recorder = Recorder()
    tester = Tester(args, recorder)
    tester.evaluate()


if __name__ == '__main__':
    test()
