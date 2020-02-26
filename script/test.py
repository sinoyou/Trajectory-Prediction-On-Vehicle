import argparse
import sys
import os

from script.tools import Recorder
from model.runner import Tester

sys.path.append('../')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo not recommended

# args parser
parser = argparse.ArgumentParser()


def test():
    # model
    parser.add_argument('--model', type=str)
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--obs_len', default=10, type=int)
    parser.add_argument('--pred_len', default=5, type=int)
    parser.add_argument('--use_sample', default=False, type=bool)
    parser.add_argument('--sample_times', default=20, type=int)

    # dataset
    parser.add_argument('--test_dataset', type=str)
    parser.add_argument('--relative', default=False, type=bool)

    # save
    parser.add_argument('--export_path', default=None, type=str)

    # print
    parser.add_argument('--silence', default=False, type=bool, help='Silent mode, only print global average result.')
    parser.add_argument('--plot', default=False, type=bool, help='plot trajectory on the tensor board.')

    args = parser.parse_args()
    recorder = Recorder()
    tester = Tester(args, recorder)
    tester.evaluate()


if __name__ == '__main__':
    test()
