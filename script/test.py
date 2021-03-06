import argparse
import sys
import os

from script.tools import Recorder
from model.runner import Tester

sys.path.append('../')
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo not recommended

# args parser
parser = argparse.ArgumentParser()


def test():
    # model
    parser.add_argument('--model', type=str)
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=4, type=int)
    parser.add_argument('--sample_times', default=0, type=int)

    # dataset
    parser.add_argument('--test_dataset', type=str)
    parser.add_argument('--train_leave', nargs='+', type=int)
    parser.add_argument('--test_scene', nargs='+', type=int)
    # scenes are used for validation. call in cmd like this: --test_scene 8
    parser.add_argument('--relative', default=False, type=bool)

    # save
    parser.add_argument('--export_path', default=None, type=str)

    # log
    parser.add_argument('--phase', default='test', type=str)
    parser.add_argument('--board_name', default='../runs', type=str)

    # print
    parser.add_argument('--silence', default=False, type=bool, help='Silent mode, only print global average result.')
    parser.add_argument('--plot', default=False, type=bool, help='plot trajectory on the tensor board.')
    parser.add_argument('--plot_mode', default=1, type=int)

    # transform test_scene from string to list()
    args = parser.parse_args()
    recorder = Recorder(args.board_name)
    tester = Tester(args, recorder)
    tester.evaluate()
    recorder.close()


if __name__ == '__main__':
    test()
