import argparse
import sys

sys.path.append('../')

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # not recommended for macOS

from script.tools import Recorder
from model.kf_runner import KF

# args parser
parser = argparse.ArgumentParser()


def run_kf():
    # !!!FINE-TUNE FIRST!!!
    model_full_name = 'kf_1_debug'
    obs_len = 3
    pred_len = 1

    # model
    parser.add_argument('--model', default='kf', type=str)
    parser.add_argument('--phase', default='test', type=str)
    parser.add_argument('--obs_len', default=obs_len, type=int)
    parser.add_argument('--pred_len', default=pred_len, type=int)
    parser.add_argument('--sample_times', default=1, type=int)
    parser.add_argument('--use_sample', default=False, type=bool)

    # dataset
    # parser.add_argument('--test_dataset', default='../data/final/kitti-test.json', type=str)
    parser.add_argument('--test_dataset', default='../data/kitti-all-label02.csv', type=str)

    # save
    parser.add_argument('--export_path', default=None, type=str)

    # print
    parser.add_argument('--silence', default=False, type=bool, help='Silent mode, only print global average result.')
    parser.add_argument('--plot', default=False, type=bool, help='plot trajectory on the tensor board.')
    parser.add_argument('--plot_mode', default=6, type=int)

    args = parser.parse_args()
    recorder = Recorder(summary_path=model_full_name)
    tester = KF(args, recorder)
    tester.predict_then_evaluate()
    recorder.close()


if __name__ == '__main__':
    run_kf()
