import argparse
import sys
import os

from script.tools import Recorder
from model.runner import Trainer

sys.path.append('../')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # not recommended

# args parser
parser = argparse.ArgumentParser()


def train(args, recorder):
    trainer = Trainer(args=args, recorder=recorder)
    trainer.train_model()


def run():
    # dataset control
    parser.add_argument('--train_dataset', type=str)
    parser.add_argument('--train_leave', nargs='+', type=int)
    # scenes are leaved and don't train. call in cmd like this: --leave_scene 1 2 3 4
    parser.add_argument('--relative', default=False, type=bool, help='Use relative trajectory instead of relative')
    parser.add_argument('--total_len', default=12, type=int, help='Total length of trajectory when forward once.')
    parser.add_argument('--pred_len', default=4, type=int, help='Length of trajectory participated in loss calc.')

    # model type
    parser.add_argument('--model', type=str)

    # model arguments
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--cell_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--batch_norm', default=False, type=bool)
    parser.add_argument('--bbox', default=False, type=bool)
    parser.add_argument('--loss', type=str, help='Supporting : 2d_gaussian, mixed')

    # train arguments
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=301, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--clip_threshold', default=1.5, type=float)
    parser.add_argument('--validate_every', default=30, type=int)
    parser.add_argument('--weight_decay', default=5e-5, type=float)

    # log
    parser.add_argument('--print_every', default=1, type=int)
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--board_name', default='../runs', type=str)

    # load and save
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--save_every', default=30, type=int)
    parser.add_argument('--restore_dir', default=None, type=str)

    # validation arguments
    parser.add_argument('--val_dataset', type=str)
    parser.add_argument('--val_phase', default='test', type=str)
    parser.add_argument('--val_scene', nargs='+', type=int)
    # scenes are used for validation. call in cmd like this: --val_scene 8
    parser.add_argument('--val_obs_len', type=int)
    parser.add_argument('--val_pred_len', type=int)
    parser.add_argument('--val_sample_times', nargs='+', type=int)
    parser.add_argument('--val_plot', default=False, type=bool)
    parser.add_argument('--val_plot_mode', default=1, type=int)

    # transform leave scene & val scene from string to list
    args = parser.parse_args()

    recorder = Recorder(summary_path=args.board_name)
    recorder.logger.info(args)
    train(args, recorder)
    recorder.close()


if __name__ == '__main__':
    run()
