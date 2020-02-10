import argparse


from script.tools import Recorder
from model.runner import Runner


# args parser
parser = argparse.ArgumentParser()


def train(args, recorder):
    pass


def run():
    # dataset control
    parser.add_argument('--train_dataset', type=str)
    parser.add_argument('--val_dataset', type=str)
    parser.add_argument('--total_len', default=12, type=int, help='Total length of trajectory when forward once.')
    parser.add_argument('--pred_len', default=4, type=int, help='Length of trajectory participated in loss calc.')

    # model type
    parser.add_argument('--model', type=str)

    # model arguments
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--cell_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--batch_norm', default=True, type=bool)
    parser.add_argument('--bbox', default=False, type=bool)

    # train arguments
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--clip_threshold', default=1.5, type=float)
    parser.add_argument('--validate_every', default=100, type=int)
    parser.add_argument('--weight_decay', default=0.99, type=float)

    # log
    parser.add_argument('--print_every', default=10, type=int)
    parser.add_argument('--board_dir', default='runs/', type=str)

    # load and save
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--save_every', default=100, type=int)
    parser.add_argument('--restore_dir', default=None, type=str)

    args = parser.parse_args()
    recoder = Recorder(args.board_dir)
    recoder.logger.info(args)
    train(args, recoder)


if __name__ == '__main__':
    run()