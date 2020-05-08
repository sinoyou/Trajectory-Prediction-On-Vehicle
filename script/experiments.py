import sys
import os

from script.tools import Recorder
from model.runner import Trainer
from attrdict import AttrDict


class ArgsMaker:
    """
    Class for making args
    """

    def __init__(self):
        self.default_args = {
            # data control
            'train_dataset': '../data/kitti-all-label02.csv',
            'leave_scene': None,
            'relative': False,
            'total_len': 12,
            'pred_len': 8,
            # model
            'model': None,
            'embedding_size': 128,
            'cell_size': 128,
            'dropout': 0.0,
            'batch_norm': False,
            'bbox': False,
            'loss': None,
            # train args
            'batch_size': 64,
            'num_epochs': 301,
            'learning_rate': 1e-3,
            'clip_threshold': 1.5,
            'validate_every': 30,
            'weight_decay': 5e-5,
            # log
            'print_every': 1,
            'phase': 'train',
            'board_name': None,
            # load and save
            'save_dir': None,
            'save_every': 30,
            'restore_dir': None,
            # validation
            'val_dataset': '../data/kitti-all-label02.csv',
            'val_phase': 'test',
            'val_scene': None,
            'val_obs_len': 8,
            'val_pred_len': 4,
            'val_use_sample': False,
            'val_sample_times': 10,
            'val_plot': False,
            'val_plot_mode': 0
        }
        self.args_names = list()
        self.args_name_brief = list()
        self.args_values = list()

    def add_arg_rule(self, arg_names, alternatives, brief=None):
        """
        Add Arguments Alternative Rules.
        :param arg_names: str or list(str), to identify what arguments are alternative.
        :param alternatives: list(value) or list(list(value)), corresponding to arg_names size.
        :param brief: str, for briefly represent arg_names in tag.
        """
        # make sure args_name shaped like []
        # make sure alternatives shaped like [[]]
        # make sure brief shaped like []
        if not isinstance(arg_names, list):
            args_name = [arg_names]
            alternatives = [[x] for x in alternatives]
        self.args_names.append(arg_names)
        self.args_values.append(alternatives)
        if brief:
            self.args_name_brief.append([brief])
        else:
            self.args_name_brief.append(args_name)

    def making_args_candidates(self):
        candidates = dict()
        self.get_args_recursively(0, self.default_args, candidates)

        attr_candidates = dict()
        for key, value in candidates.items():
            attr_candidates[key] = AttrDict(value)
        return attr_candidates

    def get_args_recursively(self, level, args, candidates):
        """
        Generate Args With Recursive Trees.
        :param level: tree depth.
        :param args: args in generation.
        :param candidates: global container.
        """
        # If to the bottom
        if level >= len(self.args_names):
            tag = ''
            for names, bnames in zip(self.args_names, self.args_name_brief):
                for bname in bnames:
                    if tag == '':
                        tag = bname
                    else:
                        tag = tag + '_' + bname
                for name in names:
                    tag = tag + '_{}'.format(args[name])
            candidates[tag] = args.copy()
            return

        # Else continue
        # HINT: In one rule, names could be a list of names,
        # and values are multiple candidates for name groups.
        names, value_groups = self.args_names[level], self.args_values[level]
        for values in value_groups:
            new_args = args.copy()
            for name, value in zip(names, values):
                new_args[name] = value
            self.get_args_recursively(level + 1, new_args, candidates)


class ArgsBlocker:
    """
    Maintain rules of blocking invalid arguments combination.
    """

    def __init__(self):
        self.black_list = list()

    def add_block_rule(self, rule):
        pass

    def is_blocked(self, attr):
        self.black_list = None
        return False


class TaskRunner:
    """
    A runner to train/validate model. Just like running a train script.
    """

    def __init__(self, prefix, task_attr):
        """
        :param prefix: An identifier for adding to the front of args.board_name.
                    Better used to identify different experiment batches.
        :param task_attr: train arguments for passing into model runner.
        """
        pass

    def run(self):
        pass


if __name__ == '__main__':
    argsMaker = ArgsMaker()
    argsMaker.add_arg_rule('val_sample_times', [5, 10, 20, 30], 'sample')
    argsMaker.add_arg_rule(['leave_scene', 'val_scene'], [[8, 8], [9, 9]], 'scene')
    for item in argsMaker.making_args_candidates():
        print(item)
