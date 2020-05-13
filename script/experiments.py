import sys
import os

from script.tools import Recorder
from model.runner import Trainer
from attrdict import AttrDict
import traceback

save_dir_root = '../save'
runs_dir_root = '../runs'


class ArgsMaker:
    """
    Class for making args
    """

    def __init__(self):
        self.default_args = {
            # data control
            'train_dataset': '../data/kitti-all-label02.csv',
            'train_leave': None,  # missing
            'relative': False,
            'total_len': 12,
            'pred_len': 4,
            # model
            'model': None,  # missing
            'embedding_size': 128,
            'cell_size': 128,
            'dropout': 0.0,
            'batch_norm': False,
            'bbox': False,
            'loss': None,  # missing
            # train args
            'batch_size': 64,
            'num_epochs': 11,  # debug!!!
            'learning_rate': 1e-3,
            'clip_threshold': 1.5,
            'validate_every': 3,  # debug!!!
            'weight_decay': 5e-5,
            # log
            'print_every': 1,
            'phase': 'train',
            'board_name': None,  # missing
            # load and save
            'save_dir': None,  # missing
            'save_every': 10,  # debug!!!
            'restore_dir': None,
            # validation
            'val_dataset': '../data/kitti-all-label02.csv',
            'val_phase': 'test',
            'val_scene': None,  # missing
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
        Add Argument Alternative Rules.
        :param arg_names: str or list(str), to identify what arguments are alternative.
        :param alternatives: list(value) or list(list(value)), corresponding to arg_names size.
        :param brief: str, for briefly represent arg_names in tag.
        """
        # make sure args_name shaped like []
        # make sure alternatives shaped like [[]]
        # make sure brief shaped like []
        if not isinstance(arg_names, list):
            arg_names = [arg_names]
            alternatives = [[x] for x in alternatives]
        self.args_names.append(arg_names)
        self.args_values.append(alternatives)
        if brief:
            self.args_name_brief.append([brief])
        else:
            self.args_name_brief.append(arg_names)

    def making_args_candidates(self):
        candidates = dict()
        self.get_args_recursively(0, self.default_args, candidates)

        attr_candidates = dict()
        for key, value in candidates.items():
            value['board_name'] = key
            value['save_dir'] = key
            attr_candidates[key] = value
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
        """
        Add block rule to black list.
        If an args configuration meeting all value in rule, then blocked.
        :param rule: dict()
        """
        self.black_list.append(rule)

    def is_blocked(self, attr):
        """
        For a generated args configuration, check it with rule one by one.
        :param attr: generated args
        :return: true/false
        """
        for rule in self.black_list:
            meet_args = [key for key, value in rule.items() if attr[key] == value]
            if len(meet_args) == len(rule.keys()):
                return True
        return False


class TaskRunner:
    """
    A runner to train/validate model. Just like running a train script.
    """

    def __init__(self, prefix, task_attr):
        """
        :param prefix: An identifier for adding to the front of args.board_name and args.save_dir.
                    Better used to identify different experiment batches.
        :param task_attr: train arguments for passing into model runner.
        """
        # transform from dict() to AttrDict
        task_attr = AttrDict(task_attr)
        self.task_attr = task_attr
        # add prefix to board_name and save_dir
        task_attr.board_name = os.path.join(runs_dir_root, prefix, task_attr.board_name)
        task_attr.save_dir = os.path.join(save_dir_root, prefix, task_attr.save_dir)
        # make dir for save dir
        if not os.path.exists(os.path.join(save_dir_root, prefix)):
            os.mkdir(os.path.join(save_dir_root, prefix))
        if not os.path.exists(task_attr.save_dir):
            os.mkdir(task_attr.save_dir)
        # initial recorder and trainer
        self.recorder = Recorder(summary_path=task_attr.board_name, logfile=True)
        self.trainer = Trainer(task_attr, self.recorder)

    def run(self, global_recorder):
        try:
            self.recorder.logger.info(self.task_attr)
            self.trainer.train_model()

            global_recorder.logger.info('Task Ends Successfully. Save Dir = {}'.format(self.task_attr.save_dir))
            self.recorder.logger.info('Task Ends Successfully. Save Dir = {}'.format(self.task_attr.save_dir))
        except Exception as _:
            global_recorder.logger.info(
                'Task Ends With Error, Details On Log. Save Dir = {}'.format(self.task_attr.save_dir))
            global_recorder.logger.info(traceback.format_exc())

            self.recorder.logger.info('Task Ends With Error. Save Dir = {}'.format(self.task_attr.save_dir))
            self.recorder.logger.info(traceback.format_exc())
        self.recorder.close()


if __name__ == '__main__':
    # How to use above three class.
    # 1. use ArgsMaker to make a list of args in a batch experiment.
    # 2. use ArgsBlocker to abandon specified args by ArgsMaker.
    # 3. deploy a task runner to run training.
    #    a. Use prefix to identify different batch experiments.
    #    b. All data in one batch experiment will be in stored in runs/prefix/ and save/prefix/
    # experiment prefix
    prefix = '0513'
    log_file = Recorder(os.path.join(runs_dir_root, prefix), board=False, logfile=False)
    # 添加生成参数的规则
    argsMaker = ArgsMaker()
    argsMaker.add_arg_rule('model', ['seq2seq', 'vanilla'])
    argsMaker.add_arg_rule('loss', ['2d_gaussian', 'mixed'])
    argsMaker.add_arg_rule(['train_leave', 'val_scene'], [([i], [i]) for i in [4, 13, 16, 17]], 'scene')
    argsMaker.add_arg_rule(['use_sample', 'sample_times'], [[False, 1]] + [[True, i] for i in [5, 10, 20]], 'sample')

    blocker = ArgsBlocker()
    blocker.add_block_rule({'loss': 'mixed', 'use_sample': True})
    candidates = argsMaker.making_args_candidates().items()
    for index, item in enumerate(candidates):
        if blocker.is_blocked(item[1]):
            log_file.logger.info('{}/{} blocked '.format(index, len(candidates) - 1) + str(item[0]))
        else:
            log_file.logger.info('{}/{} pass '.format(index, len(candidates) - 1) + str(item[0]))
            task_runner = TaskRunner(prefix, item[1])
            task_runner.run(log_file)
    log_file.close()
