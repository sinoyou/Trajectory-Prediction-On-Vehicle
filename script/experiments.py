import sys
import os

from script.tools import Recorder
from model.runner import Trainer
from attrdict import AttrDict
import traceback
import tqdm
import pandas as pd

save_dir_root = '../save'
runs_dir_root = '../runs'
cross_weights = {0: 0.014705882352941176,
                 1: 0.014705882352941176,
                 2: 0.00980392156862745,
                 3: 0.0,
                 4: 0.04411764705882353,
                 5: 0.004901960784313725,
                 6: 0.0,
                 7: 0.00980392156862745,
                 8: 0.0,
                 9: 0.004901960784313725,
                 10: 0.014705882352941176,
                 11: 0.024509803921568627,
                 12: 0.00980392156862745,
                 13: 0.24509803921568626,
                 14: 0.00980392156862745,
                 15: 0.0784313725490196,
                 16: 0.11764705882352941,
                 17: 0.05392156862745098,
                 18: 0.0,
                 19: 0.3431372549019608,
                 20: 0.0
                 }
cross_scene = [19, 17, 16, 15, 14, 13, 12, 11, 10, 9, 7, 5, 4, 2, 1, 0]
cross_metrics = ['min_loss', 'min_first_loss', 'min_final_loss',
                 'min_ade', 'min_fde', 'min_ade_x', 'min_ade_y', 'min_fde_x', 'min_fde_y',
                 'min_l2', 'min_final_l2',
                 'min_nll', 'min_first_nll', 'min_final_nll',
                 'min_nll_x', 'min_nll_y', 'min_first_nll_x', 'min_first_nll_y', 'min_final_nll_x', 'min_final_nll_y']
cross_metrics = cross_metrics + ['best_' + k for k in cross_metrics]

# only_eval_model_name = 'latest_checkpoint.ckpt'
only_eval_model_name = 'temp_checkpoint_val.ckpt'


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
            'total_len': 14,
            'pred_len': 8,
            # model
            'model': None,  # missing
            'embedding_size': 64,
            'cell_size': 128,
            'dropout': 0.0,
            'batch_norm': False,
            'bbox': False,
            'loss': None,  # missing
            # train args
            'batch_size': 256,
            'num_epochs': 401,  # debug!!!
            'learning_rate': 1e-3,
            'clip_threshold': 1.5,
            'validate_every': 40,  # debug!!!
            'weight_decay': 5e-5,
            # log
            'print_every': 1,
            'phase': 'train',
            'board_name': None,  # missing
            # load and save
            'save_dir': None,  # missing
            'save_every': 40,  # debug!!!
            'restore_dir': None,
            # validation
            'val_dataset': '../data/kitti-all-label02.csv',
            'val_phase': 'test',
            'val_scene': None,  # missing
            'val_obs_len': 6,
            'val_pred_len': 8,
            'val_sample_times': None,
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
                tag = tag.replace(' ', '')
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
        If an args configuration meeting rule, then blocked.
        :param rule: func()
        """
        self.black_list.append(rule)

    def is_blocked(self, attr):
        """
        For a generated args configuration, check it with rule one by one.
        :param attr: generated args
        :return: true/false
        """
        for rule in self.black_list:
            if rule(attr):
                return True
        return False


class CrossValidationRecorder:
    """
    An tool passed into trainer and evaluator to record result.
    Calculated reuslt will be used for cross validation.
    """

    def __init__(self):
        self.train_records = pd.DataFrame()
        self.eval_records = pd.DataFrame()

    def add_train_record(self, record, epoch, train_leave):
        if isinstance(train_leave, list):
            if len(train_leave) > 1:
                raise Exception('CV Recorder no support for multiple scenes {}'.format(train_leave))
            train_leave = train_leave[0]
        record['train_leave'] = train_leave
        record['epoch'] = epoch
        self.train_records = self.train_records.append(pd.DataFrame(data=record, index=[0]), ignore_index=True)

    def add_evaluation_result(self, record, epoch, valid_scene, sample_time):
        if isinstance(valid_scene, list):
            if len(valid_scene) > 1:
                raise Exception('CV Recorder no support for multiple scenes {}'.format(valid_scene))
            valid_scene = valid_scene[0]
        record['valid_scene'] = valid_scene
        record['epoch'] = epoch
        record['sample_time'] = sample_time
        self.eval_records = self.eval_records.append(pd.DataFrame(data=record, index=[0]), ignore_index=True)

    def calc_cv_result(self, metrics, recorder, weights):
        """
        Calculate Cross Validation Result according to evaluate result.
        :param metrics: list of string, names of metrics to be calculated.
        :param recorder: SummaryWriter
        :param weights: CrossValidation Weights
        :return: warning message
        """
        warning_msg = list()

        # check: missing scenes
        scenes = self.eval_records['valid_scene'].unique()
        no_appear_scenes = list(set(weights.keys()) - set(scenes))
        no_appear_invalid = [scene for scene in no_appear_scenes if weights[scene] > 0]
        if len(no_appear_invalid) > 0:
            warning_msg.append('Scene of weight > 0 {} never appears in data'.format(no_appear_invalid))
        # calculate by metrics
        exist_metrics = list(self.eval_records.columns)
        warning_msg.append('metrics {} not exist in recording, ignore.'.format(set(metrics) - set(exist_metrics)))

        # level 1: filter by sample_time
        # level 2: filter by metric
        # level 3: filter by epoch time (scene missing check)
        # level 4: weighted sum by scenes.
        # loop different sample configuration
        sample_times = self.eval_records['sample_time'].unique()
        for sample_time in sample_times:
            data = self.eval_records[self.eval_records['sample_time'] == sample_time]
            for metric in set(metrics).intersection(set(exist_metrics)):
                data_no_nan = data[~data[metric].isna()]  # get data without nan in 'metric'
                times = data_no_nan['epoch'].unique()  # get all time steps
                metric_result = dict()
                for time in times:
                    data_at_time = data_no_nan[data_no_nan['epoch'] == time]
                    scene_at_time = data_at_time['valid_scene'].unique()
                    # check scene data not appeared in this metric at this time.
                    no_appear_at_time = list(set(scenes) - set(scene_at_time))
                    no_appear_invalid_at_time = [scene for scene in no_appear_at_time if weights[scene] > 0]
                    if len(no_appear_invalid_at_time) > 0:
                        warning_msg.append(
                            'Scene {} missing in metric = {} at step = {} sample = {}'.format(no_appear_invalid_at_time,
                                                                                              metric, time, sample_time)
                        )
                    exist_scene_w = [weights[scene] for scene in scene_at_time]
                    exist_w_sum = sum(exist_scene_w)
                    # iterate by row and calculate cv of metric at time
                    result = 0.0
                    for _, row in data_at_time.iterrows():
                        result += row[metric] * weights[row['valid_scene']] / exist_w_sum
                    metric_result[time] = result
                metric_result = sorted(metric_result.items(), key=lambda item: item[0])
                # plot on board
                for time, value in metric_result:
                    recorder.writer.add_scalar('CV_{}/sample_{}'.format(metric, sample_time),
                                               scalar_value=value, global_step=time)
        return warning_msg

    def dump(self, path_prefix):
        """
        Dump train_records and eval_records to path.
        """
        self.train_records.to_csv(path_prefix + '_cv_train.csv', index=False)
        self.eval_records.to_csv(path_prefix + '_cv_eval.csv', index=False)


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
        if not os.path.exists(save_dir_root):
            os.mkdir(save_dir_root)
        if not os.path.exists(os.path.join(save_dir_root, prefix)):
            os.mkdir(os.path.join(save_dir_root, prefix))
        if not os.path.exists(task_attr.save_dir):
            os.mkdir(task_attr.save_dir)
        # initial recorder and trainer
        self.recorder = Recorder(summary_path=task_attr.board_name, logfile=True, stream=False)

    def run(self, global_recorder, cross_validation, only_eval):
        # if cv is True, then cv scenes are set
        if cross_validation:
            if self.task_attr.train_leave or self.task_attr.val_scene:
                raise Exception('scenes left in train or scenes for validation already defined.')
            global_recorder.logger.info('Cross Validation Mode: scenes = {}'.format(cross_scene))
            scenes = [(s, s) for s in cross_scene]
            weights = cross_weights
        # if cv is False, then transform to cross validation like format.
        else:
            scenes = [(self.task_attr.train_leave, self.task_attr.val_scene)]
            weights = {self.task_attr.val_scene: 1.0}

        # define global dict to record value for cross validation
        cv_rec = CrossValidationRecorder()

        for scene_pair in scenes:
            _task_attr = AttrDict(self.task_attr.copy())
            _task_attr.train_leave = scene_pair[0]
            _task_attr.val_scene = scene_pair[1]
            _task_attr.phase = 'leave_{}_teston_{}'.format(scene_pair[0], scene_pair[1])
            _task_attr.val_phase = 'leave_{}_teston_{}'.format(scene_pair[0], scene_pair[1])

            # in cv mode, more sub dirs will be made under current save dir.
            if cross_validation:
                _task_attr.save_dir = os.path.join(_task_attr.save_dir,
                                                   'leave_{}_test_{}'.format(scene_pair[0], scene_pair[1]))

            # in only evaluation mode, trained model needs to be restored.
            if only_eval:
                _task_attr.restore_dir = os.path.join(_task_attr.save_dir, only_eval_model_name)

            try:
                self.recorder.logger.info(_task_attr)
                trainer = Trainer(_task_attr, self.recorder)
                if cross_validation:
                    global_recorder.logger.info('CV Running @ leave {} and val {}. Save @ {}, Runs+Log @ {}'.format(
                        _task_attr.train_leave, _task_attr.val_scene, _task_attr.save_dir, _task_attr.board_name
                    ))
                    trainer.train_model(cv_recorder=cv_rec, only_eval=only_eval)
                else:
                    global_recorder.logger.info('Normal Running @ leave {} and val {}. Save @ {}, Runs+Log @ {}'.format(
                        _task_attr.train_leave, _task_attr.val_scene, _task_attr.save_dir, _task_attr.board_name
                    ))
                    trainer.train_model(only_eval=only_eval)

                global_recorder.logger.info('Task Ends Successfully. Save Dir = {}'.format(_task_attr.save_dir))
                self.recorder.logger.info('Task Ends Successfully. Save Dir = {} \n\n\n'.format(_task_attr.save_dir))

            except Exception as _:
                global_recorder.logger.info(
                    'Task Ends With Error, Details On Log in runs. Save Dir = {}'.format(_task_attr.save_dir))
                global_recorder.logger.info(traceback.format_exc())

                self.recorder.logger.info('Task Ends With Error. Save Dir = {} \n\n\n'.format(_task_attr.save_dir))
                self.recorder.logger.info(traceback.format_exc())

        # in cv mode, cv result will be plotted and general result will be dumped as csv file.
        if cross_validation:
            warns = cv_rec.calc_cv_result(cross_metrics, self.recorder, cross_weights)
            for warn in warns:
                global_recorder.logger.warn(warn)
            cv_rec.dump(os.path.join(self.task_attr.save_dir, 'summary'))
            global_recorder.logger.info('CV done for {}'.format(self.task_attr.save_dir))
        self.recorder.close()


if __name__ == '__main__':
    # How to use above three class.
    # 1. use ArgsMaker to make a list of args in a batch experiment.
    # 2. use ArgsBlocker to abandon specified args by ArgsMaker.
    # 3. deploy a task runner to run training.
    #    a. Use prefix to identify different batch experiments.
    #    b. All data in one batch experiment will be in stored in runs/prefix/ and save/prefix/
    # experiment prefix
    prefix = '0520'
    only_eval = False
    log_file = Recorder(os.path.join(runs_dir_root, prefix), board=False, logfile=True, stream=True)

    # 添加生成参数的规则
    argsMaker = ArgsMaker()
    argsMaker.add_arg_rule('model', ['seq2seq', 'vanilla'])
    argsMaker.add_arg_rule(['loss', 'val_sample_times'], [('2d_gaussian', [0, 1, 10, 20]), ('mixed', [0])], 'Lo_Sam')
    argsMaker.add_arg_rule(['embedding_size', 'cell_size'], [(64, 128), (32, 64), (16, 32), (8, 16)], brief='ebd_cell')
    # argsMaker.add_arg_rule(['embedding_size', 'cell_size'], [(64, 128)], brief='ebd_cell')
    argsMaker.add_arg_rule('relative', [False, True], brief='rel')

    blocker = ArgsBlocker()
    candidates = argsMaker.making_args_candidates().items()
    for index, item in enumerate(candidates):
        if blocker.is_blocked(item[1]):
            log_file.logger.info('{}/{} blocked '.format(index + 1, len(candidates)) + str(item[0]))
        else:
            log_file.logger.info('{}/{} pass '.format(index + 1, len(candidates)) + str(item[0]))
            task_runner = TaskRunner(prefix, item[1])
            task_runner.run(log_file, cross_validation=True, only_eval=only_eval)
    log_file.close()
