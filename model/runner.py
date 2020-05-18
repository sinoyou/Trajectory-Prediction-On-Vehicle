import numpy as np
import torch
import os
import argparse
import time
from attrdict import AttrDict
from tqdm import tqdm

# from data.dataloader import KittiDataLoader
from data.kitti_dataloader import SingleKittiDataLoader
from script.cuda import get_device, to_device
from model.utils import l2_loss, l1_loss
from model.vanilla import VanillaLSTM
from model.seq2seq import Seq2SeqLSTM


class Trainer:
    def __init__(self, args: argparse.ArgumentParser(), recorder):
        """
        Runner for model training, validation and test
        :param args: arguments of training or test
        :param recorder: log information class (stdout + tensor board)
        """
        self.args = args
        self.device = get_device()
        self.recorder = recorder
        self.pre_epoch = 0
        self.best_eval_result = dict()
        self.model, self.optimizer = self.build()
        self.data_loader = SingleKittiDataLoader(file_path=self.args.train_dataset,
                                                 batch_size=self.args.batch_size,
                                                 trajectory_length=self.args.total_len,
                                                 mode='train',
                                                 train_leave=self.args.train_leave,
                                                 device=self.device,
                                                 recorder=self.recorder,
                                                 valid_scene=None)

    def build(self):
        """
        Build new models or restore old models from the directory.
        :return: model object
        """
        # model
        if not self.args.bbox:
            if self.args.model == 'vanilla':
                model = VanillaLSTM(input_dim=2,
                                    output_dim=5,
                                    emd_size=self.args.embedding_size,
                                    cell_size=self.args.cell_size,
                                    batch_norm=self.args.batch_norm,
                                    dropout=self.args.dropout,
                                    loss=self.args.loss)
            elif self.args.model == 'seq2seq':
                model = Seq2SeqLSTM(input_dim=2,
                                    output_dim=5,
                                    pred_length=self.args.pred_len,
                                    emd_size=self.args.embedding_size,
                                    cell_size=self.args.cell_size,
                                    batch_norm=self.args.batch_norm,
                                    dropout=self.args.dropout,
                                    loss=self.args.loss)
            else:
                raise Exception('Model {} not implemented.'.format(self.args.model))
        else:
            raise Exception('More Input Not Implemented in Runner.')

        # ! cuda operation must be front of optimizer define.
        model = to_device(model, self.device)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)

        # restore
        if self.args.restore_dir and os.path.exists(self.args.restore_dir):
            self.recorder.logger.info('Restoring from {}'.format(self.args.restore_dir))
            checkpoint = torch.load(self.args.restore_dir)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            self.pre_epoch = checkpoint['epoch']
            self.best_eval_result = checkpoint['best_result']
            self.recorder.logger.info('Train continue from epoch {}'.format(checkpoint['epoch'] + 1))

        return model, optimizer

    def train_model(self, cv_recorder=None, only_eval=False):
        """
        Train model
        """
        checkpoint = dict()
        checkpoint['args'] = self.args
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        # Only Evaluation Mode, restored model will be directly used for evaluation
        # Hint: this function is used together with experiments.py/cross_validation. Other purposes are not recommended.
        if only_eval:
            self.recorder.logger.info('>>> Only Evaluation Mode')
            checkpoint['model'] = self.model.state_dict()
            checkpoint['optimizer'] = self.optimizer.state_dict()
            self.validate_model(epoch=self.pre_epoch, checkpoint=checkpoint, cv_recorder=cv_recorder)
            return

        self.recorder.logger.info(' >>> Starting training')

        # pre_epoch: restore from loaded model
        for epoch in range(self.pre_epoch + 1, self.pre_epoch + self.args.num_epochs + 1):
            start_time = time.time()
            batch_num = len(self.data_loader)  # self.count // self.batch_size

            loss_list = []
            self.data_loader.reset_ptr()
            for itr in range(batch_num):
                # load data
                batch = self.data_loader.next_batch()
                if self.args.relative:
                    data = batch['rel_data']
                else:
                    data = batch['data']
                x, y = self.model.train_data_splitter(data, self.args.pred_len)

                # forward
                result = self.model.train_step(x, pred_len=self.args.pred_len, y_gt=y)
                loss = result['loss']
                ave_loss = torch.sum(loss) / (self.args.batch_size * self.args.pred_len)

                # backward
                self.optimizer.zero_grad()
                ave_loss.backward()
                if self.args.clip_threshold > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_threshold)
                self.optimizer.step()
                loss_list.append(ave_loss)

            end_time = time.time()
            ave_loss = np.array(loss_list).sum() / len(loss_list)

            # validate -> validate_model
            # save checkpoint['model'] ['optimizer'] in temp_checkpoint_val
            if epoch > 0 and epoch % self.args.validate_every == 0:
                checkpoint['model'] = self.model.state_dict()
                checkpoint['optimizer'] = self.optimizer.state_dict()
                self.validate_model(epoch=epoch, checkpoint=checkpoint, cv_recorder=cv_recorder)

            # print
            summary = {
                'train_loss': float(ave_loss)
            }
            for name, value in summary.items():
                self.recorder.writer.add_scalar('{}/{}'.format(self.args.phase, name),
                                                scalar_value=value,
                                                global_step=epoch)  # train folder

            if epoch >= 0 and epoch % self.args.print_every == 0:
                self.recorder.logger.info('Epoch {} / {}, Train_Loss {}, Time {}'.format(
                    epoch,
                    self.args.num_epochs + self.pre_epoch,
                    ave_loss,
                    end_time - start_time
                ))
                if cv_recorder:
                    cv_recorder.add_train_record(summary, epoch, self.args.train_leave)

            # save checkpoint['model'] ['optimizer'] ['epoch'] in 'checkpoint_{}_{}_{}'.format(epoch, self.args.model, ave_loss)
            if epoch > 0 and epoch % self.args.save_every == 0:
                checkpoint['model'] = self.model.state_dict()
                checkpoint['optimizer'] = self.optimizer.state_dict()
                checkpoint['epoch'] = epoch
                checkpoint['best_result'] = self.best_eval_result
                checkpoint_path = os.path.join(self.args.save_dir,
                                               'checkpoint_{}_{}.ckpt'.format(epoch, self.args.model))
                newest_path = os.path.join(self.args.save_dir, 'latest_checkpoint.ckpt')
                self.recorder.logger.info('Save {}'.format(checkpoint_path))
                self.recorder.logger.info('Update latest checkpoint version.')
                torch.save(checkpoint, checkpoint_path)
                torch.save(checkpoint, newest_path)
                self.recorder.logger.info('Done')

    def validate_model(self, epoch, checkpoint, cv_recorder):
        """
        Validate model when training.
        1. One trained model can corresponds to multiple validation config on sample_times.
        2. After each validation, result under single config will be used for updating best result.
        """
        # save current model parameters temporarily.
        checkpoint_path = os.path.join(self.args.save_dir, 'temp_checkpoint_val')
        torch.save(checkpoint, checkpoint_path)

        for sample_time in self.args.sample_times:
            self.recorder.logger.info('validating sample_times = {}'.format(sample_time))
            # create Tester
            val_dict = AttrDict({
                'model': self.args.model,
                'load_path': os.path.join(self.args.save_dir, 'temp_checkpoint_val'),
                'obs_len': self.args.val_obs_len,
                'pred_len': self.args.val_pred_len,
                'sample_times': sample_time,
                'test_dataset': self.args.val_dataset,
                'train_leave': self.args.train_leave,
                'test_scene': self.args.val_scene,
                'silence': True,
                'plot': self.args.val_plot,
                'plot_mode': self.args.val_plot_mode,
                'relative': self.args.relative,
                'export_path': None,
                'board_name': self.args.board_name,
                'phase': self.args.val_phase + '/sample_{}'.format(sample_time)
            })
            validator = Tester(val_dict, self.recorder)
            feedback = validator.evaluate(step=epoch)

            # evaluation result process
            self.best_eval_result.setdefault(sample_time, dict())
            for key, value in feedback['global_metrics']:
                best_eval_sample = self.best_eval_result[sample_time]
                if 'best_' + key in best_eval_sample.keys():
                    best_eval_sample['best_' + key] = min(best_eval_sample['best_' + key], value)
                else:
                    best_eval_sample['best_' + key] = value
            # cv recoder not None, then combine feedback and best result to record.
            if cv_recorder:
                temp = feedback['global_metrics'].copy()
                for k, v in self.best_eval_result[sample_time]:
                    temp[k] = v
                cv_recorder.add_train_record(record=temp, epoch=epoch, train_leave=self.args.train_leave,
                                             sample_time=sample_time)
            return feedback


class Tester:
    def __init__(self, args, recorder):
        self.args = args
        self.train_args = None
        self.recorder = recorder
        self.device = torch.device('cpu')
        self.test_dataset = SingleKittiDataLoader(file_path=self.args.test_dataset,
                                                  batch_size=1,
                                                  trajectory_length=self.args.obs_len + self.args.pred_len,
                                                  mode='valid',
                                                  train_leave=self.args.train_leave,
                                                  device=self.device,
                                                  recorder=self.recorder,
                                                  valid_scene=self.args.test_scene)
        self.model = to_device(self.restore_model().train(False), self.device)

        self.args_check()

    def args_check(self):
        if self.args.sample_times >= 1:
            self.recorder.logger.info('AutoRegressive Mode: Sample')
        elif self.args.sample_times == 0:
            self.recorder.logger.info('AutoRegressive Mode: Biggest Likelihood.')
        else:
            raise Exception('Invalid args sample_times {}'.format(self.args.sample_times))

    def restore_model(self) -> torch.nn.Module:
        """
        Load trained model and allocate corresponding data splitter.
        :return: loaded model
        """
        if not os.path.exists(self.args.load_path):
            raise Exception('File {} not exists.'.format(self.args.load_path))

        checkpoint = torch.load(self.args.load_path, map_location=self.device)
        train_args = checkpoint['args']
        self.train_args = train_args
        if self.args.model == 'vanilla':
            self.model = VanillaLSTM
            model = VanillaLSTM(input_dim=2,
                                output_dim=5,
                                emd_size=train_args.embedding_size,
                                cell_size=train_args.cell_size,
                                batch_norm=train_args.batch_norm,
                                dropout=train_args.dropout,
                                loss=train_args.loss)
        elif self.args.model == 'seq2seq':
            self.model = Seq2SeqLSTM
            model = Seq2SeqLSTM(input_dim=2,
                                output_dim=5,
                                pred_length=train_args.pred_len,
                                emd_size=train_args.embedding_size,
                                cell_size=train_args.cell_size,
                                batch_norm=train_args.batch_norm,
                                dropout=train_args.dropout,
                                loss=train_args.loss)
        else:
            raise Exception('Model {} not implemented. '.format(self.args.model))

        model.load_state_dict(checkpoint['model'])

        return model

    def evaluate(self, step=1):
        """
        Evaluate Loaded Model with one-by-one case. Then calculate metrics and plot result.
        Global and Case metrics: ave_loss, final_loss, ave_l2, final_l2, ade, fde
        Hint: differences between l2 and destination error, l2 is based on relative dis while the other on absolute one.
        :param step: global evaluation step. (ex. in validation, it could be epoch and in test usually is 1)
        """
        self.recorder.logger.info('### Begin Evaluation {}, {} test cases in total'.format(
            step, len(self.test_dataset))
        )
        save_list = list()
        process = tqdm(range(len(self.test_dataset)))
        for t in range(len(self.test_dataset)):
            process.update(n=1)

            batch = self.test_dataset.next_batch()
            data, rel_data = batch['data'], batch['rel_data']
            if self.args.relative:
                x, y = self.model.evaluation_data_splitter(rel_data, self.args.pred_len)
                result = self.model.inference(datax=x,
                                              pred_len=self.args.pred_len,
                                              sample_times=self.args.sample_times)
                pred_distribution, y_hat = result['sample_pred_distribution'], result['sample_y_hat']

                # data post process
                abs_x, abs_y = self.model.evaluation_data_splitter(data, self.args.pred_len)
                # post process relative to absolute
                abs_y_hat = self.test_dataset.rel_to_abs(y_hat, start=torch.unsqueeze(abs_x[:, -1, :], dim=1))
                loss = self.model.get_loss(distribution=pred_distribution, y_gt=y)  # norm scale

            else:
                x, y = self.model.evaluation_data_splitter(data, self.args.pred_len)
                result = self.model.inference(datax=x,
                                              pred_len=self.args.pred_len,
                                              sample_times=self.args.sample_times)
                pred_distribution, y_hat = result['sample_pred_distribution'], result['sample_y_hat']

                abs_x = x
                abs_y = y
                abs_y_hat = y_hat
                loss = self.model.get_loss(distribution=pred_distribution, y_gt=y)  # norm scale

            # transform abs_* & pred_distribution to raw scale.
            # Only when used data is absolute, we need to transform it into raw scale.
            if not self.args.relative:
                x = self.test_dataset.norm_to_raw(x)
                y = self.test_dataset.norm_to_raw(y)
                y_hat = self.test_dataset.norm_to_raw(y_hat)
                abs_x = self.test_dataset.norm_to_raw(abs_x)
                abs_y = self.test_dataset.norm_to_raw(abs_y)
                abs_y_hat = self.test_dataset.norm_to_raw(abs_y_hat)
                pred_distribution = self.test_dataset.norm_to_raw(pred_distribution)

            # metric calculate
            l2 = l2_loss(y_hat, y)  # norm scale
            euler = l2_loss(abs_y_hat, abs_y)  # raw scale
            l1_x = l1_loss(torch.unsqueeze(abs_y_hat[..., 0], dim=2), torch.unsqueeze(abs_y[..., 0], dim=2))
            l1_y = l1_loss(torch.unsqueeze(abs_y_hat[..., 1], dim=2), torch.unsqueeze(abs_y[..., 1], dim=2))

            # average metrics calculation
            # Hint: when mode is absolute, abs_? and ? are the same, so L2 loss and destination error as well.
            ave_loss = torch.sum(loss) / (self.args.pred_len * self.args.sample_times)
            first_loss = torch.sum(loss[:, 0, :]) / self.args.sample_times
            final_loss = torch.sum(loss[:, -1, :]) / self.args.sample_times
            ave_l2 = torch.sum(l2) / (self.args.pred_len * self.args.sample_times)
            final_l2 = torch.sum(l2[:, -1, :]) / self.args.sample_times
            ade = torch.sum(euler) / (self.args.pred_len * self.args.sample_times)
            fde = torch.sum(euler[:, -1, :]) / self.args.sample_times
            min_ade = torch.min(torch.sum(euler, dim=[1, 2]) / self.args.pred_len)
            min_fde = torch.min(euler[:, -1, :])
            ade_x = torch.sum(l1_x) / (self.args.pred_len * self.args.sample_times)
            ade_y = torch.sum(l1_y) / (self.args.pred_len * self.args.sample_times)
            fde_x = torch.sum(l1_x[:, -1, :]) / self.args.sample_times
            fde_y = torch.sum(l1_y[:, -1, :]) / self.args.sample_times
            min_ade_x = torch.min(torch.sum(l1_x, dim=[1, 2]) / self.args.pred_len)
            min_ade_y = torch.min(torch.sum(l1_y, dim=[1, 2]) / self.args.pred_len)
            min_fde_x = torch.min(l1_x[:, -1, :])
            min_fde_y = torch.min(l1_y[:, -1, :])

            msg1 = '{}_AveLoss_{:.3}_adex_{:.3}_adey_{:.3}_fdex_{:.3}_fdey_{:.3}'.format(
                t, ave_loss, ade_x, ade_y, fde_x, fde_y)
            msg2 = '{}_Ade_{:.3}_Fde_{:.3}_MAde_{:.3f}_MFde_{:.3f}'.format(
                t, ade, fde, min_ade, min_fde)
            if not self.args.silence:
                self.recorder.logger.info(msg1 + "_" + msg2)

            # plot
            record = dict()
            record['tag'] = t
            record['step'] = step
            record['title'] = msg2

            record['x'] = x.cpu().numpy()
            record['abs_x'] = abs_x.cpu().numpy()
            record['y'] = y.cpu().numpy()
            record['abs_y'] = abs_y.cpu().numpy()
            record['y_hat'] = y_hat.cpu().numpy()
            record['abs_y_hat'] = abs_y_hat.cpu().numpy()
            record['pred_distribution'] = pred_distribution.cpu().numpy()

            record['ave_loss'] = ave_loss.cpu().numpy()
            record['final_loss'] = final_loss.cpu().numpy()
            record['first_loss'] = first_loss.cpu().numpy()
            record['ave_l2'] = ave_l2.cpu().numpy()
            record['final_l2'] = final_l2.cpu().numpy()
            record['ade'] = ade.cpu().numpy()
            record['fde'] = fde.cpu().numpy()
            record['min_ade'] = min_ade.cpu().numpy()
            record['min_fde'] = min_fde.cpu().numpy()
            record['ade_x'] = ade_x.cpu().numpy()
            record['ade_y'] = ade_y.cpu().numpy()
            record['fde_x'] = fde_x.cpu().numpy()
            record['fde_y'] = fde_y.cpu().numpy()
            record['min_ade_x'] = min_ade_x.cpu().numpy()
            record['min_ade_y'] = min_ade_y.cpu().numpy()
            record['min_fde_x'] = min_fde_x.cpu().numpy()
            record['min_fde_y'] = min_fde_y.cpu().numpy()

            save_list.append(record)

        process.close()

        # globally average metrics calculation
        self.recorder.logger.info('Calculation of Global Metrics.')
        metric_list = ['ave_loss', 'ade', 'fde', 'ade_x', 'ade_y', 'fde_x', 'fde_y',
                       'min_ade', 'min_fde', 'min_ade_x', 'min_ade_y', 'min_fde_x', 'min_fde_y']
        global_metrics = dict()
        for metric in metric_list:
            temp = list()
            for record in save_list:
                temp.append(record[metric])
            self.recorder.logger.info('{} : {}'.format(metric, sum(temp) / len(temp)))
            global_metrics[metric] = float(sum(temp) / len(temp))
            self.recorder.writer.add_scalar('{}/{}'.format(self.args.phase, metric),
                                            global_metrics[metric], global_step=step)

        # plot
        if self.args.plot:
            if self.model.loss == '2d_gaussian':
                self.recorder.logger.info('Plot trajectory')
                self.recorder.plot_trajectory(save_list, step=step, cat_point=self.args.obs_len - 1,
                                              mode=self.args.plot_mode, relavtive=self.args.relative)
            elif self.model.loss == 'mixed' and self.args.plot_mode == 1:
                self.recorder.logger.info('Plot trajectory')
                self.recorder.plot_trajectory(save_list, step=step, cat_point=self.args.obs_len - 1,
                                              mode=self.args.plot_mode, relavtive=self.args.relative)
            else:
                self.recorder.logger.info('[SKIP PLOT] No support for loss {}'.format(self.model.loss))

        # export
        if self.args.export_path:
            torch.save(save_list, self.args.export_path)
            self.recorder.logger.info('Export {} Done'.format(self.args.export_path))

        self.recorder.logger.info('### End Evaluation')

        return {'global_metrics': global_metrics}
