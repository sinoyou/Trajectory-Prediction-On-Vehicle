import numpy as np
import torch
import os
import argparse
import time
import random

from data.dataloader import KittiDataLoader
from script.tools import abs_to_rel, Recorder
from model.utils import cal_loss_by_2d_gaussian, get_2d_gaussian, l2_loss
from model.vanilla import VanillaLSTM
from model.seq2seq import Seq2SeqLSTM


class Runner:
    def __init__(self, args: argparse.ArgumentParser(), recorder, data_splitter):
        """
        Runner for model training, validation and test
        :param args: arguments of training or test
        :param recorder: log information class (stdout + tensor board)
        :param data_splitter: split data of total_length according to different architecture.
        """
        self.args = args
        self.recorder = recorder
        self.model, self.optimizer = self.build()
        self.data_loader = KittiDataLoader(self.args.train_dataset,
                                           self.args.batch_size,
                                           self.args.total_length)
        self.validate_data_loader = KittiDataLoader(self.args.val_dataset, 1, self.args.total_length)
        self.data_splitter = data_splitter

    def build(self):
        """
        Build new models or restore old models from the directory.
        :return: model object
        """
        # model
        if not self.args.bbox:
            vanilla = VanillaLSTM(input_dim=2,
                                  output_dim=5,
                                  emd_size=self.args.embedding_size,
                                  cell_size=self.args.cell_size,
                                  batch_norm=self.args.batch_norm,
                                  dropout=self.args.dropout)
        else:
            raise Exception('More Input Not Implemented in Runner.')

        # optimizer
        optimizer = torch.optim.Adam(vanilla.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)

        if self.args.load_dir and os.path.isfile(self.args.load_dir):
            checkpoint = torch.load(self.args.load_dir)
            vanilla.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        return vanilla, optimizer

    def train_model(self):
        """
        Train model
        """
        checkpoint = dict()
        checkpoint['args'] = self.args

        def loss_and_step(gaussian_output, datay):
            loss = cal_loss_by_2d_gaussian(gaussian_output, datay)
            loss = torch.sum(loss)

            self.optimizer.zero_grad()
            loss.backward()
            if self.args.clip_threshold > 0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                              self.args.clip_threshold)
            self.optimizer.step()
            return loss

        for epoch in range(self.args.num_epochs):
            self.recorder.logger.info('Starting epoch {}'.format(epoch))
            start_time = time.time()
            batch_num = len(self.data_loader)

            loss_list = []
            self.data_loader.reset_ptr()
            for itr in range(batch_num):
                # load data
                data = self.data_loader.next_batch()
                rel_data = abs_to_rel(data)
                x, y = self.data_splitter(data, self.args.pred_len)
                rel_x, rel_y = self.data_splitter(rel_data, self.args.pred_len)
                # model forward and backward
                model_output, _ = self.model(rel_x)
                gaussian_output = get_2d_gaussian(model_output=model_output)
                loss = loss_and_step(gaussian_output, rel_y)
                loss_list.append(loss)

            end_time = time.time()
            ave_loss = np.array(loss_list).sum() / len(loss_list)

            # validate
            if epoch > 0 and epoch % self.args.validate_every:
                self.recorder.logger.info('Begin Validation')
                self.validate_model(epoch=epoch)
                self.recorder.logger.info('End Validation')

            # print
            self.recorder.writer.add_sclar('loss', ave_loss, epoch)
            if epoch > 0 and epoch % self.args.print_every == 0:
                self.recorder.logger.info('Epoch {}, Loss {}, Time {}'.format(
                    epoch,
                    ave_loss,
                    end_time - start_time
                ))

            # save
            if epoch > 0 and epoch % self.args.save_every == 0:
                checkpoint['model'] = self.model.state_dict()
                checkpoint['optimizer'] = self.optimizer.state_dict()
                if not os.path.exists(self.args.save_dir):
                    os.makedirs(self.args.save_dir)
                checkpoint_path = os.path.join(self.args.save_dir,
                                               'checkpoint_{}_{}_{}'.format(epoch, self.args.model, ave_loss))
                self.recorder.logger.info('Save {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                self.recorder.logger.info('Done')

    def validate_model(self, epoch):
        """
        Validate model when training. Get ave_loss, final_loss, ade, fde
        Plot trajectories on the tensor board.
        """
        self.validate_data_loader.reset_ptr()
        trajectories = []
        ave_losses = []
        final_losses = []
        ades = []
        fdes = []
        for i in range(len(self.validate_data_loader)):
            val_data = self.validate_data_loader.next_batch()
            rel_val_data = abs_to_rel(val_data)
            x, y = self.data_splitter(val_data, self.args.pred_len)
            rel_x, rel_y = self.data_splitter(rel_val_data, self.args.pred_len)

            model_output, _ = self.model(rel_x)
            gaussian_output = get_2d_gaussian(model_output=model_output)

            loss = cal_loss_by_2d_gaussian(gaussian_output, rel_y)
            l2 = l2_loss(gaussian_output[:, :, 0:2], rel_y)

            # metrics
            ave_loss = torch.sum(loss) / self.args.pred_len
            ave_losses.append(ave_loss)

            final_loss = loss[0, -1, 0]
            final_losses.append(final_loss)

            ade = torch.sum(l2) / self.args.pred_len
            ades.append(ade)

            fde = l2[0, -1, 0]
            fdes.append(fde)

            # prediction
            trajectory = dict()
            trajectory['tag'] = i
            trajectory['x'] = torch.squeeze(x)
            trajectory['y'] = torch.squeeze(y)
            trajectory['rel_x'] = torch.squeeze(rel_x)
            trajectory['rel_y'] = torch.squeeze(rel_y)
            trajectory['gaussian_output'] = torch.squeeze(gaussian_output)
            trajectories.append(trajectory)

        ave_loss = np.sum(np.array(ave_losses)) / len(ave_losses)
        final_loss = np.sum(np.array(final_losses)) / len(final_losses)
        ade = np.sum(np.array(ades)) / len(ades)
        fde = np.sum(np.array(fdes)) / len(fdes)

        # record and plot
        self.recorder.logger.info('val: ave_loss {}, final_loss {}, ade {}, fde {}'.format(
            ave_loss, final_loss, ade, fde
        ))

        scalars = dict()
        scalars['ave_loss'] = ave_loss
        scalars['final_loss'] = final_loss
        scalars['ade'] = ade
        scalars['fde'] = fde
        self.recorder.writer.add_scalars('val', scalars, epoch)

        if self.args.plot_trajectory:
            self.recorder.plot_trajectory(trajectories, step=epoch)


class Tester:
    def __init__(self, args, recorder):
        self.args = args
        self.recorder = recorder
        self.test_dataset = KittiDataLoader(self.args.test_dataset, 1, self.args.obs_len + self.args.pred_len)
        self.splitter = None
        self.sampler = gaussian_sampler
        self.model = self.restore_model()

    def restore_model(self) -> torch.nn.Module:
        """
        Load trained model and allocate corresponding data splitter.
        :return: loaded model
        """
        if not os.path.exists(self.args.load_path):
            raise Exception('File {} not exists.'.format(self.args.load_path))

        checkpoint = torch.load(self.args.load_path)
        train_args = checkpoint['args']
        if self.args.model == 'vanilla':
            self.splitter = vanilla_evaluation_data_splitter
            model = VanillaLSTM(input_dim=2,
                                output_dim=5,
                                emd_size=train_args.embedding_size,
                                cell_size=train_args.cell_size,
                                batch_norm=train_args.batch_norm,
                                dropout=train_args.dropout)
            model.load_state_dict(checkpoint['model'])
            model.eval()
        else:
            raise Exception('Model {} not implemented. '.format(self.args.model))
        return model

    def evaluate(self):
        """
        Evaluate Loaded Model with one-by-one case. Then calculate metrics and plot result.
        Global and Case metrics: ave_loss, final_loss, ave_l2, final_l2, ade, fde
        Hint: differences between l2 and destination error, l2 is based on relative dis while the other on absolute one.
        """
        self.recorder.logger.info('Begin Evaluation, {} test cases in total'.format(len(self.test_dataset)))
        save_list = list()
        for t in range(len(self.test_dataset)):

            raw_seq = self.test_dataset.next_batch()
            rel_raw_seq = abs_to_rel(raw_seq)
            x, y = self.splitter(raw_seq)
            rel_x, rel_y = self.splitter(rel_raw_seq)

            rel_y_hat = torch.zeros_like(rel_y)
            gaussian_output_list = list()

            # initial hidden state
            output, hc = self.model(rel_x, hc=None)

            # predict iterative
            for itr in range(self.args.pred_len):
                # sampler
                gaussian_output = get_2d_gaussian(output)
                gaussian_output_list.append(gaussian_output)
                rel_y_hat[0, itr, 0], rel_y_hat[0, itr, 1] = self.sampler(gaussian_output[0, itr, 0],
                                                                          gaussian_output[0, itr, 1],
                                                                          gaussian_output[0, itr, 2],
                                                                          gaussian_output[0, itr, 3],
                                                                          gaussian_output[0, itr, 4])
                if itr == self.args.pred_len - 1:
                    break

                itr_x_rel = rel_y_hat[:, itr, :]
                output, hc = self.model(itr_x_rel, hc)

            gaussian_output = torch.stack(gaussian_output_list, dim=1)

            # metric calculate
            loss = cal_loss_by_2d_gaussian(gaussian_output, rel_y)
            l2 = l2_loss(rel_y_hat, rel_y)
            euler = l2_loss(rel_to_abs(rel_y_hat), rel_to_abs(rel_y))

            ave_loss = torch.sum(loss) / self.args.pred_len / 1
            final_loss = torch.sum(loss[0, -1, 0]) / 1 / 1
            ave_l2 = torch.sum(l2) / self.args.pred_len / 1
            final_l2 = torch.sum(l2[0, -1, 0]) / 1 / 1
            ade = torch.sum(euler) / self.args.pred_len / 1
            fde = torch.sum(euler[0, -1, 0]) / 1 / 1

            msg = 'AveLoss: {}, FinalLoss: {}, AveL2: {}, FinalL2: {}, Ade: {}, Fde: {}'.format(
                ave_loss, final_loss, ave_l2, final_l2, ade, fde)
            self.recorder.logger.info(msg)

            # plot
            record = dict()
            record['tag'] = t
            record['x'] = x
            record['y'] = y
            record['rel_x'] = rel_x
            record['rel_y'] = rel_y
            record['rel_y_hat'] = rel_y_hat
            record['gaussian_output'] = gaussian_output
            record['ave_loss'] = ave_loss
            record['final_loss'] = final_loss
            record['ave_l2'] = ave_l2
            record['final_l2'] = final_l2
            record['ade'] = ade
            record['fde'] = fde

            save_list.append(record)
            self.recorder.plot_trajectory(record, 1, msg)

        # average metrics calculation
        self.recorder.logger.info('Calculation of Global Metrics.')
        metric_list = ['ave_loss', 'final_loss', 'ave_l2', 'final_l2', 'ave_l2', 'final_l2']
        for metric in metric_list:
            temp = list()
            for record in save_list:
                temp.append(record[metric])
            self.recorder.logger.info('{} : {}'.format(metric, sum(temp) / len(temp)))

        # export
        if self.args.export_path:
            torch.save(save_list, self.args.export_path)
            self.recorder.logger.info('Export {} Done'.format(self.args.export_path))
