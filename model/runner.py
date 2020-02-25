import numpy as np
import torch
import os
import argparse
import time
from attrdict import AttrDict
from tqdm import tqdm

from data.dataloader import KittiDataLoader
from script.cuda import get_device, to_device
from model.utils import neg_likelihood_gaussian_pdf_loss, get_2d_gaussian, l2_loss, gaussian_sampler
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
        self.model, self.optimizer = self.build()
        self.model = to_device(self.model, self.device)
        self.data_loader = KittiDataLoader(self.args.train_dataset,
                                           self.args.batch_size,
                                           self.args.total_len,
                                           self.device,
                                           self.args.relative)

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
                                    dropout=self.args.dropout)
            elif self.args.model == 'seq2seq':
                model = Seq2SeqLSTM(input_dim=2,
                                    output_dim=5,
                                    pred_length=self.args.pred_len,
                                    emd_size=self.args.embedding_size,
                                    cell_size=self.args.cell_size,
                                    batch_norm=self.args.batch_norm,
                                    dropout=self.args.dropout)
            else:
                raise Exception('Model {} not implemented.'.format(self.args.model))
        else:
            raise Exception('More Input Not Implemented in Runner.')

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
            self.recorder.logger.info('Train continue from epoch {}'.format(checkpoint['epoch'] + 1))

        return model, optimizer

    def train_model(self):
        """
        Train model
        """
        checkpoint = dict()
        checkpoint['args'] = self.args

        self.recorder.logger.info(' >>> Starting training')

        # pre_epoch: restore from loaded model
        for epoch in range(self.pre_epoch + 1, self.pre_epoch + self.args.num_epochs + 1):
            start_time = time.time()
            batch_num = len(self.data_loader)

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
                result = self.model.train_step(self.model, x, self.args.pred_len)
                loss = neg_likelihood_gaussian_pdf_loss(result['gaussian_output'], y)
                ave_loss = torch.sum(loss) / (self.args.batch_size * self.args.pred_len)

                # backward
                self.optimizer.zero_grad()
                ave_loss.backward()
                if self.args.clip_threshold > 0:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip_threshold)
                self.optimizer.step()
                loss_list.append(ave_loss)

            end_time = time.time()
            ave_loss = np.array(loss_list).sum() / len(loss_list)

            # validate
            if epoch > 0 and epoch % self.args.validate_every == 0:
                checkpoint['model'] = self.model.state_dict()
                checkpoint['optimizer'] = self.optimizer.state_dict()
                self.validate_model(epoch=epoch, checkpoint=checkpoint)

            # print
            self.recorder.writer.add_scalar('train_loss', ave_loss, epoch)
            if epoch >= 0 and epoch % self.args.print_every == 0:
                self.recorder.logger.info('Epoch {} / {}, Train_Loss {}, Time {}'.format(
                    epoch,
                    self.args.num_epochs,
                    ave_loss,
                    end_time - start_time
                ))

            if epoch > 0 and epoch % self.args.save_every == 0:
                checkpoint['model'] = self.model.state_dict()
                checkpoint['optimizer'] = self.optimizer.state_dict()
                checkpoint['epoch'] = epoch
                if not os.path.exists(self.args.save_dir):
                    os.makedirs(self.args.save_dir)
                checkpoint_path = os.path.join(self.args.save_dir,
                                               'checkpoint_{}_{}_{}'.format(epoch, self.args.model, ave_loss))
                self.recorder.logger.info('Save {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                self.recorder.logger.info('Done')

    def validate_model(self, epoch, checkpoint):
        """
        Validate model when training. Get ave_loss, final_loss, ade, fde
        Plot trajectories on the tensor board.
        """
        # save current model parameters temporarily.
        checkpoint_path = os.path.join(self.args.save_dir, 'temp_checkpoint_val')
        torch.save(checkpoint, checkpoint_path)

        # create Tester
        val_dict = AttrDict({
            'model': self.args.model,
            'load_path': os.path.join(self.args.save_dir, 'temp_checkpoint_val'),
            'obs_len': self.args.val_obs_len,
            'pred_len': self.args.val_pred_len,
            'sample_times': self.args.val_sample_times,
            'test_dataset': self.args.val_dataset,
            'silence': True,
            'plot': self.args.val_plot,
            'relative': self.args.relative,
            'export_path': None
        })
        validator = Tester(val_dict, self.recorder)
        validator.evaluate(step=epoch)


class Tester:
    def __init__(self, args, recorder):
        self.args = args
        self.recorder = recorder
        self.device = torch.device('cpu')
        self.test_dataset = KittiDataLoader(self.args.test_dataset,
                                            1,
                                            self.args.obs_len + self.args.pred_len,
                                            self.device,
                                            self.args.relative)
        self.sampler = gaussian_sampler
        self.model = to_device(self.restore_model().train(False), self.device)

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
            self.model = VanillaLSTM
            model = VanillaLSTM(input_dim=2,
                                output_dim=5,
                                emd_size=train_args.embedding_size,
                                cell_size=train_args.cell_size,
                                batch_norm=train_args.batch_norm,
                                dropout=train_args.dropout)
        elif self.args.model == 'seq2seq':
            self.model = Seq2SeqLSTM
            model = Seq2SeqLSTM(input_dim=2,
                                output_dim=5,
                                emd_size=train_args.embedding_size,
                                cell_size=train_args.cell_size,
                                batch_norm=train_args.batch_norm,
                                dropout=train_args.dropout)
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
                pred_gaussian, y_hat = self.model.interface(model=self.model,
                                                            datax=x,
                                                            pred_len=self.args.pred_len,
                                                            sample_times=self.args.sample_times)
                # data post process
                abs_x, abs_y = self.model.evaluation_data_splitter(data, self.args.pred_len)
                abs_y_hat = self.test_dataset.post_process(y_hat, start=torch.unsqueeze(abs_x[:, -1, :], dim=1))

            else:
                x, y = self.model.evaluation_data_splitter(data, self.args.pred_len)
                pred_gaussian, y_hat = self.model.interface(model=self.model,
                                                            datax=x,
                                                            pred_len=self.args.pred_len,
                                                            sample_times=self.args.sample_times)
                abs_x = x
                abs_y = y
                abs_y_hat = y_hat

            # metric calculate
            loss = neg_likelihood_gaussian_pdf_loss(pred_gaussian, y)
            l2 = l2_loss(y_hat, y)
            euler = l2_loss(abs_y_hat, abs_y)

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

            msg1 = '{}_AveLoss_{:.3}_AveL2_{:.3}_FinalL2_{:.3}'.format(
                t, ave_loss, ave_l2, final_l2)
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
            record['gaussian_output'] = pred_gaussian.cpu().numpy()

            record['ave_loss'] = ave_loss.cpu().numpy()
            record['final_loss'] = final_loss.cpu().numpy()
            record['first_loss'] = first_loss.cpu().numpy()
            record['ave_l2'] = ave_l2.cpu().numpy()
            record['final_l2'] = final_l2.cpu().numpy()
            record['ade'] = ade.cpu().numpy()
            record['fde'] = fde.cpu().numpy()
            record['min_ade'] = min_ade.cpu().numpy()
            record['min_fde'] = min_fde.cpu().numpy()

            save_list.append(record)

        process.close()

        # globally average metrics calculation
        self.recorder.logger.info('Calculation of Global Metrics.')
        metric_list = ['ave_loss', 'final_loss', 'first_loss', 'ave_l2', 'final_l2', 'ade', 'fde', 'min_ade', 'min_fde']
        scalars = dict()
        for metric in metric_list:
            temp = list()
            for record in save_list:
                temp.append(record[metric])
            self.recorder.logger.info('{} : {}'.format(metric, sum(temp) / len(temp)))
            scalars[metric] = sum(temp) / len(temp)
        self.recorder.writer.add_scalars('test', scalars, global_step=step)

        # plot
        if self.args.plot:
            self.recorder.logger.info('Plot trajectory')
            self.recorder.plot_trajectory(save_list, step=step, cat_point=self.args.obs_len - 1, mode=1)

        # export
        if self.args.export_path:
            torch.save(save_list, self.args.export_path)
            self.recorder.logger.info('Export {} Done'.format(self.args.export_path))

        self.recorder.logger.info('### End Evaluation')
