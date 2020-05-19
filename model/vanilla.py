import argparse
import os
import time

from torch import nn, optim
import torch
from model.utils import make_mlp, get_2d_gaussian, get_mixed, gaussian_sampler, get_loss_by_name
from script.cuda import to_device


class VanillaLSTM(torch.nn.Module):

    def __init__(self, input_dim, output_dim,
                 emd_size, cell_size,
                 batch_norm, dropout, loss):
        """
        Implement of Vanilla LSTM as in paper "social lstm".
        For each forward process, sequence length is dynamic.
        - None seq2seq structure.
        - Input is 2D point of previous GT/Pred by default.
        - Output is 5-parameter of Gaussian Distribution by default.
        :param input_dim: default 2 (x,y)
        :param output_dim: default 5 (mu1, mu2, sigma1, sigma2, cor)
        :param emd_size: default 128
        :param cell_size: default 128
        :param batch_norm: default True
        :param dropout: default 0
        :param loss: loss type - 2d_gaussian, mixed
        """
        super(VanillaLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emd_size = emd_size
        self.cell_size = cell_size
        self.loss = loss

        self.input_norm = torch.nn.BatchNorm1d(self.input_dim) if batch_norm else None
        self.input_emd_layer = make_mlp([self.input_dim, self.emd_size],
                                        batch_norm=batch_norm, dropout=dropout, activation='relu')
        self.output_emd_layer = make_mlp([self.cell_size, self.output_dim],
                                         batch_norm=batch_norm, dropout=dropout, activation=None)
        self.rnn_cell = nn.LSTMCell(self.emd_size, self.cell_size)

    def forward(self, inputs, hc=None):
        """
        :param inputs: [batch_size, length, input_dim]
        :param hc: initial hidden state
        :return: outputs: [batch_size, length, output_dim]
        """
        seq_len = inputs.shape[1]
        # normalization
        if self.input_norm:
            inputs = inputs.reshape(-1, 2)
            inputs = self.input_norm(inputs)
            inputs = inputs.view((-1, seq_len, self.input_dim))

        # forward
        outputs = []
        for step in range(seq_len):
            step_input = inputs[:, step, :]
            emd_input = self.input_emd_layer(step_input)
            hc = self.rnn_cell(input=emd_input, hx=hc)
            emd_output = self.output_emd_layer(hc[0])
            outputs.append(emd_output)
        return torch.stack(outputs, dim=1), hc

    def train_data_splitter(self, batch_data, pred_len):
        """
        Split data [batch_size, total_len, 2] into datax and datay in train/val mode
        :param batch_data: data to be split
        :param pred_len: length of trajectories in final loss calculation
        :return: datax, datay
        """
        total_len = batch_data.shape[1]
        datax = batch_data[:, :-1, :]
        datay = batch_data[:, total_len - pred_len:, :]
        return datax, datay

    def evaluation_data_splitter(self, batch_data, pred_len):
        """
        Split data [batch_size, total_len, 2] into datax and datay in val/evaluation mode
        :param batch_data: data to be split
        :param pred_len: lengthof trajectories in final loss calculation
        :return: datax, datay
        """
        total_len = batch_data.shape[1]
        datax = batch_data[:, 0:-pred_len, :]
        datay = batch_data[:, -pred_len:, :]
        return datax, datay

    def train_step(self, x, y_gt, **kwargs):
        """
        Run one train step
        :param y_gt: Groud Truth y [batch_size, pred_len, 2]
        :param x: [batch_size, obs_len, 2]
        :return: dict()
        """
        model_output, _ = self(x, hc=None)
        model_output = model_output[:, -kwargs['pred_len']:, :]
        if self.loss == '2d_gaussian':
            pred_distribution = get_2d_gaussian(model_output)
        elif self.loss == 'mixed':
            pred_distribution = get_mixed(model_output)
        else:
            raise Exception('Distribution undefined.')
        loss = self.get_loss(pred_distribution, y_gt)
        return {'model_output': model_output, 'pred_distribution': pred_distribution, 'loss': loss}

    def get_loss(self, distribution, y_gt):
        return get_loss_by_name(distribution=distribution, y=y_gt, name=self.loss)

    def inference(self, datax, pred_len, sample_times):
        """
        During evaluation, use trained model to inference.
        :param datax: obs data [1, obs_len, 2]
        :param pred_len: length of prediction
        :param sample_times: times of sampling trajectories (if 0, means not using sample)
        :return: gaussian_output [sample_times, pred_len, 5], location_output[sample_times, pred_len, 2]
        """
        device = datax.device
        batch_size = datax.shape[0]

        if self.loss != '2d_gaussian' and sample_times >= 1:
            raise Exception('No sample support for {}'.format(self.loss))

        if sample_times == 0:
            sample_times = 1
            use_sample = False
        else:
            sample_times = sample_times
            use_sample = True

        with torch.no_grad():
            sample_distribution = list()
            sample_location = list()
            if self.loss == '2d_gaussian':
                for _ in range(sample_times):
                    y_hat = to_device(torch.zeros((batch_size, pred_len, 2)), device)
                    gaussian_output = to_device(torch.zeros((batch_size, pred_len, 5)), device)

                    # initial hidden state
                    output, hc = self(datax, hc=None)
                    output = torch.unsqueeze(output[:, -1, :], dim=1)

                    # predict iterative
                    for itr in range(pred_len):
                        gaussian_output[:, itr, :] = get_2d_gaussian(output)
                        if use_sample:
                            y_hat[:, itr, :] = gaussian_sampler(gaussian_output[..., 0], gaussian_output[..., 1],
                                                                gaussian_output[..., 2], gaussian_output[..., 3],
                                                                gaussian_output[..., 4])
                        else:
                            y_hat[:, itr, :] = gaussian_output[:, itr, 0:2]

                        if itr == pred_len - 1:
                            break

                        itr_x = to_device(torch.zeros((batch_size, 1, 2)), device)
                        itr_x[:, :, :] = y_hat[:, itr, :]
                        output, hc = self(itr_x, hc)

                    # add sample result
                    sample_distribution.append(gaussian_output)
                    sample_location.append(y_hat)
            elif self.loss == 'mixed':
                y_hat = torch.zeros((batch_size, pred_len, 2), device=device)
                mixed_output = torch.zeros((batch_size, pred_len, 5), device=device)

                # initial hidden state
                output, hc = self(datax, hc=None)
                output = torch.unsqueeze(output[:, -1, :], dim=1)

                # predict iterative
                for itr in range(pred_len):
                    mixed_output[:, itr, :] = get_mixed(output)
                    y_hat[:, itr, :] = mixed_output[:, itr, 0:2]

                    if itr == pred_len - 1:
                        break

                    itr_x = torch.zeros((batch_size, 1, 2), device=device)
                    itr_x[:, :, :] = y_hat[:, itr, :]
                    output, hc = self(itr_x, hc)

                sample_distribution.append(mixed_output)
                sample_location.append(y_hat)
            else:
                raise Exception('No inference support for {}'.format(self.loss))

        return {
            'sample_pred_distribution': torch.cat(sample_distribution, dim=0).permute(1, 0, 2, 3),
            'sample_y_hat': torch.cat(sample_location, dim=0).permute(1, 0, 2, 3)
        }
