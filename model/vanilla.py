import argparse
import os
import time

from torch import nn, optim
import torch
from model.utils import make_mlp, get_2d_gaussian, gaussian_sampler, neg_likelihood_gaussian_pdf_loss
from script.cuda import get_device, to_device


class VanillaLSTM(torch.nn.Module):

    def __init__(self, input_dim=2, output_dim=5,
                 emd_size=128, cell_size=128,
                 batch_norm=True, dropout=0):
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
        """
        super(VanillaLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emd_size = emd_size
        self.cell_size = cell_size

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

    @staticmethod
    def train_data_splitter(batch_data, pred_len):
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

    @staticmethod
    def evaluation_data_splitter(batch_data, pred_len):
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

    @staticmethod
    def train_step(vanilla, x, pred_len):
        """
        Run one train step
        :param vanilla: vanilla model
        :param x: [batch_size, obs_len, 2]
        :param pred_len: length of prediction
        :return: dict()
        """
        model_output, _ = vanilla(x, hc=None)
        gaussian_output = get_2d_gaussian(model_output=model_output)
        gaussian_output = gaussian_output[:, -pred_len:, :]
        return {'gaussian_output': gaussian_output}

    @staticmethod
    def interface(model, datax, pred_len, sample_times, use_sample):
        """
        During evaluation, use trained model to interface.
        :param model: Loaded Vanilla Model
        :param datax: obs data [1, obs_len, 2]
        :param pred_len: length of prediction
        :param sample_times: times of sampling trajectories
        :param use_sample: if True, applying sample in interface. if False, use average value in gaussian.
        :return: gaussian_output [sample_times, pred_len, 5], location_output[sample_times, pred_len, 2]
        """
        device = datax.device
        with torch.no_grad():
            sample_gaussian = list()
            sample_location = list()
            for _ in range(sample_times):
                rel_y_hat = to_device(torch.zeros((1, pred_len, 2)), device)
                gaussian_output = to_device(torch.zeros((1, pred_len, 5)), device)

                # initial hidden state
                output, hc = model(datax, hc=None)
                output = torch.unsqueeze(output[:, -1, :], dim=1)

                # predict iterative
                for itr in range(pred_len):
                    gaussian_output[0, itr, :] = get_2d_gaussian(output)
                    if use_sample:
                        rel_y_hat[0, itr, 0], rel_y_hat[0, itr, 1] = gaussian_sampler(
                            gaussian_output[0, itr, 0].cpu().numpy(),
                            gaussian_output[0, itr, 1].cpu().numpy(),
                            gaussian_output[0, itr, 2].cpu().numpy(),
                            gaussian_output[0, itr, 3].cpu().numpy(),
                            gaussian_output[0, itr, 4].cpu().numpy())
                    else:
                        rel_y_hat[0, itr, :] = gaussian_output[0, itr, 0:2]

                    if itr == pred_len - 1:
                        break

                    itr_x_rel = to_device(torch.zeros((1, 1, 2)), device)
                    itr_x_rel[:, :, :] = rel_y_hat[:, itr, :]
                    output, hc = model(itr_x_rel, hc)

                # add sample result
                sample_gaussian.append(gaussian_output)
                sample_location.append(rel_y_hat)

        return torch.cat(sample_gaussian, dim=0), torch.cat(sample_location, dim=0)
