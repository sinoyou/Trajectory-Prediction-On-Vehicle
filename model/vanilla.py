import argparse
import os
import time

from torch import nn, optim
import torch
from model.utils import make_mlp


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

        self.input_emd_layer = make_mlp([self.input_dim, self.emd_size],
                                        batch_norm=batch_norm, dropout=dropout, activation='relu')
        self.output_emd_layer = make_mlp([self.cell_size, self.output_dim],
                                         batch_norm=batch_norm, dropout=dropout, activation=None)
        self.rnn_cell = nn.LSTMCell(self.emd_size, self.cell_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, hc=None):
        """
        :param inputs: [batch_size, length, input_dim]
        :param hc: initial hidden state
        :return: outputs: [batch_size, length, output_dim]
        """
        seq_length = inputs.shape[1]
        outputs = []
        for step in range(seq_length):
            step_input = inputs[:, step, :]
            emd_input = self.input_emd_layer(step_input)
            hc = self.rnn_cell(input=emd_input, hx=hc)
            emd_output = self.output_emd_layer(hc[0])
            outputs.append(emd_output)
        return torch.stack(outputs, dim=1), hc


def vanilla_train_data_splitter(batch_data, pred_len):
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


def vanilla_evaluation_data_splitter(batch_data, pred_len):
    """
    Split data [batch_size, total_len, 2] into datax and datay in evaluation mode
    :param batch_data: data to be split
    :param pred_len: lengthof trajectories in final loss calculation
    :return: datax, datay
    """
    total_len = batch_data.shape[1]
    datax = batch_data[:, 0:total_len - pred_len, :]
    datay = batch_data[:, -pred_len:, :]
    return datax, datay
