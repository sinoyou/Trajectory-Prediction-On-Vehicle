from torch import nn
import torch


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    """Factory for multi-layer perceptron."""
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class VanillaLSTM(torch.nn.Module):

    def __init__(self, input_dim=2, output_dim=5,
                 emd_size=128, cell_size=128):
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
        """
        super(VanillaLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emd_size = emd_size
        self.cell_size = cell_size

        self.input_emd_layer = make_mlp([self.input_dim, self.emd_size])
        self.output_emd_layer = make_mlp([self.cell_size, self.output_dim])
        self.rnn_cell = nn.LSTMCell(self.input_dim, self.cell_size)

    def forward(self, inputs):
        """
        :param inputs: [batch_size, length, input_dim]
        :return: outputs: [batch_size, length, output_dim]
        """
        seq_length = inputs.shape[1]
        outputs = []
        hx = None
        for step in range(seq_length):
            step_input = inputs[:, step, :]
            emd_input = self.input_emd_layer(step_input)
            hx = self.rnn_cell(input=emd_input, hx=hx)
            emd_output = self.output_emd_layer(hx[0])
            outputs.append(emd_output)
        return torch.stack(outputs, dim=1)


class Seq2SeqLSTM(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=5,
                 obs_length=8, pred_length=4,
                 emd_size=128, cell_size=128):
        """
        Implement of Seq2Seq LSTM structure
        Input sequence length can vary!
        :param input_dim: 2
        :param output_dim: 5 as 2D Gaussian Distribution
        :param pred_length: 4
        :param emd_size: 128
        :param cell_size: 128
        """
        super(Seq2SeqLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_length = pred_length
        self.emd_size = emd_size
        self.cell_size = cell_size

        self.input_emd_layer = make_mlp([self.input_dim, self.emd_size])
        self.output_emd_layer = make_mlp([self.cell_size, self.output_dim])
        self.encoder = nn.LSTM(self.emd_size, self.cell_size, batch_first=True)
        self.decoder_cell = nn.LSTMCell(self.emd_size, self.cell_size)

    def forward(self, inputs, output_parser=None):
        """
        :param inputs: shape as [batch_size, obs_length, input_dim]
        :param output_parser: differential operation, parse output of each step into input_dim format
         which can be accepted as next input to decoder cell.
        :return: outputs [batch_size, pred_length, output_dim]
        """
        assert inputs.shape(2) == self.input_dim
        # encoding
        output, hx = self.encoder(inputs)

        # decoding
        outputs = []
        hx = hx
        prev_pos = inputs[:, -1, :]
        for step in range(self.pred_length):
            emd_input = self.input_emd_layer(prev_pos)
            hx = self.decoder_cell(input=emd_input, hx=hx)
            emd_output = self.output_emd_layer(hx[0])
            prev_pos = output_parser(emd_output)
            outputs.append(emd_output)
        return torch.stack(outputs, dim=1)
