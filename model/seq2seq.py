from torch import nn
import torch
from model.utils import make_mlp

from model.utils import get_2d_gaussian, gaussian_sampler
from script.cuda import to_device


class Seq2SeqLSTM(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=5,
                 pred_length=4,
                 emd_size=128,
                 cell_size=128,
                 batch_norm=True,
                 dropout=0.0):
        """
        Implement of Seq2Seq LSTM structure
        Input sequence length can vary!
        :param input_dim: 2
        :param output_dim: 5 as 2D Gaussian Distribution
        :param pred_length: 4
        :param emd_size: 128
        :param cell_size: 128
        :param batch_norm: batch normalization
        :param dropout: dropout in mlp
        """
        super(Seq2SeqLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_length = pred_length
        self.emd_size = emd_size
        self.cell_size = cell_size

        self.input_emd_layer = make_mlp([self.input_dim, self.emd_size],
                                        activation='rule', batch_norm=batch_norm, dropout=dropout)
        self.output_emd_layer = make_mlp([self.cell_size, self.output_dim],
                                         activation=None, batch_norm=batch_norm, dropout=dropout)
        self.encoder = nn.LSTM(self.emd_size, self.cell_size, batch_first=True)
        self.decoder_cell = nn.LSTMCell(self.emd_size, self.cell_size)
        # self.decoder = nn.LSTM(self.cell_size, self.cell_size, batch_first=True)

    def forward(self, inputs, output_parser=None):
        """
        :param inputs: shape as [batch_size, obs_length, input_dim]
        :param output_parser: parse output (indifferential) when doing interface.
        :return: outputs [batch_size, pred_length, output_dim]
        """
        assert inputs.shape[2] == self.input_dim
        # encoding
        seq_len = inputs.shape[1]
        padded_inputs = inputs.reshape((-1, self.input_dim))
        embedding_inputs = self.input_emd_layer(padded_inputs)
        embedding_inputs = embedding_inputs.view((-1, seq_len, self.emd_size))

        output, hc = self.encoder(embedding_inputs)

        # # decoding - legacy
        # h = hc[0]
        # dec_input = h.repeat((self.pred_length, 1, 1))
        # dec_output, dec_hc = self.decoder(dec_input)
        # outputs = list()
        # for step in range(self.pred_length):
        #     output = torch.squeeze(dec_output[:, step, :], dim=1)
        #     emd_output = self.output_emd_layer(output)
        #     outputs.append(emd_output)

        # decoding
        outputs = []
        hx = torch.squeeze(hc[0], dim=0)
        hx = (hx, torch.zeros_like(hx, device=hx.device))  # (h, c=0)
        prev_pos = inputs[:, -1, :]
        for step in range(self.pred_length):
            emd_input = self.input_emd_layer(prev_pos)
            hx = self.decoder_cell(input=emd_input, hx=hx)
            emd_output = self.output_emd_layer(hx[0])
            if output_parser:
                prev_pos = output_parser(emd_output)
            else:
                prev_pos = emd_output[:, 0:2]
            outputs.append(emd_output)

        return torch.stack(outputs, dim=1), hx[0]

    @staticmethod
    def train_data_splitter(batch_data, pred_len):
        """
        Split data [batch_size, total_len, 2] into datax and datay in train mode
        :param batch_data: data to be split
        :param pred_len: length of trajectories in final loss calculation
        :return: datax, datay
        """
        return batch_data[:, :-pred_len, :], batch_data[:, -pred_len:, :]

    @staticmethod
    def evaluation_data_splitter(batch_data, pred_len):
        """
        Split data [batch_size, total_len, 2] into datax and datay in val/evaluation mode
        :param batch_data: data to be split
        :param pred_len: lengthof trajectories in final loss calculation
        :return: datax, datay
        """
        return batch_data[:, :-pred_len, :], batch_data[:, -pred_len:, :]

    @staticmethod
    def interface(model, input_x, pred_len, sample_times):
        """
        During evaluation, use trained model to interface.
        :param model: Loaded Vanilla Model
        :param input_x: obs data [1, obs_len, 2]
        :param pred_len: length of prediction
        :param sample_times: times of sampling trajectories
        :return: gaussian_output [sample_times, pred_len, 5], location_output[sample_times, pred_len, 2]
        """
        sample_gaussian = list()
        sample_location = list()

        def interface_output_parser(x):
            """
            Only used in interface!
            :param x: (1, 5)
            """
            x = get_2d_gaussian(x)
            sample_location = gaussian_sampler(x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4])
            return torch.tensor(sample_location, device=x.device).view(1, 2)

        with torch.no_grad():
            for _ in range(sample_times):
                model_outputs, _ = model(input_x, interface_output_parser)

                gaussian_output = to_device(torch.zeros((1, pred_len, 5)), input_x.device)
                rel_y_hat = to_device(torch.zeros((1, pred_len, 2)), input_x.device)

                for itr in range(pred_len):
                    model_output = model_outputs[:, itr, :]
                    gaussian_output[:, itr, :] = get_2d_gaussian(model_output=model_output)
                    rel_y_hat[0, itr, 0], rel_y_hat[0, itr, 1] = gaussian_sampler(
                        gaussian_output[0, itr, 0].cpu().numpy(),
                        gaussian_output[0, itr, 1].cpu().numpy(),
                        gaussian_output[0, itr, 2].cpu().numpy(),
                        gaussian_output[0, itr, 3].cpu().numpy(),
                        gaussian_output[0, itr, 4].cpu().numpy())

                sample_gaussian.append(gaussian_output)
                sample_location.append(rel_y_hat)

        return torch.cat(sample_gaussian, dim=0), torch.cat(sample_location, dim=0)
