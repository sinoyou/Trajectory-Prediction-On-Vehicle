from torch import nn
import torch
from model.utils import make_mlp, get_loss_by_name

from model.utils import get_2d_gaussian, gaussian_sampler, get_mixed
from script.cuda import to_device


class Seq2SeqLSTM(torch.nn.Module):
    def __init__(self, input_dim, output_dim, pred_length, emd_size, cell_size, batch_norm, dropout, loss):
        """
        Implement of Seq2Seq LSTM structure
        Input sequence length can vary
        :param input_dim: 2
        :param output_dim: 5 as 2D Gaussian Distribution
        :param pred_length: 4
        :param emd_size: 128
        :param cell_size: 128
        :param batch_norm: batch normalization
        :param dropout: dropout in mlp
        :param loss: loss type - 2d_gaussian, mixed
        """
        super(Seq2SeqLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_length = pred_length
        self.emd_size = emd_size
        self.cell_size = cell_size
        self.loss = loss

        self.encoder_norm = torch.nn.BatchNorm1d(self.input_dim) if batch_norm else None
        self.decoder_norm = torch.nn.BatchNorm1d(self.input_dim) if batch_norm else None
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
        :param output_parser: parse output (indifferential) when doing inference.
        :return: outputs [batch_size, pred_length, output_dim]
        """
        assert inputs.shape[2] == self.input_dim
        seq_len = inputs.shape[1]

        # save the last observed trajectory in advance, in case of batch norm.
        prev_pos = inputs[:, -1, :]

        # normalization
        if self.encoder_norm:
            inputs = inputs.reshape(-1, 2)
            inputs = self.encoder_norm(inputs)
            inputs = inputs.view((-1, seq_len, self.input_dim))

        # encoding
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
        for step in range(self.pred_length):
            if self.decoder_norm:
                prev_pos = self.decoder_norm(prev_pos)
            emd_input = self.input_emd_layer(prev_pos)
            hx = self.decoder_cell(input=emd_input, hx=hx)
            emd_output = self.output_emd_layer(hx[0])
            if output_parser:
                prev_pos = output_parser(emd_output)
            else:
                prev_pos = emd_output[:, 0:2]
            outputs.append(emd_output)

        return torch.stack(outputs, dim=1)

    def train_data_splitter(self, batch_data, pred_len):
        """
        Split data [batch_size, total_len, 2] into datax and datay in train mode
        :param batch_data: data to be split
        :param pred_len: length of trajectories in final loss calculation
        :return: datax, datay
        """
        return batch_data[:, :-pred_len, :], batch_data[:, -pred_len:, :]

    def evaluation_data_splitter(self, batch_data, pred_len):
        """
        Split data [batch_size, total_len, 2] into datax and datay in val/evaluation mode
        :param batch_data: data to be split
        :param pred_len: lengthof trajectories in final loss calculation
        :return: datax, datay
        """
        return batch_data[:, :-pred_len, :], batch_data[:, -pred_len:, :]

    def train_step(self, x, **kwargs):
        """
        Run one train step.
        :param x: [batch_size, obs_len, 2]
        :return: dict()
        """
        model_output = self(x)
        return {'model_output': model_output}

    def get_loss(self, distribution, y_gt):
        return get_loss_by_name(distribution, y_gt, self.loss)

    def inference(self, datax, pred_len, sample_times, use_sample):
        """
        During evaluation, use trained model to inference.
        :param model: Loaded Vanilla Model
        :param datax: obs data [1, obs_len, 2]
        :param pred_len: length of prediction
        :param sample_times: times of sampling trajectories
        :param use_sample: if True, applying sample in inference. if False, use average value in gaussian.
        :return: gaussian_output [sample_times, pred_len, 5], location_output[sample_times, pred_len, 2]
        """
        sample_distribution = list()
        sample_location = list()

        if (self.loss != '2d_gaussian') and use_sample:
            raise Exception('No sample support for {}'.format(self.loss))

        def sample_output_parser(x):
            """
            Only used in inference!
            :param x: (1, 5)
            """
            x = get_2d_gaussian(x)
            sample_location = gaussian_sampler(x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4])
            return torch.tensor(sample_location, device=x.device).view(1, 2)

        with torch.no_grad():
            if self.loss == '2d_gaussian':
                for _ in range(sample_times):
                    interface_output_parser = sample_output_parser if use_sample else None
                    model_outputs = self(datax, interface_output_parser)

                    gaussian_output = to_device(torch.zeros((1, pred_len, 5)), datax.device)
                    y_hat = to_device(torch.zeros((1, pred_len, 2)), datax.device)

                    for itr in range(pred_len):
                        model_output = model_outputs[:, itr, :]
                        gaussian_output[:, itr, :] = get_2d_gaussian(model_output=model_output)
                        if use_sample:
                            y_hat[0, itr, 0], y_hat[0, itr, 1] = gaussian_sampler(
                                gaussian_output[0, itr, 0].cpu().numpy(),
                                gaussian_output[0, itr, 1].cpu().numpy(),
                                gaussian_output[0, itr, 2].cpu().numpy(),
                                gaussian_output[0, itr, 3].cpu().numpy(),
                                gaussian_output[0, itr, 4].cpu().numpy())
                        else:
                            y_hat[0, itr, :] = gaussian_output[:, itr, 0:2]

                    sample_distribution.append(gaussian_output)
                    sample_location.append(y_hat)

            elif self.loss == 'mixed':
                interface_output_parser = None
                model_outputs = self(datax, interface_output_parser)
                y_hat = torch.zeros((1, pred_len, 2), device=datax.device)

                mixed_output = torch.zeros((1, pred_len, 5), device=datax.device)
                for itr in range(pred_len):
                    model_output = model_outputs[:, itr, :]

                    mixed_output[0, itr, :] = get_mixed(model_output)
                    y_hat[0, itr, :] = mixed_output[:, itr, 0:2]

                sample_distribution.append(mixed_output)
                sample_location.append(y_hat)

            else:
                raise Exception('No inference support for {}'.format(self.loss))

        return torch.cat(sample_distribution, dim=0), torch.cat(sample_location, dim=0)
