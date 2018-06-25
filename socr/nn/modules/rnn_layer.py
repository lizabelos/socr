# Code from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py

import torch


class SequenceWise(torch.nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class RNNLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=torch.nn.GRU, bidirectional=True, biadd=True, batch_norm=True, input_view=False, output_view=False, dropout=0):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.biadd = biadd
        self.batch_norm = SequenceWise(torch.nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False, dropout=dropout)
        self.num_directions = 2 if bidirectional else 1
        self.input_view = input_view
        self.output_view = output_view

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.input_view or self.output_view:
            batch_size = x.data.size()[0]
            channel_num = x.data.size()[1]
            height = x.data.size()[2]
            width = x.data.size()[3]

        if self.input_view:
            x = x.view(batch_size, channel_num * height, width)
            #x.permute(2, 0, 1)
            x = torch.transpose(x, 1, 2)
            x = torch.transpose(x, 0, 1).contiguous()

        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional and self.biadd:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum

        if self.output_view:
            #x.permute(1, 2, 0)
            x = torch.transpose(x, 0, 1)
            x = torch.transpose(x, 1, 2)
            x = x.contiguous().view(batch_size, channel_num, height, width)

        return x
