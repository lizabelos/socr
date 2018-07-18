import torch.nn as nn

from socr.models.model import Model
from socr.nn import IndRNN


class LM(Model):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken):
        super(LM, self).__init__()
        self.encoder = nn.Embedding(ntoken, 256)

        self.rnn = nn.GRU(256, 256, num_layers=3, bidirectional=False, dropout=0.3)

        self.decoder = nn.Linear(256, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden=None):
        emb = self.encoder(input)
        if hidden is None:
            output, hidden = self.rnn(emb)
        else:
            output, hidden = self.rnn(emb, hidden)

        output = output.contiguous()

        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
