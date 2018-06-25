import torch

from socr.models.convolutional_model import ConvolutionalModel
from socr.models.loss import CTCTextLoss
from socr.nn import RNNLayer


class OcropyLineNetwork(ConvolutionalModel):

    def __init__(self, labels):
        super().__init__()

        self.labels = labels
        self.output_numbers = len(self.labels) + 1

        self.activation = torch.nn.ReLU()

        self.convolutions_output_size = self.get_cnn_output_size()

        self.lstm = RNNLayer(self.convolutions_output_size[1] * self.convolutions_output_size[2], 100, rnn_type=torch.nn.LSTM, bidirectional=True, batch_norm=False, biadd=False)
        self.fc = torch.nn.Linear(200, self.output_numbers)

    def forward_cnn(self, x):
        return x

    def forward_fc(self, x):

        batch_size = x.data.size()[0]
        channel_num = x.data.size()[1]
        height = x.data.size()[2]
        width = x.data.size()[3]

        x = x.view(batch_size, channel_num * height, width)
        # x is (batch_size x hidden_size x width)
        x = torch.transpose(x, 1, 2)
        # x is (batch_size x width x hidden_size)
        x = torch.transpose(x, 0, 1).contiguous()

        x = self.lstm(x)

        x = self.fc(x)

        x = x.transpose(0, 1)

        if not self.training:
            x = torch.nn.functional.softmax(x, dim=2)

        return x

    def get_input_image_width(self):
        return None

    def get_input_image_height(self):
        return 48

    def create_loss(self):
        return CTCTextLoss(self.labels)