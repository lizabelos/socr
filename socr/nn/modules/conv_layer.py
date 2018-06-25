import torch


class ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, mpstride=2, bn=True, activation=torch.nn.LeakyReLU(0.1), maxpool=True, same_padding=True):
        """
        A simple convolutional layer

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: kernel size of the convolution
        :param stride: stride of the convolution
        :param bn: if True, use a batch normalized
        :param activation: activation fonction to use (if not None)
        :param maxpool: if True, use a maxpool of kernel 2 and stride 2
        :param same_padding: if True, add a padding so the output of the convolution is the same as the input
        """
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2 if same_padding else 0

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels) if bn else None
        self.activation = activation
        self.maxpooling = torch.nn.MaxPool2d(2, stride=mpstride) if maxpool else None

    def forward(self, x):
        if not isinstance(x, torch.autograd.Variable):
            raise TypeError("x is not a Variable")

        x = self.conv(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.maxpooling is not None:
            x = self.maxpooling(x)

        if self.bn is not None:
            x = self.bn(x)

        return x

