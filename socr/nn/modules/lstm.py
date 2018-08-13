import torch
import numpy as np

def asnd(x, torch_axes=None):
    """Convert torch/numpy to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.IntTensor)):
        x = x.cpu()
    assert isinstance(x, torch.Tensor)
    x = x.numpy()
    if torch_axes is not None:
        x = x.transpose(torch_axes)
    return x


def typeas(x, y):
    """Make x the same type as y, for numpy, torch, torch.cuda."""
    assert not isinstance(x, torch.autograd.Variable)
    if isinstance(y, torch.autograd.Variable):
        y = y.data
    if isinstance(y, np.ndarray):
        return asnd(x)
    if isinstance(x, np.ndarray):
        if isinstance(y, (torch.FloatTensor, torch.cuda.FloatTensor)):
            x = torch.FloatTensor(x)
        else:
            x = torch.DoubleTensor(x)
    return x.type_as(y)


class RowwiseLSTM(torch.nn.Module):
    def __init__(self, ninput=None, noutput=None, ndir=2):
        torch.nn.Module.__init__(self)
        self.ndir = ndir
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = torch.nn.LSTM(ninput, noutput, 1, bidirectional=self.ndir - 1)

    def forward(self, img):
        volatile = not isinstance(img, torch.autograd.Variable) or img.volatile
        b, d, h, w = img.size()
        # BDHW -> WHBD -> WLD
        seq = img.permute(3, 2, 0, 1).contiguous().view(w, h * b, d)
        # WLD
        h0 = torch.zeros(self.ndir, h * b, self.noutput).cuda()
        c0 = torch.zeros(self.ndir, h * b, self.noutput).cuda()
        h0 = torch.autograd.Variable(h0, volatile=volatile)
        c0 = torch.autograd.Variable(c0, volatile=volatile)
        seqresult, _ = self.lstm(seq, (h0, c0))
        # WLD' -> BD'HW
        result = seqresult.view(w, h, b, self.noutput * self.ndir).permute(2, 3, 1, 0)
        return result


class Lstm2D(torch.nn.Module):
    """A 2D LSTM module."""

    def __init__(self, ninput=None, noutput=None, npre=None, nhidden=None, ndir=2, ksize=3):
        torch.nn.Module.__init__(self)
        self.ndir = ndir
        npre = npre or noutput
        nhidden = nhidden or noutput
        self.sizes = (ninput, npre, nhidden, noutput)
        assert ksize % 2 == 1
        padding = (ksize - 1) // 2
        self.conv = torch.nn.Conv2d(ninput, npre, kernel_size=ksize, padding=padding)
        self.hlstm = RowwiseLSTM(npre, nhidden, ndir=ndir)
        self.vlstm = RowwiseLSTM(self.ndir * nhidden, noutput, ndir)

    def forward(self, img, volatile=False):
        ninput, npre, nhidden, noutput = self.sizes
        # BDHW
        filtered = self.conv(img)
        horiz = self.hlstm(filtered)
        horizT = horiz.permute(0, 1, 3, 2).contiguous()
        vert = self.vlstm(horizT)
        vertT = vert.permute(0, 1, 3, 2).contiguous()
        return vertT