"""
This is a tensorflow-multi-dimensional-lstm port by Belos Thomas
The original code came from https://raw.githubusercontent.com/philipperemy/tensorflow-multi-dimensional-lstm/master/md_lstm.py
"""
import torch
from torch.nn.modules.normalization import LayerNorm


class MultiDimensionalLSTMCell(torch.nn.Module):

    def __init__(self, features, num_units, forget_bias=0.0, activation=torch.nn.Tanh, normalize=False):
        super(MultiDimensionalLSTMCell, self).__init__()
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation()
        self.normalize = normalize

        # change bias argument to False since LN will add bias via shift
        self.linear = torch.nn.Linear(features + 2 * self._num_units,5 * self._num_units, bias=not normalize)

        if normalize:
            self.ln_i = LayerNorm(self._num_units)
            self.ln_j = LayerNorm(self._num_units)
            self.ln_f1 = LayerNorm(self._num_units)
            self.ln_f2 = LayerNorm(self._num_units)
            self.ln_o = LayerNorm(self._num_units)
            self.ln_new_c = LayerNorm(self._num_units)

    def forward(self, inputs, state, scope=None):
        c1, c2, h1, h2 = state

        concat = torch.cat((inputs, h1, h2), dim=1)
        concat = self.linear(concat)

        i, j, f1, f2, o = torch.split(concat, self._num_units, dim=1)

        # add layer normalization to each gate
        if self.normalize:
            i = self.ln_i(i)
            j = self.ln_j(j)
            f1 = self.ln_f1(f1)
            f2 = self.ln_f2(f2)
            o = self.ln_o(o)

        new_c = (c1 * torch.nn.functional.sigmoid(f1 + self._forget_bias) +
                c2 * torch.nn.functional.sigmoid(f2 + self._forget_bias) + torch.nn.functional.sigmoid(i) *
                self._activation(j))

        # add layer_normalization in calculation of new hidden state
        if self.normalize:
            new_h = self._activation(self.ln_new_c(new_c)) * torch.nn.functional.sigmoid(o)
        else:
            new_h = self._activation(new_c) * torch.nn.functional.sigmoid(o)

        return new_h, (new_c, new_h)


class MultiDimensionalLSTM(torch.nn.Module):

    # input_data: the data to process of shape [batch,h,w,channels]

    def __init__(self, channels, win_dim, rnn_size, reverted=False):
        super(MultiDimensionalLSTM, self).__init__()

        self.rnn_size = rnn_size
        self.reverted = reverted
        self.channels = channels

        self.X_win, self.Y_win = win_dim
        self.features = self.Y_win * self.X_win * self.channels

        self.cell = MultiDimensionalLSTMCell(self.features, rnn_size)

    def forward(self, x):
        batch_size = x.data.size()[0]
        channels = x.data.size()[1]
        X_dim, Y_dim = x.data.size()[3], x.data.size()[2]

        # If the input cannot be exactly sampled by the window, we patch it with zeros
        if X_dim % self.X_win != 0:
            raise NotImplementedError("Sorry :( The size must correspond")

        # The same but for Y axis
        if Y_dim % self.Y_win != 0:
            raise NotImplementedError("Sorry :( The size must correspond")

        h, w = int(X_dim / self.X_win), int(Y_dim / self.Y_win)

        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(batch_size, h, w, self.features)
        x = x.permute(1, 2, 0, 3)
        x = x.contiguous().view(h * w * batch_size, self.features)
        inputs_ta = torch.split(x, batch_size, dim=0) # So we have h * w input of [batch_size, self.features size]
        outputs_ta = [None] * (h * w)
        states_ta = [None] * w

        if x.is_cuda:
            zero = torch.autograd.Variable(torch.zeros([batch_size, self.rnn_size]), requires_grad=False).cuda()
        else:
            zero = torch.autograd.Variable(torch.zeros([batch_size, self.rnn_size]), requires_grad=False)

        # Controls the initial index
        time = 0

        # Run the looped operation
        while time < h * w:

            rev_time = time if not self.reverted else (h * w - (time + 1))

            # If we are the first line, then use the initial
            if time <= w:
                state_up = zero, zero
            else:
                state_up = states_ta[(time - w) % w]

            # If we are the first cols, then use the initial
            if time % w > 0:
                state_last = states_ta[(time - 1) % w]
            else:
                state_last = zero, zero

            current_state = state_up[0], state_last[0], state_up[1], state_last[1]
            out, state = self.cell(inputs_ta[rev_time], current_state)
            outputs_ta[time] = out

            states_ta[time % w] = state

            time = time + 1

        # Extract the output tensors from the processesed tensor array
        outputs = torch.stack(outputs_ta)
        outputs = outputs.view(h, w, batch_size, self.rnn_size)
        outputs = outputs.permute(2, 0, 1, 3)
        # outputs = outputs.permute(2, 3, 0, 1)

        return outputs.contiguous()


class BidirectionalMultiDimensionalLSTM(torch.nn.Module):

    def __init__(self, channels, win_dim, rnn_size):
        super(BidirectionalMultiDimensionalLSTM, self).__init__()

        self.lstm1 = MultiDimensionalLSTM(channels, win_dim, rnn_size)
        self.lstm2 = MultiDimensionalLSTM(channels, win_dim, rnn_size, reverted=True)

    def forward(self, x):
        self.x1 = self.lstm1(x)
        self.x2 = self.lstm2(x)
        return torch.cat((self.x1, self.x2), dim=3) # Channels dim

class OptimizedMultiDimensionalLSTM(torch.nn.Module):

    # input_data: the data to process of shape [batch,h,w,channels]

    def __init__(self, input_dim, win_dim, rnn_size, reverted=False):
        super(OptimizedMultiDimensionalLSTM, self).__init__()

        self.rnn_size = rnn_size
        self.reverted = reverted

        self.X_dim, self.Y_dim, self.channels = input_dim
        self.X_win, self.Y_win = win_dim

        # If the input cannot be exactly sampled by the window, we patch it with zeros
        if self.X_dim % self.X_win != 0:
            raise NotImplementedError("Sorry :( The size must correspond")

        # The same but for Y axis
        if self.Y_dim % self.Y_win != 0:
            raise NotImplementedError("Sorry :( The size must correspond")

        self.h, self.w = int(self.X_dim / self.X_win), int(self.Y_dim / self.Y_win)
        self.features = self.Y_win * self.X_win * self.channels

        self.cell = MultiDimensionalLSTMCell(self.features, rnn_size)
        # self.cells = [MultiDimensionalLSTMCell(self.features, rnn_size) for i in range(0, self.w * self.h)]
        # self.cells_modulelist = torch.nn.ModuleList(self.cells)

    def forward(self, x):
        batch_size = x.data.size()[0]

        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(batch_size, self.h, self.w, self.features)
        x = x.permute(1, 2, 0, 3)
        x = x.contiguous().view(self.h * self.w * batch_size, self.features)
        inputs_ta = torch.split(x, batch_size, dim=0) # So we have h * w input of [batch_size, self.features size]
        outputs_ta = [None] * (self.h * self.w)
        states_ta = [None] * (self.h * self.w + 1)

        if x.is_cuda:
            states_ta[self.h * self.w] = (
                torch.autograd.Variable(torch.zeros([batch_size, self.rnn_size]), requires_grad=False).cuda(),
                torch.autograd.Variable(torch.zeros([batch_size, self.rnn_size]), requires_grad=False).cuda())
        else:
            states_ta[self.h * self.w] = (
                torch.autograd.Variable(torch.zeros([batch_size, self.rnn_size]), requires_grad=False),
                torch.autograd.Variable(torch.zeros([batch_size, self.rnn_size]), requires_grad=False))

        # TODO : states can be stored as diagonal * ???

        diagonal = 0
        while diagonal < (self.h + self.w - 1):
            # Get the diagonal
            if diagonal < self.w:
                diagonal_size = diagonal + 1
                diagonal_start = diagonal
            else:
                diagonal_size = self.w - (diagonal - self.w) - 1
                diagonal_start = (diagonal - self.w + 2) * self.w - 1

            diagonal_elements = [inputs_ta[i + (self.h * i)] for i in range(0, diagonal_size)]

            # Get the states
            c1, c2, h1, h2 = [], [], [], []
            time = diagonal_start
            for i in range(0, diagonal_size):

                # If we are the first line, then use the initial
                if time <= self.w:
                    state_up = states_ta[self.h * self.w]
                else:
                    state_up = states_ta[time - self.w]

                # If we are the first cols, then use the initial
                if time % self.w > 0:
                    state_last = states_ta[time - 1]
                else:
                    state_last = states_ta[self.h * self.w]

                c1.append(state_up[0])
                c2.append(state_last[0])
                h1.append(state_up[1])
                h2.append(state_last[1])

                time = time + self.w - 1

            c1 = torch.stack(c1).view(batch_size * diagonal_size, self.rnn_size)
            c2 = torch.stack(c2).view(batch_size * diagonal_size, self.rnn_size)
            h1 = torch.stack(h1).view(batch_size * diagonal_size, self.rnn_size)
            h2 = torch.stack(h2).view(batch_size * diagonal_size, self.rnn_size)

            # Stack elements so we have a matrix of [diagnol, batch_size, self.feature_size]
            diagonal_elements = torch.stack(diagonal_elements)

            # Consider diagonal as batch
            diagonal_elements = diagonal_elements.view(diagonal_size * batch_size, self.features)

            # The returned state is two matrix of size [diagonal * batch_size, rnn_size]
            out, state = self.cell(diagonal_elements, (c1, c2, h1, h2))
            new_c, new_h = state

            # Split elements
            out = torch.split(out, batch_size, dim=0)
            new_c = torch.split(new_c, batch_size, dim=0)
            new_h = torch.split(new_h, batch_size, dim=0)

            # Store the output
            time = diagonal_start
            for i in range(0, diagonal_size):
                outputs_ta[time] = out[i]
                states_ta[time] = (new_c[i], new_h[i])
                time = time + self.w - 1

            diagonal = diagonal + 1


        # Extract the output tensors from the processesed tensor array
        outputs = torch.stack(outputs_ta)
        outputs = outputs.view(self.h, self.w, batch_size, self.rnn_size)
        outputs = outputs.permute(2, 0, 1, 3)
        # outputs = outputs.permute(2, 3, 0, 1)

        return outputs.contiguous()