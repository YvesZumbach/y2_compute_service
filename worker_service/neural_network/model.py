
from torch import nn

from worker_service.communications.communication import Communication


class RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, communication: Communication):
        super(RecurrentModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size

        self.recurrent = nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.n_layers,
                                nonlinearity='tanh',
                                bias=True)
        self.output = nn.Linear(in_features=self.hidden_size,
                                out_features=self.output_size,
                                bias=True)
        self.softmax = nn.LogSoftmax()

        self.communication = communication

    def forward(self, x):
        # x: batch_size, length, n_features
        hidden = None
        rnn_output, hidden = self.recurrent(x, hidden)
        # rnn_output: batch_size, length, n_hidden
        # hidden: 5, length, n_hidden
        rnn_output_flat = rnn_output.view(-1, self.hidden_size)
        # rnn_output_flat: batch_size*length, n_hidden
        lin_output = self.output(rnn_output_flat)
        # lin_output: batch_size*length, n_out
        output_flat = self.softmax(lin_output)
        # output_flat: batch_size*length, n_out
        output = output_flat.view(rnn_output.size(0), rnn_output.size(1), output_flat.size(1))
        # output: batch_size, length, n_out

        return output
