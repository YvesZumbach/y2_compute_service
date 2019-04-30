
import asyncio
import os
import numpy as np
import torch
from torch import nn
from timeit import default_timer as timer

from communication import Communication


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


class ModelTrainer():
    _delta = 2.0

    def __init__(self, model, val_loader, train_loader, n_epochs, communicate):
        self.model = model
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.n_epochs = n_epochs
        self.epoch = 0

        self.gpu = False
        self.val_losses = []
        self.train_losses = []

        self.communicate = communicate

        if self.gpu:
            self.model = self.model.cuda()

        self.residuals = []
        for tensor in self.model.parameters():
            y = tensor.clone().detach()
            self.residuals.append(y)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.008)
        self.criterion = nn.CTCLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)

    def train(self):

        for _ in range(self.epoch, self.n_epochs):
            start = timer()
            self.model.train()

            epoch_loss = 0
            epoch_decompressing_time = 0
            epoch_training_time = 0
            epoch_compressing_time = 0

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                print('\rBatch {:03}/{:03} Current Loss: {:7.4f}'.format(batch_idx + 1, len(self.train_loader),
                                                                         epoch_loss), end='')
                local_loss, local_training, local_compress, local_decompress = self.train_batch(inputs, targets)
                epoch_loss += local_loss
                epoch_training_time += local_training
                epoch_compressing_time += local_compress
                epoch_decompressing_time += local_decompress

            epoch_loss = epoch_loss / len(self.train_loader)

            self.train_losses.append(epoch_loss)

            self.epoch += 1
            end = timer()
            if self.communicate:
                # of samples = 256 * n.epoch
                sample_size = 256 * len(self.train_loader)
                runtime_message = bytearray(0)
                runtime_message.join(int(sample_size).to_bytes(4, byteorder="big"))
                runtime_message.join(int(epoch_decompressing_time * 1000).to_bytes(4, byteorder="big"))
                runtime_message.join(int(epoch_training_time * 1000).to_bytes(4, byteorder="big"))
                runtime_message.join(int(epoch_compressing_time * 1000).to_bytes(4, byteorder="big"))
                runtime_message.join(int(epoch_loss * 1000).to_bytes(4, byteorder="big"))
                self.model.communication.send(2, runtime_message)

            print('\r[TRAIN] Epoch {:02}/{:02} Loss {:7.4f}'.format(
                self.epoch, self.n_epochs, epoch_loss
            ), end='\t')
            print('Time elapsed: {}'.format(end-start))
            self.scheduler.step()

    def train_batch(self, inputs, targets):
        compress_time = 0
        decompress_time = 0
        start_batch = timer()
        if self.gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
        outputs = self.model(inputs)

        targets = targets.type(torch.LongTensor)
        input_lens = np.sum(inputs.detach().numpy()[:,:,0] != -1, axis=1)
        targets_lens = np.sum(targets.detach().numpy() != -1, axis=1)

        loss = self.criterion(outputs.permute(1, 0, 2), targets, tuple(input_lens), tuple(targets_lens))
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        end_batch = timer()
        training_time = end_batch - start_batch

        if self.communicate:
            start_compress = timer()
            deltas_to_send = self.compress_gradients()
            end_compress = timer()
            compress_time = end_compress - start_compress
            self.model.communication.send(1, deltas_to_send)

            messages = self.model.communication.receive(1)
            start_decompress = timer()
            self.decompress_and_apply_messages(messages)
            end_decompress = timer()
            decompress_time = end_decompress - start_decompress

        # applies the gradients to the network
        apply_start = timer()
        self.optimizer.step()
        apply_end = timer()
        training_time += apply_end - apply_start

        return_loss = loss.detach().cpu().item()

        return return_loss, training_time, compress_time, decompress_time

    def save(self):
        torch.save(
            {'state_dict': self.model.state_dict()},
            os.path.join('experiments', 'model_{}.pkl'.format(self.epoch))
        )

        with open(os.path.join('experiments', 'train_losses.txt'), 'w') as fw:
            for i in range(len(self.train_losses)):
                fw.write('{:02} {:10.6}\n'.format(i, self.train_losses[i]))

    def compress_gradients(self):
        message = bytearray(0)
        parameters = list(self.model.parameters())
        count = 0
        for i in range(len(parameters)):
            tensor = parameters[i]
            dimensions = len(tensor.size())
            if dimensions == 1:
                for j in range(len(tensor)):
                    self.residuals[i][j] += tensor.grad[j]
                    tensor.grad[j] = 0.0
                    if abs(self.residuals[i][j]) >= ModelTrainer._delta:
                        count += 1
                        tensor_index = i << 22
                        weight_index = j << 1
                        delta = tensor_index | weight_index
                        if self.residuals[i][j] >= ModelTrainer._delta:
                            delta |= 1
                            self.residuals[i][j] -= ModelTrainer._delta
                            tensor.grad[j] = ModelTrainer._delta
                        else:
                            self.residuals[i][j] += ModelTrainer._delta
                            tensor.grad[j] = -ModelTrainer._delta
                        message.join(delta.to_bytes(4, byteorder="big"))
            else:
                first_dim = tensor.size()[0]
                second_dim = tensor.size()[1]
                for first in range(first_dim):
                    for second in range(second_dim):
                        self.residuals[i][first][second] += tensor.grad[first][second]
                        tensor.grad[first][second] = 0.0
                        if abs(self.residuals[i][first][second]) >= ModelTrainer._delta:
                            count += 1
                            tensor_index = i << 22
                            linear_index = first * second_dim + second
                            weight_index = linear_index << 1
                            delta = tensor_index | weight_index
                            if self.residuals[i][first][second] >= ModelTrainer._delta:
                                delta |= 1
                                self.residuals[first][second] -= ModelTrainer._delta
                                tensor.grad[first][second] = ModelTrainer._delta
                            else:
                                self.residuals[i][first][second] += ModelTrainer._delta
                                tensor.grad[first][second] = -ModelTrainer._delta
                            message.join(delta.to_bytes(4, byteorder="big"))
        print("Total num of msgs")
        print(count)
        return message

    def decompress_and_apply_messages(self, messages: asyncio.Queue):
        weight_index_mask = 0b111111111111111111111
        while not messages.empty():
            message = messages.get_nowait()
            for i in range(0, len(message), 4):
                delta = message[i:i+4]
                int_msg = int.from_bytes(delta, byteorder='big')
                is_positive = int_msg & 0b1
                int_msg >>= 1
                weight_index = int_msg & weight_index_mask
                int_msg >>= 21
                tensor_index = int_msg
                tensor = self.model.parameters()[tensor_index]
                dimensions = tensor.size()
                if len(dimensions) == 1:
                    tensor.grad[weight_index] += ModelTrainer._delta if is_positive else -ModelTrainer._delta
                else:
                    second_dim = dimensions[1]
                    second = weight_index % second_dim
                    first = weight_index / second_dim
                    tensor.grad[first][second] += ModelTrainer._delta if is_positive else -ModelTrainer._delta
