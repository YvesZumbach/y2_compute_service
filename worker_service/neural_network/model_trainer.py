
import bisect
import logging
import multiprocessing
import numpy as np
import os
from timeit import default_timer as timer
import torch
from torch import nn


class ModelTrainer:
    _delta = 2.0
    _parallel = False

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

        self.list_parameters = list(self.model.parameters())

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.008)
        self.criterion = nn.CTCLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)

        self.log = logging.getLogger(__name__)

    def train(self):
        total_loss = 0

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
                runtime_message.extend(int(sample_size).to_bytes(4, byteorder="big"))
                runtime_message.extend(int(epoch_decompressing_time * 1000).to_bytes(4, byteorder="big"))
                runtime_message.extend(int(epoch_training_time * 1000).to_bytes(4, byteorder="big"))
                runtime_message.extend(int(epoch_compressing_time * 1000).to_bytes(4, byteorder="big"))
                runtime_message.extend(int(epoch_loss * 1000).to_bytes(4, byteorder="big"))
                self.model.communication.send(2, runtime_message)

            print('\r[TRAIN] Epoch {:02}/{:02} Loss {:7.4f}'.format(
                self.epoch, self.n_epochs, epoch_loss
            ), end='\t')
            print('Time elapsed: {}'.format(end-start))
            self.scheduler.step()
            total_loss += epoch_loss

        print('Total job loss: {}'.format(total_loss))

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

        if self.communicate and self.epoche > 1:
            start_compress = timer()
            if ModelTrainer._parallel:
                deltas_to_send = self.compress_gradients_parallel()
            else:
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
                        delta_bytes = delta.to_bytes(4, byteorder="big")
                        message.extend(delta_bytes)
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
                                self.residuals[i][first][second] -= ModelTrainer._delta
                                tensor.grad[first][second] = ModelTrainer._delta
                            else:
                                self.residuals[i][first][second] += ModelTrainer._delta
                                tensor.grad[first][second] = -ModelTrainer._delta
                            delta_bytes = delta.to_bytes(4, byteorder="big")
                            message.extend(delta_bytes)
        print("Total num of msgs")
        print(count)
        return message

    def decompress_and_apply_messages(self, messages):
        print("Decompressing received messages")
        weight_index_mask = 0b111111111111111111111
        while not messages.empty():
            message = messages.get_nowait()
            print("Received one message")
            print("Length of msg: {}".format(len(message)))
            for i in range(0, len(message), 4):
                delta = message[i:i + 4]
                int_msg = int.from_bytes(delta, byteorder='big')
                is_positive = int_msg & 0b1
                int_msg >>= 1
                weight_index = int_msg & weight_index_mask
                int_msg >>= 21
                tensor_index = int_msg
                tensor = self.model.parameters()[tensor_index]
                dimensions = tensor.size()
                print("Apply delta {} at tensor_index {} at weight_index {}".format(is_positive, tensor_index, weight_index))
                if len(dimensions) == 1:
                    tensor.grad[weight_index] += ModelTrainer._delta if is_positive else -ModelTrainer._delta
                else:
                    second_dim = dimensions[1]
                    second = weight_index % second_dim
                    first = weight_index / second_dim
                    tensor.grad[first][second] += ModelTrainer._delta if is_positive else -ModelTrainer._delta

    def compress_gradients_parallel(self):
        reduced_message = bytearray(0)
        total_count = 0

        cpu_count = multiprocessing.cpu_count()
        with multiprocessing.Pool(cpu_count) as p:
            # We don't care about the ordering of the deltas, so we use the lazy unordered version of parallel map to
            # slightly improve performances
            for message, count in p.imap_unordered(
                    self.compress_gradients_parallel_inner, range(len(self.list_parameters))):
                reduced_message += message
                total_count += count

        self.log.info("Gradient compression finished. Delta message contains " + str(total_count) +
                      " individual deltas to apply.")
        return reduced_message

    def compress_gradients_parallel_inner(self, tensor_index):
        """
        Takes the index of a gradient tensor in the gradient array, add the weight changes in the tensor to the
        corresponding value in the residual tensor. If the value in the residual crosses the predefined delta threshold
        (tau), then the delta value is removed from the residual and transferred into the gradient. For the residual
        values that cross the threshold value a delta message will be created.
        :param tensor_index: The index of the gradient tensor to process in the array of gradient.
        :return: A tuple containing first a single delta message that contains the concatenation of all the deltas to
        apply to the corresponding weight tensor, then the number of single delta to apply contained in the message.
        """
        tensor = self.list_parameters[tensor_index]
        message = bytearray(0)
        count = 0
        dimensions = len(tensor.size())
        if dimensions is 1:
            for weight_index in range(len(tensor)):
                self.residuals[tensor_index][weight_index] += tensor.grad[weight_index]
                tensor.grad[weight_index] = 0.0
                if abs(self.residuals[tensor_index][weight_index]) >= ModelTrainer._delta:
                    count += 1
                    tensor_index = tensor_index << 22
                    weight_index = weight_index << 1
                    delta = tensor_index | weight_index
                    if self.residuals[tensor_index][weight_index] >= ModelTrainer._delta:
                        delta |= 1
                        self.residuals[tensor_index][weight_index] -= ModelTrainer._delta
                        tensor.grad[weight_index] = ModelTrainer._delta
                    else:
                        self.residuals[tensor_index][weight_index] += ModelTrainer._delta
                        tensor.grad[weight_index] = -ModelTrainer._delta
                    message += delta.to_bytes(4, byteorder="big")
        else:
            first_dim = tensor.size()[0]
            second_dim = tensor.size()[1]
            for first in range(first_dim):
                for second in range(second_dim):
                    self.residuals[tensor_index][first][second] += tensor.grad[first][second]
                    tensor.grad[first][second] = 0.0
                    if abs(self.residuals[tensor_index][first][second]) >= ModelTrainer._delta:
                        count += 1
                        tensor_index = tensor_index << 22
                        linear_index = first * second_dim + second
                        weight_index = linear_index << 1
                        delta = tensor_index | weight_index
                        if self.residuals[tensor_index][first][second] >= ModelTrainer._delta:
                            delta |= 1
                            self.residuals[first][second] -= ModelTrainer._delta
                            tensor.grad[first][second] = ModelTrainer._delta
                        else:
                            self.residuals[tensor_index][first][second] += ModelTrainer._delta
                            tensor.grad[first][second] = -ModelTrainer._delta
                        message += delta.to_bytes(4, byteorder="big")
        return message, count
