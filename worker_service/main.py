import logging
import pickle as pkl
import time
import torch
import threading

import data
import model
from communication import Communication

if __name__ == '__main__':
    print("Starting the y2 worker.")

    # Configure the loggers
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

    # Start the communications
    communication_crashed = threading.Event()
    communication = Communication(communication_crashed)

    decoding_map = pkl.load(open("tools/dec_map.pkl", "rb"))
    decoding_string = ''.join([e for e in decoding_map.values()])

    # Initialize the RNN
    rnn = model.RecurrentModel(13, 500, 5, 30, communication)
    load = torch.load('experiments/model_2.pkl')
    rnn.load_state_dict(load['state_dict'])
    rnn.eval()

    # Initializer data loaders
    val_loader = data.val_loader()
    train_loader = data.train_loader()

    # Perform the actual training
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        print('\rBatch {:03}/{:03}'.format(batch_idx + 1, len(val_loader)), end='')
        output = rnn(inputs)
        targets_text = []
        for t in targets:
            target = []
            for ch in list(t):
                if ch.item() == 29:
                    target.append('')
                else:
                    target.append(decoding_map[ch.item() - 1])
            targets_text.append(target)
        output_text = []
        for o in output[0]:
            o = [e.item() for e in o]
            o = o[1:]
            o = o[:-1]
            max_index = o.index(max(o))
            output_text.append(decoding_map[max_index])
        print(''.join([s for s in targets_text[0]]))
        print(''.join([s for s in output_text]))

    try:
        while not communication_crashed.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        communication.stop()
