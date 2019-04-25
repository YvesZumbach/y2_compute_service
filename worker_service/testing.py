import data
import model
from timeit import default_timer as timer
import torch
import pickle as pkl

decoding_map = pkl.load(open("tools/dec_map.pkl", "rb"))

# initialize the RNN
rnn = model.RecurrentModel(13, 500, 5, 30)
load = torch.load('experiments/model_2.pkl')
rnn.load_state_dict(load['state_dict'])
rnn.eval()

# initializer data loaders
val_loader = data.val_loader()
train_loader = data.train_loader()


for batch_idx, (inputs, targets) in enumerate(val_loader):
    print('\rBatch {:03}/{:03}'.format(batch_idx + 1, len(val_loader)), end='')
    output = rnn(inputs)
    print(targets[0])
    print(targets[0][0])
    targets_text = []
    for t in targets:
        target = []
        for ch in list(t):
            if ch.item() == 0:
                target.append('')
            else:
                target.append(decoding_map[ch.item() - 1])
        targets_text.append(target)
    output_text = []
    for o in output[0]:
        o_val = [it.item() for it in o]
        index_max = o_val.index(max(o_val))
        if index_max == 0:
            output_text.append('')
        else:
            output_text.append(decoding_map[index_max - 1])

    print(targets_text[0])
    print(output_text)
