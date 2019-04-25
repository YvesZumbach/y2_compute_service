import data
import model
import torch
import pickle as pkl

decoding_map = pkl.load(open("tools/dec_map.pkl", "rb"))
decoding_string = ''.join([e for e in decoding_map.values()])

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
    # actual = beam_search.ctcBeamSearch(output[0].detach().numpy(), decoding_string, None)
