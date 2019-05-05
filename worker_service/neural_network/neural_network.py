import neural_network.data as data
import neural_network.model as model
from neural_network.model_trainer import ModelTrainer
from timeit import default_timer as timer

# initialize the RNN
rnn = model.RecurrentModel(13, 200, 4, 30, None)

# initializer data loaders
val_loader = data.val_loader(0, 1)
train_loader = data.train_loader(0, 1)

# initialize trainer
n_epochs = 10
start = timer()
trainer = ModelTrainer(rnn, val_loader, train_loader, n_epochs, False)

# run the training
trainer.train()
end = timer()
print('Total time elapsed, end of job: {}', end-start)
