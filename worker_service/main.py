import logging
import threading
import time
from timeit import default_timer as timer

from communications.communication import Communication
import neural_network.data as data
import neural_network.all_data as all_data
from neural_network.model import RecurrentModel
from neural_network.model_trainer import ModelTrainer

if __name__ == '__main__':
    # Configure the loggers
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info('Starting the y2 worker.')

    # Start the communications
    communication_crashed = threading.Event()
    communication = Communication(communication_crashed)

    messages = communication.receive(0)
    while messages.empty():
        time.sleep(.5)
        messages = communication.receive(0)

    message = messages.get_nowait()
    node_id = int.from_bytes(message[:4], byteorder='big')
    total_nodes = int.from_bytes(message[-4:], byteorder='big')
    log.info("This worker received node id " + str(node_id) + " in a total of " + str(total_nodes) + " nodes.")

    # Instantiate the NN
    rnn = RecurrentModel(13, 200, 4, 30, communication)

    # initializer data loaders
    val_loader = data.val_loader(node_id, total_nodes)
    train_loader = data.train_loader(node_id, total_nodes)

    # initialize data loaders with all values for first epochs
    all_val_loader = all_data.val_loader()
    all_train_loader = all_data.train_loader()

    # initialize trainer
    n_epochs = 10
    start = timer()
    trainer = ModelTrainer(rnn, val_loader, train_loader, all_val_loader, all_train_loader, n_epochs, True)

    # run the training
    trainer.train()
    end = timer()
    print(end - start)
    # trainer.save()

    # Send the Finished message
    communication.send(3, bytearray(0))
    communication.stop()
