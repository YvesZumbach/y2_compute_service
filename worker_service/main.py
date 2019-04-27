import logging
import time
import threading
import data
import model
from communication import Communication
from timeit import default_timer as timer

if __name__ == '__main__':
    print("Starting the y2 worker.")

    # Configure the loggers
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

    # Start the communications
    communication_crashed = threading.Event()
    communication = Communication(communication_crashed)

    messages = communication.receive()
    while messages.empty():
        time.sleep(2)
        messages = communication.receive()

    # TODO Process message of type NodeIndex specifically
    message_type, message = messages.get()
    node_id = int.from_bytes(message[:4], byteorder='big')
    total_nodes = int.from_bytes(message[-4:], byteorder='big')
    print(node_id, total_nodes)

    rnn = model.RecurrentModel(13, 500, 5, 30, communication)

    # initializer data loaders
    val_loader = data.val_loader(node_id, total_nodes)
    train_loader = data.train_loader(node_id, total_nodes)

    # initialize trainer
    n_epochs = 2
    start = timer()
    trainer = model.ModelTrainer(rnn, val_loader, train_loader, n_epochs)

    # run the training
    trainer.train()
    end = timer()
    print(end - start)
    trainer.save()

    try:
        while not communication_crashed.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        communication.stop()
