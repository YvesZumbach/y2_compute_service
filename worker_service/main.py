
import logging
import time
import threading

from communication import Communication

if __name__ == '__main__':
    # Configure the loggers
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

    communication_crashed = threading.Event()
    communication = Communication(communication_crashed)

    communication.send(b"test1")
    communication.send(b"test2")
    communication.send(b"test3\n")

    messages = communication.receive()
    while not messages.empty():
        print(messages.get_nowait())

    try:
        while not communication_crashed.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        communication.stop()

