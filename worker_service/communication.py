
import zmq


class Communication:
    """
    Implements communications between the communication service and the worker service.
    """

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("ipc://localhost:5563")
    subscriber.setsockopt(zmq.SUBSCRIBE, b"B")

    while True:
        # Read envelope with address
        [address, contents] = subscriber.recv_multipart()
        print("[%s] %s" % (address, contents))

    # We never get here but clean up anyhow
    subscriber.close()
