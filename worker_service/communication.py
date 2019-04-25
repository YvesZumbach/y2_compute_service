import asyncio
import logging
import threading


class Communication:
    """
    Implements communications between the communication service and the worker service.
    """

    def __init__(self, communication_crashed):
        self.communication_crashed = communication_crashed

        self.reader = None
        self.writer = None

        self.loop = asyncio.get_event_loop()
        self.read_queue = asyncio.Queue()
        self.write_queue = asyncio.Queue()

        self.log = logging.getLogger(__name__)
        self.task = None

        self.start()

    def start(self):
        # Start the asynchronous connection to the communication service
        self.task = self.loop.create_task(self.start_async())
        t = threading.Thread(target=lambda: self.loop.run_until_complete(self.task))
        t.start()

    async def start_async(self):
        self.reader = None
        self.writer = None
        self.log.info("Connecting to the communication service.")
        while self.reader is None or self.writer is None:
            try:
                self.reader, self.writer = await asyncio.open_connection('127.0.0.1', 8888)
            except asyncio.CancelledError:
                self.log.info("Connection aborted before completion.")
                return
            except ConnectionRefusedError:
                await asyncio.sleep(1)
        self.log.info("Connected to the communication service.")
        # Start the async read and write processes
        await asyncio.gather(self.read(), self.write())

    def stop(self):
        self.log.info("Gracefully stopping the communications with the communication service.")
        self.loop.call_soon_threadsafe(self.stop_inner)

    def stop_inner(self):
        if self.task is not None:
            self.task.cancel()

    def send(self, msg: bytes):
        self.write_queue.put_nowait(msg)
        self.log.info("Message added to the queue of message to send to the communication service.")

    def receive(self) -> asyncio.Queue:
        out = self.read_queue
        self.read_queue = asyncio.Queue()
        self.log.info("All received messages in the queue were read.")
        return out

    async def read(self):
        while True:
            try:
                # Retrieve the size of the message
                size_data = await self.reader.read(4)
                if len(size_data) is 0:
                    raise ConnectionResetError
                size = int.from_bytes(size_data, byteorder='big')
                # Actually retrieve the data
                data = await self.reader.read(size)
                self.log.info("Received a message from communication service of length " + str(size))
                self.read_queue.put_nowait(data)
            except asyncio.CancelledError:
                return
            except ConnectionResetError:
                self.log.error("Connection with the communication service lost...")
                self.communication_crashed.set()
                self.stop_inner()
                return

    async def write(self):
        while True:
            try:
                data = await self.write_queue.get()
                size = (len(data)).to_bytes(4, byteorder='big')
                self.writer.write(size)
                self.writer.write(data)
                self.log.info("Sent a message to the communication service of length " + str(len(data)))
            except asyncio.CancelledError:
                return
            except ConnectionResetError:
                return
