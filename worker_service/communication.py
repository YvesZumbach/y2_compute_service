
import asyncio
import queue
import typing


class Communication:
    """
    Implements communications between the communication service and the worker service.
    """

    def __init__(self):
        self.reader = None
        self.writer = None

        self.readQueue = queue.Queue(0)
        self.writeQueue = queue.Queue(0)

        asyncio.run(self.start())

    async def start(self):
        self.reader, self.writer = await asyncio.open_connection('127.0.0.1', 8888)
        await asyncio.gather(self.receive(), self.send())

    def write(self, msg: typing.List):
        self.writeQueue.put_nowait(msg)

    def read(self):
        out = self.readQueue
        self.readQueue = queue.Queue(0)
        return out

    async def receive(self):
        while True:
            data = await self.reader.read()
            self.readQueue.put(data)

    async def send(self):
        while True:
            self.writer.write(self.writeQueue.get())
