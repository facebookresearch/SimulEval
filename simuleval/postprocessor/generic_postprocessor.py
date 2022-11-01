from collections import deque


class GenericPostProcessor:
    def __init__(self):
        self.reset()
        pass

    def push(self, item):
        self.deque.append(item)

    def pop(self):
        raise NotImplementedError

    def reset(self):
        self.deque = deque()
