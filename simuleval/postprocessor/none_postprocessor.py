from . generic_postprocessor import GenericPostProcessor

class NonePostProcessor(GenericPostProcessor):
    def pop(self):
        return self.deque.popleft()