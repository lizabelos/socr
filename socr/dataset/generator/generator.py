from abc import ABCMeta, abstractmethod


class Generator:
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate(self, index): pass
