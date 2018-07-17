from abc import ABCMeta, abstractmethod


class AbstractTransformation:
    __metaclass__ = ABCMeta

    @abstractmethod
    def transform_image(self, image): pass

    @abstractmethod
    def transform_position(self, x, y, width, height): pass

    @abstractmethod
    def generate_random(self): pass
