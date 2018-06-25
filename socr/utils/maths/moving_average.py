class MovingAverage:
    def __init__(self, moving_average_size=256):
        self.count = 0
        self.i = 0
        self.moving_average_size = moving_average_size
        self.values = self.moving_average_size * [None]

    def reset(self):
        self.count = 0
        self.i = 0

    def make_average(self, min, max):
        if min == max:
            return self.values[min]
        else:
            diff = (max - min) / 2
            return (self.make_average(min, max - int(diff + 0.5)) + self.make_average(min + int(diff + 0.5), max)) / 2

    def moving_average(self):
        return self.make_average(0, self.count - 1)

    def addn(self, value):
        self.values[self.i] = value
        self.count = min(self.count + 1, self.moving_average_size)
        self.i = (self.i + 1) % self.moving_average_size
