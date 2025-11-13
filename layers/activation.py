import math

class Activation:
    @staticmethod
    def threshold(x):
        return 1 if x >= 0 else 0

    @staticmethod
    def d_threshold(x):
        return 0

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def d_relu(x):
        return 1 if x > 0 else 0

    @staticmethod
    def sigmoid(x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    @staticmethod
    def d_sigmoid(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)
