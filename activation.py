import math

def sigmoid(x):
    # numeric stable-ish
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def relu(x):
    return x if x > 0 else 0.0

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def threshold(x):
    return 1 if x >= 0.5 else 0

def threshold_derivative(x):
    # not useful for gradient based training, return 0
    return 0.0
