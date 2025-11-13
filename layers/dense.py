from .activation import Activation
import random

class DenseLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [random.uniform(-1, 1) for _ in range(output_size)]
        self.activation_name = activation
        self.last_input = []
        self.last_z = []

    def activate(self, x):
        return getattr(Activation, self.activation_name)(x)

    def activate_derivative(self, x):
        return getattr(Activation, f"d_{self.activation_name}")(x)

    def forward(self, inputs):
        self.last_input = inputs
        outputs = []
        self.last_z = []
        for w_row, b in zip(self.weights, self.biases):
            z = sum(i * w for i, w in zip(inputs, w_row)) + b
            self.last_z.append(z)
            outputs.append(self.activate(z))
        return outputs

    def backward(self, output_error, learning_rate):
        input_error = [0 for _ in range(len(self.last_input))]
        for j in range(len(self.weights)):
            delta = output_error[j] * self.activate_derivative(self.last_z[j])
            for i in range(len(self.weights[j])):
                input_error[i] += delta * self.weights[j][i]
                # 🔽 Perbaikan: gunakan minus, bukan plus
                self.weights[j][i] -= learning_rate * delta * self.last_input[i]
            # 🔽 Bias juga dikurangi
            self.biases[j] -= learning_rate * delta
        return input_error
