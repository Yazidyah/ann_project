import random
import activation

class Layer:
    def __init__(self, input_size, output_size, activation_func):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation_func.lower()

        # initialize weights and biases
        self.weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [random.uniform(-1, 1) for _ in range(output_size)]

        # set function pointers
        if self.activation_name == "sigmoid":
            self.activation = activation.sigmoid
            self.activation_derivative = activation.sigmoid_derivative
        elif self.activation_name == "relu":
            self.activation = activation.relu
            self.activation_derivative = activation.relu_derivative
        elif self.activation_name == "threshold":
            self.activation = activation.threshold
            self.activation_derivative = activation.threshold_derivative
        else:
            raise ValueError("Fungsi aktivasi tidak dikenal! Gunakan sigmoid / relu / threshold")

        # caches
        self.last_input = None
        self.last_z = None
        self.output = None

    def forward(self, inputs):
        self.last_input = list(inputs)
        self.last_z = []
        self.output = []
        for j in range(self.output_size):
            z = 0.0
            for i in range(self.input_size):
                z += self.weights[j][i] * inputs[i]
            z += self.biases[j]
            self.last_z.append(z)
            self.output.append(self.activation(z))
        return list(self.output)

    def backward(self, output_error, learning_rate):
        """
        output_error should be dE/dy for this layer's outputs (vector length = output_size OR <= output_size)
        Returns dE/dx for previous layer inputs (length = input_size).
        Updates weights and biases using gradient descent:
            w -= lr * delta * input
        where delta = dE/dz = dE/dy * f'(z)
        """
        # handle mismatch safely
        n = min(len(output_error), self.output_size)
        d_input = [0.0 for _ in range(self.input_size)]

        for j in range(n):
            # dE/dz_j
            delta = output_error[j] * self.activation_derivative(self.last_z[j])
            # accumulate dE/dx_i
            for i in range(self.input_size):
                d_input[i] += self.weights[j][i] * delta
            # update weights and bias (gradient descent)
            for i in range(self.input_size):
                self.weights[j][i] -= learning_rate * delta * self.last_input[i]
            self.biases[j] -= learning_rate * delta

        return d_input
