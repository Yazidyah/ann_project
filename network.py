from layer import Layer

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, activation_func):
        layer = Layer(input_size, output_size, activation_func)
        self.layers.append(layer)

    def forward(self, inputs):
        x = list(inputs)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, target, learning_rate):
        # compute dE/dy for output layer: for MSE (0.5*(y-t)^2) -> dE/dy = (y - t)
        last = self.layers[-1]
        # ensure length match
        m = min(len(target), last.output_size)
        output_error = []
        for j in range(m):
            output_error.append(last.output[j] - target[j])
        # if target shorter than output, fill remaining with 0
        if last.output_size > m:
            output_error += [0.0] * (last.output_size - m)

        # backpropagate through layers
        err = output_error
        for layer in reversed(self.layers):
            err = layer.backward(err, learning_rate)

    def train(self, data_in, data_out, epochs=1000, learning_rate=0.1):
        errors = []
        for epoch in range(epochs):
            total_error = 0.0
            for x, y in zip(data_in, data_out):
                out = self.forward(x)
                # compute and store error (MSE sum of squared errors)
                total_error += sum((out[i] - y[i]) ** 2 for i in range(min(len(out), len(y))))
                # backprop
                self.backward(y, learning_rate)
            errors.append(total_error)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: total error = {total_error:.6f}")
        return errors
