import random
import math

class Neuron:
    def __init__(self, input_size):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(input_size)]
        self.bias = random.uniform(-0.5, 0.5)

    def forward(self, inputs):
        self.sum_value = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.sum_value

    def activate_leaky_relu(self, x):
        """
        Applies the Leaky ReLU activation function to the input.

        The Leaky ReLU function is defined as:
        - f(x) = x if x > 0
        - f(x) = 0.2 * x if x <= 0

        Parameters:
        x (float): The input value to be activated.

        Returns:
        float: The activated value.
        """
        print('activate_leaky_relu', x if x > 0 else 0.2 * x)
        return x if x > 0 else 0.2 * x

    def activate_sigmoid(self, x):
        print('activate sigmoid', 1.0 / (1.0 + math.exp(-x)))
        return 1.0 / (1.0 + math.exp(-x))

    def update(self, inputs, gradient, learning_rate):
        for i in range(len(self.weights)):
            
            self.weights[i] += learning_rate * gradient * inputs[i]
            print( 'update-weights = ' ,self.weights[i], learning_rate, 'gradient:' , gradient,'inputs[i]:', inputs[i])
        self.bias += learning_rate * gradient

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, learning_rate=0.01):
        self.hidden_layers = [[Neuron(input_size if i == 0 else hidden_sizes[i - 1]) for _ in range(size)] for i, size in enumerate(hidden_sizes)]
        self.output_neuron = Neuron(hidden_sizes[-1])
        self.learning_rate = learning_rate

    def forward(self, inputs):
        self.hidden_outputs = []
        print('hidden outputs', self.hidden_outputs, 'inputs', inputs)
        current_inputs = inputs

        for layer in self.hidden_layers:
            layer_output = [neuron.activate_leaky_relu(neuron.forward(current_inputs)) for neuron in layer]
            self.hidden_outputs.append(layer_output)
            current_inputs = layer_output

        output = self.output_neuron.activate_sigmoid(self.output_neuron.forward(current_inputs))
        return output

    def train(self, inputs, target):
        output = self.forward(inputs)
        error = target - output

        output_gradient = error * output * (1 - output)
        self.output_neuron.update(self.hidden_outputs[-1], output_gradient, self.learning_rate)

        for i in reversed(range(len(self.hidden_layers))):
            print('i', i)
            layer = self.hidden_layers[i]
            next_gradient = [0.0] * len(layer)

            for j, neuron in enumerate(layer):
                hidden_gradient = output_gradient * self.output_neuron.weights[i] * (1 if neuron.sum_value > 0 else 0.01)
                neuron.update(inputs if i == 0 else self.hidden_outputs[i - 1], hidden_gradient, self.learning_rate)
                next_gradient[j] += hidden_gradient

#generiere trainings daten
def generate_training_data():
    training_data = []
    for _ in range(110):
        inputs = [random.choice([0, 1]) for _ in range(10)]
        target = 1.0 if any(inputs[i:i+4] == [1, 1, 1, 1] for i in range(7)) else 0.0
        training_data.append((inputs, target))
        print(training_data)
    return training_data

# Netzwerk erstellen und trainieren
nn = NeuralNetwork(input_size=10, hidden_sizes=[12, 6], learning_rate=0.01)
training_data = generate_training_data()

for epoch in range(500):
    for inputs, target in training_data:
        nn.train(inputs, target)

# Testen des Netzwerks
test_data = [
    ([0, 0, 0, 0, 1, 1, 1, 1, 0, 0], 1.0),
    ([1, 1, 0, 0, 0, 0, 0, 1, 1, 0], 0.0),
    ([1, 1, 1, 1, 0, 0, 0, 1, 1, 0], 1.0),
    ([1, 1, 0, 0, 0, 0, 0, 1, 1, 0], 0.0),
]

for inputs, expected in test_data:
    output = nn.forward(inputs)
    print(f'Eingaben: {inputs}, Erwartet: {expected}, Ausgabe: {output:.2f}')