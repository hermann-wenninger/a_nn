import random
import math
class Neuron:
    def __init__(self, input_size):
        # Initialisiere Gewichte und Bias mit kleinen Zufallswerten
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(input_size)]
        self.bias = random.uniform(-0.5, 0.5)

    def forward(self, inputs):
        # Vorwärtspropagation: gewichtete Summe berechnen
        self.sum_value = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.sum_value

    def activate_relu(self, x):
        # ReLU-Aktivierungsfunktion
        return max(0.0, x)

    def activate_sigmoid(self, x):
        # Sigmoid-Aktivierungsfunktion
        return 1.0 / (1.0 + math.exp(-x))

    def update(self, inputs, gradient, learning_rate):
        # Aktualisiere Gewichte und Bias
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * gradient * inputs[i]
        self.bias += learning_rate * gradient

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, learning_rate):
        # Initialisiere versteckte und Ausgabeschicht
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_neuron = Neuron(hidden_size)
        self.learning_rate = learning_rate

    def forward(self, inputs):
        # Vorwärtspropagation durch die versteckte Schicht
        self.hidden_outputs = [neuron.activate_relu(neuron.forward(inputs)) for neuron in self.hidden_layer]
        # Vorwärtspropagation durch die Ausgabeschicht
        output = self.output_neuron.activate_sigmoid(self.output_neuron.forward(self.hidden_outputs))
        return output

    def train(self, inputs, target):
        # Berechne die Vorwärtspropagation
        output = self.forward(inputs)
        # Berechne den Fehler
        error = target - output

        # Rückwärtspropagation für die Ausgabeschicht
        output_gradient = error * output * (1 - output)
        self.output_neuron.update(self.hidden_outputs, output_gradient, self.learning_rate)

        # Rückwärtspropagation für die versteckte Schicht
        for i, hidden_neuron in enumerate(self.hidden_layer):
            hidden_gradient = output_gradient * self.output_neuron.weights[i] * (1 if hidden_neuron.sum_value > 0 else 0)
            hidden_neuron.update(inputs, hidden_gradient, self.learning_rate)

# Beispiel: Trainingsdatensatz erstellen
def generate_training_data():
    training_data = []
    for _ in range(100):
        inputs = [random.choice([0, 1]) for _ in range(10)]
        target = 1.0 if any(inputs[i:i+4] == [1, 1, 1, 1] for i in range(7)) else 0.0
        training_data.append((inputs, target))
        print(training_data)
    return training_data

# Netzwerk erstellen und trainieren
nn = NeuralNetwork(input_size=10, hidden_size=4, learning_rate=0.01)
training_data = generate_training_data()

# Training des Netzwerks
for epoch in range(10000):  # 1000 Epochen
    for inputs, target in training_data:
        nn.train(inputs, target)

# Testen des Netzwerks
test_data = [
    ([0, 0, 0, 0, 1, 1, 1, 1, 0, 0], 1.0),
    ([1, 1, 0, 0, 0, 0, 0, 1, 1, 0], 0.0),
    ([0, 1, 1, 1, 1, 0, 0, 0, 0, 0], 1.0),
    ([1, 0, 0, 0, 0, 1, 1, 0, 0, 0], 0.0),
]

# Ergebnisse ausgeben
for inputs, expected in test_data:
    output = nn.forward(inputs)
    print(f'Eingaben: {inputs}, Erwartet: {expected}, Ausgabe: {output:.2f}')