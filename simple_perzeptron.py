import random
import math

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialisierung der Gewichte und des Bias
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(input_size)]
        self.bias = random.uniform(-0.5, 0.5)
        self.learning_rate = learning_rate

    def activation(self, x):
        # Sigmoid-Aktivierungsfunktion
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        # Vorwärtspropagation: gewichtete Summe berechnen und Aktivierung anwenden
        sum_value = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.activation(sum_value)

    def train(self, inputs, target):
        # Ausgabe berechnen
        output = self.forward(inputs)
        # Fehler berechnen
        error = target - output
        # Gewichte und Bias aktualisieren
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i]
        self.bias += self.learning_rate * error

# Beispiel: Trainingsdatensatz erstellen
def generate_training_data():
    # Trainingsdaten für einen Eingabevektor mit 10 Elementen
    # Nur wenn 4 fortlaufende positive Werte auftreten, ist die Zielausgabe 1
    training_data = []
    for _ in range(100):
        inputs = [random.choice([0, 1]) for _ in range(10)]
        target = 1.0 if any(inputs[i:i+4] == [1, 1, 1, 1] for i in range(7)) else 0.0
        training_data.append((inputs, target))
        print('train data', training_data)
    return training_data

# Perzeptron erstellen und trainieren
perceptron = Perceptron(input_size=10, learning_rate=0.1)
training_data = generate_training_data()

# Training des Perzeptrons
for epoch in range(50):  # 1000 Epochen
    for inputs, target in training_data:
        perceptron.train(inputs, target)

# Testen des Perzeptrons
test_data = [
    ([0, 0, 0, 0, 1, 1, 1, 1, 0, 0], 1.0),
    ([1, 1, 0, 0, 0, 0, 0, 1, 1, 0], 0.0),
    ([0, 1, 1, 1, 1, 0, 0, 0, 0, 0], 1.0),
    ([1, 0, 0, 0, 0, 1, 1, 0, 0, 0], 0.0),
]

# Ergebnisse ausgeben
for inputs, expected in test_data:
    output = perceptron.forward(inputs)
    print(f'Eingaben: {inputs}, Erwartet: {expected}, Ausgabe: {output:.2f}')