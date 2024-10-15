import numpy as np

# Step 1: Data Preparation
def generate_data(num_samples=110):
    X = np.random.randint(0, 2, (num_samples, 10))  # Binary data (0 or 1)
    y = np.array([1 if '1111' in ''.join(map(str, x)) else 0 for x in X])  # 1 if four 1s in a row
    print([X  , y])
    return X, y

# Generate training and test datasets
X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(200)

# Step 2: Model Definition
class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size):
        # Initialize weights and biases
        self.weights_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_output = np.random.randn(hidden_size, 1) * 0.01
        self.bias_output = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def forward(self, x):
        # Forward pass through the hidden layer
        self.hidden_input = np.dot(x, self.weights_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Forward pass through the output layer
        self.output_input = np.dot(self.hidden_output, self.weights_output) + self.bias_output
        self.output = self.leaky_relu(self.output_input)

        return self.output

    def backward(self, x, y, learning_rate=0.001):
        # Compute the loss gradient at the output layer
        output_error = self.output - y.reshape(-1, 1)  # Error at the output
        output_delta = output_error * self.leaky_relu_derivative(self.output_input)

        # Compute the loss gradient at the hidden layer
        hidden_error = output_delta.dot(self.weights_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_output -= learning_rate * self.hidden_output.T.dot(output_delta)
        self.bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weights_hidden -= learning_rate * x.T.dot(hidden_delta)
        self.bias_hidden -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if (epoch + 1) % 10 == 0:
                loss = np.mean((self.output - y.reshape(-1, 1)) ** 2)
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    def predict(self, X):
        predictions = self.forward(X)
        print(predictions)
        return (predictions > 0.5).astype(int)

# Initialize the model
model = SimpleNeuralNet(input_size=10, hidden_size=16)

# Train the model
model.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Step 4: Evaluate the model on test data
predictions = model.predict(X_test)
accuracy = np.mean(predictions.flatten() == y_test) * 100
accuracy
