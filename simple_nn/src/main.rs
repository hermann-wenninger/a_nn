
// TENSOR
#[derive(Debug, Clone)]
struct Tensor {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Tensor {
    // Konstruktor für einen neuen Tensor
    fn new(rows: usize, cols: usize, init_value: f64) -> Self {
        Tensor {
            data: vec![init_value; rows * cols],
            rows,
            cols,
        }
    }

    // Funktion zum Setzen eines Wertes
    fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    // Funktion zum Abrufen eines Wertes
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }
}

// PERZEPTRON
#[derive(Debug)]
struct Perceptron {
    weights: Tensor,
    bias: f64,
    learning_rate: f64,
}

impl Perceptron {
    // Konstruktor für ein neues Perzeptron
    fn new(input_size: usize, learning_rate: f64) -> Self {
        // Initialisierung der Gewichte mit kleinen Zufallswerten
        let mut weights = Tensor::new(1, input_size, 0.0);
        for i in 0..input_size {
            weights.set(0, i, rand::random::<f64>() - 0.5);
        }

        Perceptron {
            weights,
            bias: rand::random::<f64>() - 0.5,
            learning_rate,
        }
    }

    // Aktivierungsfunktion (z.B. sigmoid)
    fn activation(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    // Vorwärtspropagation
    fn forward(&self, inputs: &[f64]) -> f64 {
        let mut sum = self.bias;
        for i in 0..inputs.len() {
            sum += self.weights.get(0, i) * inputs[i];
        }
        Self::activation(sum)
    }

    // Backpropagation
    fn train(&mut self, inputs: &[f64], target: f64) {
        let output = self.forward(inputs);
        let error = target - output;
        
        // Update der Gewichte und des Bias
        for i in 0..inputs.len() {
            let new_weight = self.weights.get(0, i) + self.learning_rate * error * inputs[i];
            self.weights.set(0, i, new_weight);
        }
        self.bias += self.learning_rate * error;
    }
}

//MAIN FUNCTION
fn main() {
    let mut perceptron = Perceptron::new(2, 0.1);

    // Trainingsdaten für ein logisches UND
    let training_data = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 0.0),
        (vec![1.0, 0.0], 0.0),
        (vec![1.0, 1.0], 1.0),
    ];

    // Training des Perzeptrons
    for _ in 0..1000 {
        for (inputs, target) in &training_data {
            perceptron.train(inputs, *target);
        }
    }

    // Testen des Perzeptrons
    for (inputs, _) in &training_data {
        let output = perceptron.forward(inputs);
        println!("Eingaben: {:?}, Ausgabe: {}", inputs, output);
    }
}
