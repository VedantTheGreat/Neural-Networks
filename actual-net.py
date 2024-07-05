import numpy as np

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid_function(x) * (1 - sigmoid_function(x))

def mean_squared_error(y_actual, y_predicted):
    return ((y_actual - y_predicted) ** 2).mean()

class BaseNetwork:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward_pass(self, inputs):
        total = np.dot(self.weights, inputs) + self.biases
        return sigmoid_function(total)

class NeuralNetwork(BaseNetwork):
    def __init__(self):
        self.weight1 = np.random.normal()
        self.weight2 = np.random.normal()
        self.weight3 = np.random.normal()
        self.weight4 = np.random.normal()
        self.weight5 = np.random.normal()
        self.weight6 = np.random.normal()

        self.bias1 = np.random.normal()
        self.bias2 = np.random.normal()
        self.bias3 = np.random.normal()

    def forward_pass(self, inputs):
        neuron1 = sigmoid_function(self.weight1 * inputs[0] + self.weight2 * inputs[1] + self.bias1)
        neuron2 = sigmoid_function(self.weight3 * inputs[0] + self.weight4 * inputs[1] + self.bias2)
        output = sigmoid_function(self.weight5 * neuron1 + self.weight6 * neuron2 + self.bias3)
        return output

    def train_network(self, training_data, target_data):
        learning_rate = 0.02
        epochs = 10

        for epoch in range(epochs):
            for data_point, target in zip(training_data, target_data):
                sum_n1 = self.weight1 * data_point[0] + self.weight2 * data_point[1] + self.bias1
                neuron1 = sigmoid_function(sum_n1)

                sum_n2 = self.weight3 * data_point[0] + self.weight4 * data_point[1] + self.bias2
                neuron2 = sigmoid_function(sum_n2)

                sum_output = self.weight5 * neuron1 + self.weight6 * neuron2 + self.bias3
                predicted_output = sigmoid_function(sum_output)

                error = -2 * (target - predicted_output)

                grad_predicted_output = derivative_sigmoid(sum_output) * self.weight5
                grad_neuron1 = derivative_sigmoid(sum_n1) * data_point[0]
                grad_neuron2 = derivative_sigmoid(sum_n1) * data_point[1]
                grad_pred_neuron2 = derivative_sigmoid(sum_output) * self.weight6
                grad_neuron3 = derivative_sigmoid(sum_n2) * data_point[0]
                grad_neuron4 = derivative_sigmoid(sum_n2) * data_point[1]
                grad_pred_w5 = derivative_sigmoid(sum_output) * neuron1
                grad_pred_w6 = derivative_sigmoid(sum_output) * neuron2
                grad_bias1 = derivative_sigmoid(sum_n1)
                grad_bias2 = derivative_sigmoid(sum_n2)
                grad_pred_bias3 = derivative_sigmoid(sum_output)

                self.weight1 -= learning_rate * error * grad_predicted_output * grad_neuron1
                self.weight2 -= learning_rate * error * grad_predicted_output * grad_neuron2
                self.weight3 -= learning_rate * error * grad_pred_neuron2 * grad_neuron3
                self.weight4 -= learning_rate * error * grad_pred_neuron2 * grad_neuron4
                self.weight5 -= learning_rate * error * grad_pred_w5
                self.weight6 -= learning_rate * error * grad_pred_w6
                self.bias1 -= learning_rate * error * grad_predicted_output * grad_bias1
                self.bias2 -= learning_rate * error * grad_pred_neuron2 * grad_bias2
                self.bias3 -= learning_rate * error * grad_pred_bias3
                predictions = np.apply_along_axis(self.forward_pass, 1, training_data)
                loss = mean_squared_error(target_data, predictions)
                print(f"Epoch {epoch} loss: {loss:.3f}")

training_data = np.array([
    [-2, -1],  # Example 1
    [25, 6],   # Example 2
    [17, 4],   # Example 3
    [-15, -6], # Example 4
])

target_values = np.array([
    1,  # Example 1
    0,  # Example 2
    0,  # Example 3
    1,  # Example 4
])

neural_net = NeuralNetwork()
neural_net.train_network(training_data, target_values)