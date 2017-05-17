"""
3-layer Neural Network Class with Backpropagation

Developed by Joao Francisco B. S. Martins <joaofbsm@dcc.ufmg.br>
"""
import math
import numpy as np

# Sigmoid activation function
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

# By the chain rule we know that the derivative of the sigmoid function(s)
# is: s' = s * (1 - s). x is the value of s already calculated previously.
def sigmoid_derivative(x):
	return x * (1.0 - x)

# Cross-entropy cost function
def cost_function(output, expected):
	return np.dot(-expected, np.log(output)) - np.dot((1 - expected), np.log(1 - output))

class NeuralNetwork(object):

	def __init__(self, n_input, n_hidden, n_output):

		# Number of units per layer
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_output = n_output

		np.random.seed(1)

		# Initialize weights with random floats from the interval [-0.001, 0.001]. Last weight from every row is the bias
		self.hidden_weights = (2 * np.random.random((n_hidden, n_input + 1)) - 1) * 0.001
		self.output_weights = (2 * np.random.random((n_output, n_hidden + 1)) - 1) * 0.001

		# Activation values for neurons generated in the forward propagation step
		self.input_activation = np.zeros(n_input + 1)
		self.hidden_activation = np.zeros(n_hidden + 1)
		self.output_activation = np.zeros(n_output)

		# Set bias unit activation to 1
		self.hidden_activation[-1] = 1
 
		# Delta values initialized as array of 0's
		self.hidden_delta = np.zeros(n_hidden)
		self.output_delta = np.zeros(n_output)

		# DELTA is the accumulator for the (mini)batch gradient descent
		self.hidden_DELTA = np.zeros((n_hidden, n_input + 1))
		self.output_DELTA = np.zeros((n_output, n_hidden + 1))

	# Forward propagation in the network for the given input. The activation function used is the sigmoid function.
	def forward_propagate(self, input):

		# Input layer(The activation function is not used here)
		self.input_activation = input

		# Hidden layer
		self.hidden_activation[:(self.n_hidden)] = sigmoid(np.dot(self.hidden_weights, self.input_activation))

		# Output layer
		self.output_activation = sigmoid(np.dot(self.output_weights, self.hidden_activation))

		return self.output_activation

	# First part in the backpropagation algorithm
	def calculate_deltas(self, expected):

		# Output layer deltas
		self.output_delta = np.multiply(np.subtract(self.output_activation, expected), sigmoid_derivative(self.output_activation))
		self.output_DELTA += np.outer(self.output_delta, self.hidden_activation)

		# Hidden layer deltas
		self.hidden_delta = np.dot(self.output_weights[:,:(self.n_hidden)].transpose(), self.output_delta) * sigmoid_derivative(self.hidden_activation[:(self.n_hidden)])
		self.hidden_DELTA += np.outer(self.hidden_delta, self.input_activation)

	# Second part in the backpropagation algorithm. Only update weights after the batch is over.
	def update_weights(self, l_rate):

		# Weights of the connections that go from the hidden layer to the output layer(output_weights)
		self.output_weights -= l_rate * self.output_DELTA

		# Weights of the connections that go from the input layer to the hidden layer(hidden_weights)
		self.hidden_weights -= l_rate * self.hidden_DELTA

	# We use the same training function to run every gradient descent algorithm requested:
	# - Standard Gradient Descent: batch size = number of input instances.
	# - Stochastic Gradient Descent: batch size = 1.
	# - Mini-batch Gradient Descent: 1 < batch size < number of input instances.
	def train(self, x, y, n_epoch, batch_size, l_rate, output_file):
		n_instance = x.shape[0]
		n_batch = n_instance / batch_size
		for epoch in range(n_epoch):
			instance = 0
			for batch in range(n_batch):
				loss = 0.0
				self.hidden_DELTA = np.zeros((self.n_hidden, self.n_input + 1))
				self.output_DELTA = np.zeros((self.n_output, self.n_hidden + 1))
				for i in range(batch_size):
					output =  self.forward_propagate(x[instance])
					expected = y[instance]
					loss = cost_function(output, expected)
					self.calculate_deltas(expected)
					instance += 1
				self.output_DELTA /= batch_size
				self.hidden_DELTA /= batch_size
				self.update_weights(l_rate)

			with open(output_file, 'a') as f:
				f.write(str(epoch) + ',' + str(loss) + '\n')
				f.close()
			print ">epoch=", epoch, "loss=", loss

	def predict(self, input):
		pass

	def evaluate_precision(self):
		pass