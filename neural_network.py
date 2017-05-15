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

# Convert output layer neurons value to single value output shown in dataset
def network_to_dataset(layer):
	return value

# Convert single value output shown in dataset to output layer neurons value
def dataset_to_network(value):
	return layer


class NeuralNetwork(object):

	def __init__(self, n_input, n_hidden, n_output, n_epoch, epoch_size, l_rate):

		# Number of units per layer
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_output = n_output

		# Initialize weights with random values from the standard normal(Gaussian) distribution
		self.hidden_weights = np.random.randn(n_hidden, n_input + 1)   
		self.output_weights = np.random.randn(n_output, n_hidden + 1)  

		# Activation values for neurons generated in the forward propagation step
		self.input_activation = np.zeros(n_input)
		self.hidden_activation = np.zeros(n_hidden)
		self.output_activation = np.zeros(n_output)

		# Delta values initialized as array of 0's
		self.hidden_delta = np.zeros((n_hidden, n_input))
		self.output_delta = np.zeros((n_output, n_hidden))

		# Learning parameters
		self.n_epoch = n_epoch
		self.epoch_size = epoch_size
		self.l_rate = l_rate

	# Forward propagation in the network for the given input. The activation function used is the sigmoid function.
	def forward_propagate(self, input):

		# Input layer(The activation function is not used here)
		self.input_activation = input

		# Hidden layer
		for neuron in range(self.n_hidden):
			activation = self.hidden_weights[neuron][-1]  # Input bias weight
			for synapse in range(self.n_input):
				activation += self.input_activation[synapse] * self.hidden_weights[neuron][synapse]
			self.hidden_activation[neuron] = sigmoid(activation)

		# Output layer
		for neuron in range(self.n_output):
			activation = self.output_weights[neuron][-1]  # Hidden bias weight
			for synapse in range(self.n_hidden):
				activation += self.hidden_activation[synapse] * self.output_weights[neuron][synapse]
			self.output_activation[neuron] = sigmoid(activation)

		return self.output_activation

	# Back propagation 
	def back_propagate():
		back_propagate_error()
		update_weights()

	# We use the same training function to run every gradient descent algorithm requested:
	# - Standard Gradient Descent: epoch size = number of input instances.
	# - Stochastic Gradient Descent: epoch size = 1.
	# - Mini-batch Gradient Descent: 1 < epoch size < number of input instances.
	def gradient_descent(network, epoch_size, ):
		pass

	def predict(self, input):
		pass

	def evaluate_precision(self):
		pass