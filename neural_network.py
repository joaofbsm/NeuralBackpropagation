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
def dataset_to_network(value, n_units):
	output_layer = np.zeros(n_units)
	output_layer[value] = 1
	return output_layer

# Cross-entropy cost function
def cost_function(output, expected):
	cost = sum([((-expected[k] * np.log(output[k])) - (1 - expected[k]) * np.log(1 - output[k])) for k in range(len(expected))])
	return cost

class NeuralNetwork(object):

	def __init__(self, n_input, n_hidden, n_output):

		# Number of units per layer
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_output = n_output

		# Initialize weights with random floats from the half-open interval [0.0, 1.0) * 2 - 1
		self.hidden_weights = 2 * np.random.random((n_hidden, n_input + 1)) - 1
		self.output_weights = 2 * np.random.random((n_output, n_hidden + 1)) - 1

		# Activation values for neurons generated in the forward propagation step
		self.input_activation = np.zeros(n_input)
		self.hidden_activation = np.zeros(n_hidden)
		self.output_activation = np.zeros(n_output)

		# Delta values initialized as array of 0's
		self.hidden_delta = np.zeros(n_hidden)
		self.output_delta = np.zeros(n_output)

		# DELTA is the accumulator for the (mini)batch gradient descent
		self.hidden_DELTA = np.zeros((n_hidden, n_input))
		self.output_DELTA = np.zeros((n_output, n_hidden))

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

	def calculate_deltas(self, expected):

		# Output layer
		for k in range(self.n_output):
			self.output_delta[k] = (self.output_activation[k] - expected[k]) * sigmoid_derivative(self.output_activation[k]) 

#		# Update acumulator for weights connected to the output layer
#		for j in range(self.n_hidden):
#			for k in range(self.n_output):
#				self.output_DELTA[k][j] += self.output_delta[k] * self.hidden_activation[j]

		# Hidden layer
		for j in range(self.n_hidden):
			error = 0.0
			for k in range(self.n_output):
				error += self.output_delta[k] * self.output_weights[k][j]
			self.hidden_delta[j] = error * sigmoid_derivative(self.hidden_activation[j]) 
		
#		# Update acumulator for weights connected to the hidden layer from the input
#		for i in range(self.n_input):
#			for j in range(self.n_hidden):
#				self.hidden_DELTA[j][i] += self.hidden_delta[j] * self.input_activation[i]

	def update_weights(self, l_rate):
		# Weights of the connections that go from the hidden layer to the output layer(output_weights)
		for k in range(self.n_output):
			self.output_weights[k][-1] = self.output_weights[k][-1] + (l_rate * self.output_delta[k])  # Update hidden bias weight
			for j in range(self.n_hidden):
		#		self.output_weights[k][j] = self.output_weights[k] - (l_rate * self.output_DELTA[k][j])
				self.output_weights[k][j] = self.output_weights[k][j] + (l_rate * self.output_delta[k] * self.hidden_activation[j])

		# Weights of the connections that go from the input layer to the hidden layer(hidden_weights)
		for j in range(self.n_hidden):
			self.hidden_weights[j][-1] = self.hidden_weights[j][-1] + (l_rate * self.hidden_delta[j])  # Update input bias weight
			for i in range(self.n_input):
		#		self.hidden_weights[j] = self.hidden_weights[j] - (l_rate * self.hidden_DELTA[j][i])
				self.hidden_weights[j][i] = self.hidden_weights[j][i] + (l_rate * self.hidden_delta[j] * self.input_activation[i])

	# Back propagate the errors so we can update the weights. Expected should be converted by using dataset to network.
	def back_propagate(self, expected):

		self.calculate_deltas(expected)
		self.update_weights()

	# We use the same training function to run every gradient descent algorithm requested:
	# - Standard Gradient Descent: epoch size = number of input instances.
	# - Stochastic Gradient Descent: epoch size = 1.
	# - Mini-batch Gradient Descent: 1 < epoch size < number of input instances.
	def train(self, x, y, n_epoch, batch_size, l_rate):
		n_instance = x.shape[0]
		n_batch = n_instance / batch_size
		for epoch in range(n_epoch):
			instance = 0
			for batch in range(n_batch):
				loss = 0.0
		#		self.hidden_DELTA = np.zeros((self.n_hidden, self.n_input))
		#		self.output_DELTA = np.zeros((self.n_output, self.n_hidden))
				for i in range(batch_size):
					output =  self.forward_propagate(x[instance])
					expected = dataset_to_network(y[instance], self.n_output)
					loss += cost_function(output, expected)
					self.calculate_deltas(expected)
					instance += 1
				loss /= batch_size
		#		self.output_DELTA /= batch_size
		#		self.hidden_DELTA /= batch_size
				self.update_weights(l_rate)

				print "> epoch=", epoch, "batch=", batch, "loss=", loss
		#		print "output", output, "expected", expected
				print self.output_weights

	def predict(self, input):
		pass

	def evaluate_precision(self):
		pass