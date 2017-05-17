""" 
3-layer Neural Networks with Backpropagation

Developed by Joao Francisco B. S. Martins <joaofbsm@dcc.ufmg.br>

TO EXECUTE
 python main.py <OUTPUT_FILE> <HIDDEN_UNITS> <BATCH_SIZE> <LEARNING_RATE>

 In order to choose between different gradient descent algorithms, choose BATCH_SIZE such that:
	 - Standard Gradient Descent: batch size = number of input instances.
	 - Stochastic Gradient Descent: batch size = 1.
	 - Mini-batch Gradient Descent: 1 < batch size < number of input instances.

 To execute the script directly:
 chmod +x ./main
 ./main <ARGS>
"""

import sys
import numpy as np
import neural_network as nn

def load_data(dataset):
	data = np.genfromtxt(dataset, delimiter = ",")

	y = data[:, 0]   # Output values are in the first column
	x = data[:, 1:]  # Input values are the rest of the data

	aux = np.zeros((y.shape[0], 10))
	for i in range(y.shape[0]):
		aux[i][y[i]] = 1
	y = aux
	
	# Feature scaling. All values are now between 0 and 1.
	x -= x.min()
	x /= x.max()

	x = np.c_[x, np.ones(x.shape[0])]  # Adds column of ones for bias as last column

	return x, y
	
def save_data(error):
	np.savetxt("error.csv", error, delimiter = ",", newline="\n")

data_file = "data_tp1"
output_file = sys.argv[1]
x, y = load_data(data_file)

f = open(output_file, 'w')

n_input = 784 # Number of input units(input features) excluding bias
n_hidden = int(sys.argv[2]) # Number of hidden units excluding bias
n_output = 10 # Number of output units(output classes)

n_epoch = 100 # Number of epochs with 5000 instances each
batch_size = int(sys.argv[3])
l_rate = float(sys.argv[4])

network = nn.NeuralNetwork(n_input, n_hidden, n_output)
network.train(x, y, n_epoch, batch_size, l_rate, output_file)

"""
hidden = (25, 50, 100)
batch_size = (1, 10, 50, 5000)
rate = (0.5, 1, 10)
for n_hidden in hidden:
	for b_size in batch_size:
		for l_rate in rate:
			network = nn.NeuralNetwork(n_input, n_hidden, n_output)
			network.train(x, y, n_epoch, b_size, l_rate)
"""


