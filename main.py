""" 
3-layer Neural Networks with Backpropagation

Developed by Joao Francisco B. S. Martins <joaofbsm@dcc.ufmg.br>
"""

# TODO
# - DELTA tem que ter tamanho dos weights, incluindo bias

import numpy as np
import neural_network as nn

def load_data(dataset):
	data = np.genfromtxt(dataset, delimiter = ",")

	y = data[:, 0]   # Output values are in the first column
	x = data[:, 1:]  # Input values are the rest of the data
	
	# Feature scaling. All values are now between 0 and 1.
	x -= x.min()
	x /= x.max()

	return x, y
	
def save_data(error):
	np.savetxt("error.csv", error, delimiter = ",", newline="\n")

data_file = "data_tp1"
out_file = sys.argv[1]
x, y = load_data(data_file)

n_input = x.shape[1] # Number of input units(input features)
n_hidden = sys.argv[2]
n_output = 10 # Number of output units(output classes)
n_epoch = 100 # Number of epochs with 5000 instances each
batch_size = sys.argv[3]
l_rate = sys.argv[4]

network = nn.NeuralNetwork(n_input, n_hidden, n_output)
#print network.forward_propagate(x[0])
network.train(x, y, n_epoch, batch_size, l_rate)

"""
hidden = (25, 50, 100)
rate = (0.5, 1, 10)
batch_size = (1, 10, 50, 5000)
for n_hidden in hidden:
	for l_rate in rate:
		for in in range(len(epoch)):
			network = nn.NeuralNetwork()

"""


