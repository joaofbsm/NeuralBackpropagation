""" 
3-layer Neural Networks with Backpropagation

Developed by Joao Francisco B. S. Martins <joaofbsm@dcc.ufmg.br>
"""

# TODO
# - Checar se tamanho dos pesos esta muito grande
#

import numpy as np
import neural_network as nn

def load_data(dataset):
	data = np.genfromtxt(dataset, delimiter = ",")

	y = data[:, 0]   # Output values are in the first column
	x = data[:, 1:]  # Input values are the rest of the data
	
	# Data normalization
	x -= x.min()
	x /= x.max()

#	x = np.c_[x, np.ones(x.shape[0])]  # Adds column of ones for bias as last column

	return x, y
	
def save_data(error):
	np.savetxt("error.csv", error, delimiter = ",", newline="\n")

data_file = "data_tp1"
x, y = load_data(data_file)

n_input = x.shape[1]  # Number of input units(input features)
n_output = 10         # Number of output units(output classes)

n_hidden = 25 
n_epoch = 500
epoch_size = 1
l_rate = 1

network = nn.NeuralNetwork(n_input, n_hidden, n_output, n_epoch, epoch_size, l_rate)
print network.forward_propagate(x[0])

"""


hidden = (25, 50, 100)
rate = (0.5, 1, 10)
epoch = ((50, 1), (20, 10), (10, 50), (5, 5000))
for n_hidden in hidden:
	for l_rate in rate:
		for in in range(len(epoch)):
			network = nn.NeuralNetwork()


"""


