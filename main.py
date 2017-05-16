""" 
3-layer Neural Networks with Backpropagation

Developed by Joao Francisco B. S. Martins <joaofbsm@dcc.ufmg.br>
"""

# TODO
# - Devemos alterar n_input e n_hidden para o mesmo + 1
# - Cuidado com pesos iniciais negativos
# - Mudar ordem sinais no calculo do erro -> expected - output
# - Pode ser + no Update_Weights

# ATE O FORWARD ESTA FUNCIONANDO
# DELTA tem que ter tamanho dos weights, incluindo bias


import numpy as np
import neural_network as nn

def load_data(dataset):
	data = np.genfromtxt(dataset, delimiter = ",")

	y = data[:, 0]   # Output values are in the first column
	x = data[:, 1:]  # Input values are the rest of the data
	
	# Feature scaling. All values are now between 0 and 1.
	x -= x.min()
	x /= x.max()

#	x = np.c_[x, np.ones(x.shape[0])]  # Adds column of ones for bias as last column

	return x, y
	
def save_data(error):
	np.savetxt("error.csv", error, delimiter = ",", newline="\n")

data_file = "data_tp1"
x, y = load_data(data_file)

# Input tem 784 entradas + 1 de Bias
# Hidden tem x + 1

n_input = x.shape[1] # Number of input units(input features)
n_hidden = 25 #REMOVE
n_output = 10         # Number of output units(output classes)
n_epoch = 1
batch_size = 1 #REMOVE
l_rate = 10 #REMOVE

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


