hidden = (25, 50, 100)
batch_size = (1, 10, 50, 5000)
rate = (0.5, 1, 10)
for n_hidden in hidden:
	for b_size in batch_size:
		for l_rate in rate:
			print "python main.py", str(n_hidden) + "_" + str(b_size) + "_" + str(l_rate) + ".csv", n_hidden, b_size, l_rate