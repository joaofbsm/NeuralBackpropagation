import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('TkAgg')  
plt.style.use('bmh')

learningRateToPlot = 10
nodesInHidden = [25, 50, 100]
batchSize = {1:"SGD",10:"Mini-batch 10",50: "Mini-batch 50",5000: "Batch"}

fig, axes = plt.subplots(nrows=1, ncols=3)
for i, nodes in enumerate(nodesInHidden):
    tempDataFrame = pd.DataFrame()
    for bSize, name in batchSize.iteritems():
        file_name = str(nodes) + "_" + str(bSize) + "_" + str(learningRateToPlot) + ".csv"
        mlResults = pd.read_csv(file_name, header = 0, index_col = 0)
        tempDataFrame[name] = mlResults

    axes[i].set_xlabel('Epochs')
    axes[i].set_ylabel('Cost (J)')
    tempDataFrame.plot(ax=axes[i], title = "With {} nodes in hidden layer.".format(nodes))
plt.suptitle("Learning rate: {}".format(learningRateToPlot))
# plt.figsize=((9, 3))
plt.show()