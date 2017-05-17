import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('TkAgg')  
plt.style.use('ggplot')

mlResults = pd.read_csv("mlResults.csv")


learningRateToPlot = 10
nodesInHidden = [25, 50, 100]
batchSize = {1:"SGD",10:"Mini-Batch 10",50: "Mini-Batch 50",5000: "Batch"}


fig, axes = plt.subplots(nrows=1, ncols=3)
for i, nodes in enumerate(nodesInHidden):
    tempDataFrame = pd.DataFrame()
    for bSize, name in batchSize.iteritems():
        # print "{}, {}, {}, {}".format(learningRateToPlot, nodes, bSize, name)
        identifier = "hdn_{}_btsz_{}_lrn_{}".format(nodes, bSize, learningRateToPlot)
        tempDataFrame[name] = mlResults[identifier]

    axes[i].set_xlabel('Epochs')
    axes[i].set_ylabel('Cost (J)')
    tempDataFrame.plot(ax=axes[i], title = "With {} nodes in hidden layer.".format(nodes))
plt.suptitle("Learning rate: {}".format(learningRateToPlot))
# plt.figsize=((9, 3))
plt.show()