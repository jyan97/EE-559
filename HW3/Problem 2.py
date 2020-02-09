from plotDecBoundaries import plotDecBoundaries
import numpy as np

train_data = [[1, -3], [1, -5], [1, 1], [1, -1]]
train_label = [1, 1, 2, 2]

mean = [1, 0.25 * sum(x[1] for x in train_data)]

plotDecBoundaries(np.array(train_data), np.array(train_label), mean)

reflected_data = [[-x for x in train_data[i]] if train_label[i] != 1 else train_data[i] for i in range(len(train_data))]

plotDecBoundaries(np.array(reflected_data), np.array(train_label), mean)
