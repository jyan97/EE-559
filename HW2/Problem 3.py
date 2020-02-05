from plotDecBoundaries import plotDecBoundaries
import numpy as np

data = [[-4,-4],[4,4]]
mean_1 = [[0, -2], [0, 1]]
label_1 = [5,6]
plotDecBoundaries(np.array(data), np.array(label_1), np.array(mean_1), 1)
mean_2 = [[0, -2], [0, 1], [2, 0]]
label_2 = [5,6,7]
plotDecBoundaries(np.array(data), np.array(label_2), np.array(mean_2), 1)

