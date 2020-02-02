import math
import numpy as np
from matplotlib import pyplot as plt

coeff_mat = [[-1, -1, 5], [-1, 0, 3], [-1, 1, -1]]
coeff_mat = [[-(x[1] / x[0]), -(x[2] / x[0])] for x in coeff_mat]
dot_set = [[4, 1], [1, 5], [0, 0]]
dot_x, dot_y = list(map(list, zip(*dot_set)))

plt.figure(figsize=(8, 8))
y_axis = np.linspace(-5, 15, 100)
plt.xlim(-5, 10)
plt.ylim(-5, 10)

for i in range(len(coeff_mat)):
    x_axis = np.array(coeff_mat[i][0]) * y_axis + np.array(coeff_mat[i][1])
    plt.plot(x_axis, y_axis, linewidth=4)

x1 = np.linspace(-5, 10, 150)
y1, y2 = - x1 + 5, x1 + 1

plt.fill_between(x1, y1, -5, where=x1 <= 3.1, alpha=0.5, label='Class 1')
plt.fill_between(x1, y2, -5, where=x1 >= 2.9, alpha=0.5, label='Class 2')
plt.fill_between(x1, y1, np.max(y1), where=x1 <= 2.1, color='orchid', alpha=0.5, label='Class 3')
plt.fill_between(x1, y2, np.max(y2), where=x1 >= 2, color='orchid', alpha=0.5)

x_bar = plt.gca()
x_bar.spines['right'].set_color('none')
x_bar.spines['top'].set_color('none')
x_bar.spines['bottom'].set_position(('data', 0))
x_bar.spines['left'].set_position(('data', 0))

plt.scatter(dot_x, dot_y, color='r', zorder = 100)
plt.grid(True)
plt.legend()

plt.show()
