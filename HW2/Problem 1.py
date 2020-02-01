import math
import numpy as np
from matplotlib import pyplot as plt

coeff_mat = [[-1, -1, 5], [-1, 0, 3], [-1, 1, -1]]
coeff_mat = [[-(x[1]/x[0]), -(x[2]/x[0])] for x in coeff_mat]

plt.figure(figsize=(8, 8))
y_axis = np.linspace(-5, 15, 100)
plt.xlim(-15, 15)
plt.ylim(-5, 15)


for i in range(len(coeff_mat)):
    x_axis = np.array(coeff_mat[i][0]) * y_axis + np.array(coeff_mat[i][1])
    plt.plot(x_axis, y_axis)
    plt.fill_between


x_bar = plt.gca()
x_bar.spines['right'].set_color('none')
x_bar.spines['top'].set_color('none')

x_bar.spines['bottom'].set_position(('data', 0))
x_bar.spines['left'].set_position(('data', 0))


plt.show()
