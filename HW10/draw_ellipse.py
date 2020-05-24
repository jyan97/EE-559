import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# n_std is the scaler of Ma-Distance. Here we pick 1 as default
def draw_ellipse(mean, cov, ax, n_std=1, facecolor='none', **kwargs):
    pearson = cov[0][1] / np.sqrt(cov[0][0] * cov[1][1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0][0]) * n_std
    mean_x = mean[0]

    scale_y = np.sqrt(cov[1][1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


fig, axis = plt.subplots(figsize=(6, 6))

cov_mat = [[0.5, -0.5],
           [-0.5, 2]]
# mu1 = [1, 2]
# mu2 = [1, -1]
# mu3 = [-2,2]

# Parameters for Problem 1

mu1 = [1, 4]
mu2 = [4, 2]

axis.axvline(c='grey', lw=1)
axis.axhline(c='grey', lw=1)
draw_ellipse(mu1, cov_mat, axis, n_std=1,
             label='1', edgecolor='firebrick')
draw_ellipse(mu2, cov_mat, axis, n_std=1,
             label='2', edgecolor='fuchsia')
# draw_ellipse(mu3, cov_mat, axis, n_std=1,
#              label='3', edgecolor='blue')
# Parameters for Problem 1

xx = np.arange(-10, 10)
yy1 = -10 * xx + 28 + 1.5 * np.log(0.1)
plt.plot(xx, yy1)
yy1 = -10 * xx + 28 + 1.5 * np.log(0.4)
plt.plot(xx, yy1)

# yy1 = -1*xx+3/2
# plt.plot(xx, yy1)
# yy2 = -2*xx +1
# plt.plot(xx, yy2)
# yy3 = xx
# plt.plot(0*xx-1/2, yy3)

# yy1 = 0*xx+1/2
# plt.plot(xx, yy1)
# yy2 = 2*xx +3/2
# plt.plot(xx, yy2)
# yy3 = xx
# plt.plot(0*xx-1/2, yy3)

# plt.xlim(-4, 3)
# plt.ylim(-3, 4)
plt.xlim(-0, 8)
plt.ylim(-0, 8)
plt.show()
