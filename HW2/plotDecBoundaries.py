###############################################
## EE559 HW Wk2, Prof. Jenkins, Spring 2018
## Created by Arindam Jati, TA
## Modified by Jingquan Yan
## Tested in Python 3.6.3, OSX El Captain
################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def plotDecBoundaries(training, label_train, sample_mean, *argv):
    if len(argv) == 0:
        nclass = max(np.unique(label_train))

        # Set the feature range for ploting
        max_x = np.ceil(max(training[:, 0])) + 1
        min_x = np.floor(min(training[:, 0])) - 1
        max_y = np.ceil(max(training[:, 1])) + 1
        min_y = np.floor(min(training[:, 1])) - 1

        xrange = (min_x, max_x)
        yrange = (min_y, max_y)

        # step size for how finely you want to visualize the decision boundary.
        inc = 0.01

        # generate grid coordinates. this will be the basis of the decision
        # boundary visualization.
        (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc),
                             np.arange(yrange[0], yrange[1] + inc / 100, inc))

        # size of the (x, y) image, which will also be the size of the
        # decision boundary image that is used as the plot background.
        image_size = x.shape
        xy = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1, order='F'),
                        y.reshape(y.shape[0] * y.shape[1], 1, order='F')))  # make (x,y) pairs as a bunch of row vectors.

        # distance measure evaluations for each (x,y) pair.
        dist_mat = cdist(xy, sample_mean)
        pred_label_1 = np.argmax(dist_mat[:, 0:2],
                                 axis=1).tolist()  # It is obvious that here has to be argmax instead of min.
        pred_label_2 = np.argmax(dist_mat[:, 2:4], axis=1).tolist()
        pred_label_3 = np.argmax(dist_mat[:, 4:6], axis=1).tolist()  # I am not familiar with numpy so I use list instead.

        pred_label = list(zip(*[pred_label_1, pred_label_2, pred_label_3]))  # unzip and list
        pred_label = [m.index(1) + 1 if m.count(1) == 1 else 0 for m in pred_label]
        pred_label = np.array(pred_label)

        # reshape the idx (which contains the class label) into an image.
        decisionmap = pred_label.reshape(image_size, order='F')

        # show the image, give each coordinate a color according to its class label
        plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

        # plot the class training data.
        plt.plot(training[label_train == 1, 0], training[label_train == 1, 1], 'rx')
        plt.plot(training[label_train == 2, 0], training[label_train == 2, 1], 'go')
        if nclass == 3:
            plt.plot(training[label_train == 3, 0], training[label_train == 3, 1], 'b*')

        # include legend for training data
        if nclass == 3:
            l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
        else:
            l = plt.legend(('Class 1', 'Class 2'), loc=2)
        plt.gca().add_artist(l)

        # plot the class mean vector.
        m1, = plt.plot(sample_mean[0, 0], sample_mean[0, 1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
        m2, = plt.plot(sample_mean[2, 0], sample_mean[2, 1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
        if nclass == 3:
            m3, = plt.plot(sample_mean[4, 0], sample_mean[4, 1], 'bd', markersize=12, markerfacecolor='b',
                           markeredgecolor='w')

        # include legend for class mean vector
        if nclass == 3:
            l1 = plt.legend([m1, m2, m3], ['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)
        else:
            l1 = plt.legend([m1, m2], ['Class 1 Mean', 'Class 2 Mean'], loc=4)

        plt.gca().add_artist(l1)

    else:
        nclass = len(np.unique(label_train))
        xrange = (-5, 5)
        yrange = (-5, 5)

        # step size for how finely you want to visualize the decision boundary.
        inc = 0.01

        (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc),
                             np.arange(yrange[0], yrange[1] + inc / 100, inc))

        image_size = x.shape
        xy = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1, order='F'), y.reshape(y.shape[0] * y.shape[1], 1,
                                                                                    order='F')))  # make (x,y) pairs as a bunch of row vectors.
        #  meshgrid 一开始x和y是分开的，这里将x和y先reshape展平，然后压扁（每个x和y一组）
        # distance measure evaluations for each (x,y) pair.
        dist_mat = cdist(xy, sample_mean)  # 生成n * 3的距离矩阵，3列是因为有3个class
        pred_label = np.argmin(dist_mat, axis=1)  # 找到每个meshgrid坐标点对三个mean的距离最小的index，即分类。axis=1是每一行找

        # reshape the idx (which contains the class label) into an image.
        decisionmap = pred_label.reshape(image_size, order='F')

        # show the image, give each coordinate a color according to its class label
        plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')  # 将[0,0]放在左上角还是左下角

        # include legend for training data
        if nclass == 3:
            l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
        else:
            l = plt.legend(('Class 1', 'Class 2'), loc=2)
        plt.gca().add_artist(l)

        # plot the class mean vector.
        m1, = plt.plot(sample_mean[0, 0], sample_mean[0, 1], 'rd', markersize=12, markerfacecolor='r',
                       markeredgecolor='w')
        m2, = plt.plot(sample_mean[1, 0], sample_mean[1, 1], 'gd', markersize=12, markerfacecolor='g',
                       markeredgecolor='w')
        if nclass == 3:
            m3, = plt.plot(sample_mean[2, 0], sample_mean[2, 1], 'bd', markersize=12, markerfacecolor='b',
                           markeredgecolor='w')

        # include legend for class mean vector
        if nclass == 3:
            l1 = plt.legend([m1, m2, m3], ['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)
        else:
            l1 = plt.legend([m1, m2], ['Class 1 Mean', 'Class 2 Mean'], loc=4)

        plt.gca().add_artist(l1)
    plt.show()
