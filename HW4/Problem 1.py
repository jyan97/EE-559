import numpy as np
from matplotlib import pyplot as plt


def read_data(data_dir, label_dir):
    with open(data_dir, 'r') as dataset:
        reader = dataset.readlines()
        data_set = [rows.split(',') for rows in reader]
    for i in range(len(data_set)):
        data_set[i] = list(map(float, data_set[i]))
        data_set[i].append(1)

    with open(label_dir, 'r') as labelset:
        reader = labelset.readlines()
        label_set = [rows.split(',') for rows in reader]
    label_set = [int(i) for j in label_set for i in j]
    data_set_ref = [
        [-data_set[i][0], -data_set[i][1], -data_set[i][2]] if label_set[i] == 2 else [data_set[i][0], data_set[i][1],
                                                                                       data_set[i][2]]
        for i in range(len(label_set))]
    return np.array(data_set), np.array(data_set_ref), np.array(label_set)


x, x_reflected, y = read_data("feature_train.csv", "label_train.csv")
index = np.random.permutation(len(y))
x, x_reflected, y = x[index], x_reflected[index], y[index]  # shuffle data and label simultaneously

w_temp = np.asarray([0.1,0.1,0.1])
loop_counter = 0
stop_sign = 0
w = w_temp.copy()
while not stop_sign:
    no_change_count = np.zeros(len(y))
    for i in range(len(y)):
        if np.sum(w_temp * x_reflected[i]) <= 0 :
            w = np.vstack((w, w_temp + x_reflected[i]))
            no_change_count[i] = 1
        else: w = np.vstack((w, w_temp))
        w_temp = w[-1].copy()
    loop_counter += 1
    if loop_counter == 1000 or np.count_nonzero(no_change_count) == 0: stop_sign = 1


print(w)






# x_points = np.linspace(-10, 10, 20)
# y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
# plt.plot(x_points, y_)
#
# # plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
# # plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
#
# plt.show()
