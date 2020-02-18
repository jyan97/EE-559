import numpy as np
from plotDecBoundaries import plotDecBoundaries
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


x, x_reflected, y = read_data("synthetic1_train.csv", "synthetic1_train_label.csv")
index = np.random.permutation(len(y))
x, x_reflected, y = x[index], x_reflected[index], y[index]  # shuffle data and label simultaneously
w_temp = np.asarray([0.1, 0.1, 0.1])
loop_counter = 0
stop_sign = 0
w = w_temp.copy()
while not stop_sign:
    no_change_count = np.zeros(len(y))
    for i in range(len(y)):
        if np.sum(w_temp * x_reflected[i]) <= 0:
            w = np.vstack((w, w_temp + x_reflected[i]))
            no_change_count[i] = 1
        else:
            w = np.vstack((w, w_temp))
        w_temp = w[-1].copy()
    loop_counter += 1
    if loop_counter == 1000 or np.count_nonzero(no_change_count) == 0: stop_sign = 1

w_test = w[-1]
wrong = np.zeros(len(y))
for i in range(len(y)):
    if x[i][0] * w_test[0] + x[i][1] * w_test[1] + w[-1][2] < 0 and y[i] == 1:
        wrong[i] = 1
    elif x[i][0] * w_test[0] + x[i][1] * w_test[1] + w[-1][2] > 0 and y[i] == 2:
        wrong[i] = 1

err_rate = np.count_nonzero(wrong) / len(y)
print("The final weight vector is:", w_test)
print("The error rate is :", err_rate)
plotDecBoundaries(x, y, w_test)
