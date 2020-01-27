import math
import numpy as np
from datasets.plotDecBoundaries import plotDecBoundaries

train_data_dir = './datasets/wine_train.csv'
test_data_dir = './datasets/wine_test.csv'
first_column = 0
second_column = 11
label_column = 13


def read_and_mean(dir, i, j, k):
    with open(dir, 'r') as train_data:
        reader = train_data.read().splitlines()
        x_data, y_data, label = [], [], []
        for rows in reader:
            rows = rows.split(',')
            x_data.append(rows[i])
            y_data.append(rows[j])
            label.append(rows[k])
        x_data = list(map(float, x_data))
        y_data = list(map(float, y_data))
        label = list(map(int, label))

    class_no = len(set(label))  # number of class labels
    # class_index = [[] for i in range(class_no)]   # the index of classes with size k * n
    class_mean = []
    for i in range(class_no):
        temp = [j for j in range(len(label)) if label[j] == (i + 1)]  # for label 1, temp is from 0 to 49
        # class_index[i] = temp
        mean_coordinate = [sum(x_data[min(temp):(max(temp) + 1)]) / len(temp),
                           sum(y_data[min(temp):(max(temp) + 1)]) / len(temp)]
        class_mean.append(mean_coordinate)
    return x_data, y_data, label, class_mean


def judge(test, mean):
    temp = []
    out_put = [None] * len(test)
    for i in range(len(mean)):
        temp.append([math.sqrt((test1[0] - mean[i][0]) ** 2 + ((test1[1] - mean[i][1]) ** 2)) for test1 in test])
    for i in range(len(test)):
        out_put[i] = 1 if temp[0][i] < temp[1][i] else 2
    return out_put


def error_rate(output_label, test_label):
    boolean_out = [1 if i == j else 0 for i, j in
                   zip(output_label, test_label)]  # input 1 if the output is correct, else 0
    correct = 0
    for i in range(len(boolean_out)):
        if boolean_out[i] == 1: correct += 1
    return correct / len(output_label)


def read_test_data(test_dir, i, j, k):
    with open(test_dir, 'r') as train_data:
        reader = train_data.readlines()  # if using csv, the reader could only read once since it's a iterator
        test = [rows.split(',') for rows in reader]
        test_data = [data[:2] for data in test]
        test_label = [data[13] for data in test]
        test_label = list(map(int, test_label))
    for i in range(len(test_data)):
        test_data[i] = list(map(float, test_data[i]))
    return test_data, test_label


# test_data, test_label = read_test_data(test_data_dir, first_column, second_column, label_column)
# x_data, y_data, train_label, mean_data = read_and_mean(train_data_dir, first_column, second_column, label_column)
# train_dataset = np.array([x_data, y_data]).T  # merge x_data and y_data and convert to np.array
# mean_data = np.array(mean_data)
# train_label = np.array(train_label)
#
# output = judge(train_dataset.tolist(), mean_data)  # or replace test_data with train_dataset.tolist()
# err_rate = 1 - error_rate(output, train_label.tolist())  # or replace test_label with train_label.tolist()
# print("The error rate is:", err_rate)
#
# plotDecBoundaries(train_dataset, train_label, mean_data)

opt_i, opt_j, opt_err_rate = 0, 0, 100
for i in range(label_column):  # we assume the label is the last column
    for j in range(i, label_column):
        test_data, test_label = read_test_data(test_data_dir, i, j, label_column)
        x_data, y_data, train_label, mean_data = read_and_mean(train_data_dir, i, j, label_column)
        train_dataset = np.array([x_data, y_data]).T  # merge x_data and y_data and convert to np.array
        train_label = np.array(train_label)

        output = judge(train_dataset.tolist(), mean_data)  # or replace test_data with train_dataset.tolist()
        err_rate = 1 - error_rate(output, train_label.tolist())  # or replace test_label with train_label.tolist()
        if err_rate < opt_err_rate: opt_i, opt_j, opt_err_rate = i, j, err_rate


print("The minimum error rate comes with columns {} and {} with an error-rate of {}".format(opt_i, opt_j, opt_err_rate))

