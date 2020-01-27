import math
import numpy as np
from datasets.plotDecBoundaries import plotDecBoundaries


def read_and_mean(dir):
    with open(dir, 'r') as train_data:
        reader = train_data.read().splitlines()
        x_data, y_data, label = [], [], []
        for rows in reader:
            rows = rows.split(',')
            x_data.append(rows[0])
            y_data.append(rows[1])
            label.append(rows[2])
        x_data = list(map(float, x_data))
        y_data = list(map(float, y_data))
        label = list(map(int, label))

    class_no = len(set(label))  # number of class labels
    # class_index = [[] for i in range(class_no)]   # the index of classes with size k * n
    class_mean = []
    for i in range(class_no):
        temp = [j for j in range(len(label)) if label[j] == (i + 1)] # for label 1, temp is from 0 to 49
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


x_data, y_data, label_data, mean_data = read_and_mean('./datasets/synthetic1_train.csv')

with open('./datasets/synthetic1_test.csv', 'r') as train_data:
    reader = train_data.readlines()               # if you use csv, the reader could only read once
    test = [rows.split(',') for rows in reader]
    test_data = [data[:2] for data in test]
    test_label = [data[2] for data in test]
    test_label = list(map(int, test_label))

for i in range(len(test_data)):
    test_data[i] = list(map(float, test_data[i]))


output = judge(test_data, mean_data)
err_rate = 1 - error_rate(output, test_label)
print("The error rate is:", err_rate)

train_dataset = [x_data, y_data]
train_dataset = np.array(train_dataset)  # merge x and y training data and transfer into np.array

mean_data = np.array(mean_data)
label_data = np.array(label_data)
plotDecBoundaries(train_dataset.T, label_data, mean_data)
