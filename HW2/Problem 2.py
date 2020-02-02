import math
import numpy as np
from plotDecBoundaries import plotDecBoundaries  # datasets and script are supposed to be in same directory

train_data_dir = 'wine_train.csv'  # datasets and script are supposed to be in same directory
test_data_dir = 'wine_test.csv'
first_column = 0  # list index which starts from 0
second_column = 1  # list index which starts from 0
label_column = 13  # list index which starts from 0


def read_and_mean(directory):
    with open(directory, 'r') as train_data:
        reader = train_data.read().splitlines()
        x_data, y_data, label = [], [], []
        for rows in reader:
            rows = rows.split(',')
            x_data.append(rows[first_column])
            y_data.append(rows[second_column])
            label.append(rows[label_column])
        x_data = list(map(float, x_data))
        y_data = list(map(float, y_data))
        label = list(map(int, label))

    class_no = len(set(label))  # number of class labels
    class_mean = []
    for i in range(class_no):
        complement_x = [x_data[m] for m in range(len(label)) if label[m] != i + 1]
        complement_y = [y_data[m] for m in range(len(label)) if label[m] != i + 1]
        data_x = [x_data[m] for m in range(len(label)) if label[m] == i + 1]
        data_y = [y_data[m] for m in range(len(label)) if label[m] == i + 1]
        class_mean.append([sum(data_x)/len(data_x), sum(data_y)/len(data_y),
                           sum(complement_x) / len(complement_x), sum(complement_y) / len(complement_y)])
    return class_mean


print(read_and_mean('wine_train.csv'))
