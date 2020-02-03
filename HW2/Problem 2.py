import math
import numpy as np
from plotDecBoundaries import plotDecBoundaries  # datasets and script are supposed to be in same directory
from scipy.spatial.distance import cdist

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
        class_mean.append([sum(data_x) / len(data_x), sum(data_y) / len(data_y),
                           sum(complement_x) / len(complement_x), sum(complement_y) / len(complement_y)])
    return list(map(list, zip(*[x_data, y_data]))), label, class_mean


# def cal_bound(mean_arr):
#     return [[(m[0]-m[2])/(m[3]-m[1]), (m[2]**2-m[0]**2+m[3]**2-m[1]**2)/(2*(m[3]-m[1]))] for m in mean_arr]

def judge_and_err(input_data, mean):
    ss = [1 if (m[0] - i[0]) ** 2 + (m[1] - i[1]) ** 2 < (m[2] - i[0]) ** 2 + (m[3] - i[1]) ** 2 else 0 for i in
          input_data for m in mean]  # direct distance calculation instead of calling cdist function
    judged = list(zip(*[iter(ss)] * 3))  # using iteration to group every 3 of the elements in the list
    # judged = [(1, 0, 0), (1, 0, 0), (1, 1, 0)...]
    seq = [m.index(1) + 1 if m.count(1) == 1 else 0 for m in judged]
    # seq = [1, 1, 0, 1, 1, 1, ..., 2, 0 ...]
    return judged, seq


train_data, train_label, train_mean = read_and_mean('wine_train.csv')
judged, seq = judge_and_err(train_data, train_mean)
