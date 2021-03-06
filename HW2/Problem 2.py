import math
import numpy as np
from plotDecBoundaries import plotDecBoundaries  # datasets and script are supposed to be in same directory

train_data_dir = 'wine_train.csv'  # datasets and script are supposed to be in same directory
test_data_dir = 'wine_test.csv'
first_column = 0  # list index which starts from 0
second_column = 1  # list index which starts from 0
label_column = 13  # list index which starts from 0


def read_and_mean(directory):
    global first_column, second_column, label_column
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


def read_test_data(test_dir):
    global first_column, second_column, label_column
    with open(test_dir, 'r') as train_data:
        reader = train_data.readlines()  # if using csv, the reader could only read once since it's a iterator
        test = [rows.split(',') for rows in reader]
        test_data = [[data[first_column], data[second_column]] for data in test]
        test_label = [data[label_column] for data in test]
        test_label = list(map(int, test_label))
    for first_column in range(len(test_data)):
        test_data[first_column] = list(map(float, test_data[first_column]))
    return test_data, test_label

# def cal_bound(mean_arr):
#     return [[(m[0]-m[2])/(m[3]-m[1]), (m[2]**2-m[0]**2+m[3]**2-m[1]**2)/(2*(m[3]-m[1]))] for m in mean_arr]

def judge_and_err(input_data, mean):
    ss = [1 if (m[0] - i[0]) ** 2 + (m[1] - i[1]) ** 2 < (m[2] - i[0]) ** 2 + (m[3] - i[1]) ** 2 else 0 for i in
          input_data for m in mean]  # direct distance calculation instead of calling cdist function
    judged = list(zip(*[iter(ss)] * 3))  # using iteration to group each 3 of the elements in the list
    # judged = [(1, 0, 0), (1, 0, 0), (1, 1, 0)...]
    seq = [m.index(1) + 1 if m.count(1) == 1 else 0 for m in judged]
    # seq = [1, 1, 0, 1, 1, 1, ..., 2, 0 ...]
    return judged, seq


train_data, train_label, train_mean = read_and_mean('wine_train.csv')
test_data, test_label = read_test_data('wine_test.csv')
judged, seq = judge_and_err(train_data, train_mean)  # change train_data to what you like to input  ###
result = list(map(lambda x, y: 1 if x == y else 0, seq, train_label))  # turn 1, 2 and 3 in seq all into 1  ###
print("The accuracy of training dataset is {}.".format(result.count(1) / len(result)))
train_mean_input = np.array(
    list(zip(*[iter(sum(train_mean, []))] * 2)))  # Convert mean_array to format that fit the plot func
plotDecBoundaries(np.array(train_data), np.array(train_label), train_mean_input)  ###

judged, seq = judge_and_err(test_data, train_mean)  # change train_data to what you like to input  ###
result = list(map(lambda x, y: 1 if x == y else 0, seq, test_label))  # turn 1, 2 and 3 in seq all into 1  ###
print("The accuracy of testing dataset is {}.".format(result.count(1) / len(result)))

train_mean_input = np.array(
    list(zip(*[iter(sum(train_mean, []))] * 2)))  # Convert mean_array to format that fit the plot func
print(train_mean_input)
plotDecBoundaries(np.array(test_data), np.array(test_label), train_mean_input)  ###
