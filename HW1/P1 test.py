import csv
import math


def read_and_mean(dir):
    with open(dir, 'r') as train_data:
        reader = csv.reader(train_data)
        x_data, y_data, label = [], [], []
        for rows in reader:
            x_data.append(rows[0])
            y_data.append(rows[1])
            label.append(rows[2])
        x_data = list(map(float, x_data))
        y_data = list(map(float, y_data))
        label = list(map(float, label))

    class_no = len(set(label))  # number of class labels
    # class_index = [[] for i in range(class_no)]   # the index of classes with size k * n
    class_mean = []
    for i in range(class_no):
        temp = [j for j in range(len(label)) if label[j] == (i + 1)]
        # class_index[i] = temp
        mean_coordinate = []
        mean_coordinate.append(sum(x_data[min(temp):(max(temp) + 1)]) / len(temp))
        mean_coordinate.append(sum(y_data[min(temp):(max(temp) + 1)]) / len(temp))
        class_mean.append(mean_coordinate)
    return class_mean


def judge(test, mean):
    temp = []
    out_put = [None] * len(test)
    for i in range(len(mean)):
        temp.append([math.sqrt((test1[0] - mean[i][0]) ** 2 + ((test1[1] - mean[i][1]) ** 2)) for test1 in test])
    for i in range(len(test)):
        out_put[i] = 1 if temp[0][i] < temp[1][i] else 2
    return out_put


mean_data = read_and_mean('./datasets/synthetic1_train.csv')

with open('./datasets/synthetic1_test.csv', 'r') as train_data:
    reader = csv.reader(train_data)
    test_label = []
    test = [data for data in reader]
    test_data = [data[:2] for data in test]
    test_label = [data[2] for data in test]
    test_label = list(map(int, test_label))

for i in range(len(test_data)):
    test_data[i] = list(map(float, test_data[i]))


a = judge(test_data, mean_data)
print(a)
print(test_label)
