from sklearn.svm import SVC
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import plotSVMBoundaries as pSVMB


def read_data(data_dir):
    with open(data_dir, 'r') as dataset:
        reader = dataset.readlines()
        data = [rows.split(',') for rows in reader]
        data = [list(map(float, e)) for e in data]
    return np.array(data)



training_data1 = read_data("HW8_1_csv/train_x.csv")
training_label1 = read_data("HW8_1_csv/train_y.csv")
training_label1 = training_label1.T

#question1
#(a)
training = np.loadtxt(open("HW8_1_csv/train_x.csv"), delimiter=",")
training_label=np.loadtxt(open("HW8_1_csv/train_y.csv"), delimiter=",")

training_label1 = np.hstack(training_label1)

print(training_label1)
print(training_label)
