from sklearn.svm import SVC
import numpy as np

np.set_printoptions(threshold=np.inf)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import plotSVMBoundaries


def read_data(data_dir):
    with open(data_dir, 'r') as dataset:
        reader = dataset.readlines()
        data = [rows.split(',') for rows in reader]
        data = [list(map(float, e)) for e in data]
    return np.array(data)

# # a
# training_data = read_data("HW8_1_csv/train_x.csv")
# training_label = read_data("HW8_1_csv/train_y.csv")
# training_label = np.hstack(training_label.T)
#
# # c=1
# # svm_c_1 = SVC(kernel='linear', C=1)
# # svm_c_1.fit(training_data, training_label)
# # svm_c_1_predict = svm_c_1.predict(training_data)
# # svm_c_1_accu = accuracy_score(training_label, svm_c_1_predict)
# # print('The accuracy when c=1 is: ', svm_c_1_accu)
# # plotSVMBoundaries.plotSVMBoundaries(training_data, training_label, svm_c_1)
#
# # c=100
# svm_c_100 = SVC(kernel='linear', C=100)
# svm_c_100.fit(training_data, training_label)
# svm_c_100_predict = svm_c_100.predict(training_data)
# svm_c_100_accu = accuracy_score(training_label, svm_c_100_predict)
# print('The accuracy when c=100 is: ', svm_c_100_accu)
# # plotSVMBoundaries.plotSVMBoundaries(training_data, training_label, svm_c_100)
#
# # b
# support_vector_c_100 = svm_c_100.support_vectors_
# plotSVMBoundaries.plotSVMBoundaries(training_data, training_label, svm_c_100, support_vector_c_100)
# w_vector = svm_c_100.coef_
# w0_vector = svm_c_100.intercept_
# print("The support vector is: (", w_vector[0][1], w_vector[0][0], w0_vector[0], ")")
# print("The decision boundary equation is : ", w_vector[0][0], "x1+", w_vector[0][1], "x2+", w0_vector[0], "= 0")
#
# # c
# gx = [m[1]*w_vector[0][1]+ m[0]*w_vector[0][0]+w0_vector[0] for m in support_vector_c_100]
# print("The g(x) value of the support vectors are: ", gx)

# d
training_data_rbf = read_data("HW8_2_csv/train_x.csv")
training_label_rbf = read_data("HW8_2_csv/train_y.csv")
training_label_rbf = np.hstack(training_label_rbf.T)

# rbf C=50
# rbf_c_50 = SVC(kernel='rbf', C=50)
# rbf_c_50.fit(training_data_rbf, training_label_rbf)
# rbf_c_50_predict = rbf_c_50.predict(training_data_rbf)
# rbf_c_50_accu = accuracy_score(training_label_rbf, rbf_c_50_predict)
# print('The SVM accuracy when c=50 with rbf is: ', rbf_c_50_accu)
# plotSVMBoundaries.plotSVMBoundaries(training_data_rbf, training_label_rbf, rbf_c_50)

# rbf C=5000
# rbf_c_50 = SVC(kernel='rbf', C=5000, gamma='auto')
# rbf_c_50.fit(training_data_rbf, training_label_rbf)
# rbf_c_50_predict = rbf_c_50.predict(training_data_rbf)
# rbf_c_50_accu = accuracy_score(training_label_rbf, rbf_c_50_predict)
# print('The SVM accuracy when c=5000 with rbf is: ', rbf_c_50_accu)
# plotSVMBoundaries.plotSVMBoundaries(training_data_rbf, training_label_rbf, rbf_c_50)

# e
# gamma = 10
# rbf_c_50 = SVC(kernel='rbf', gamma=10)
# rbf_c_50.fit(training_data_rbf, training_label_rbf)
# rbf_c_50_predict = rbf_c_50.predict(training_data_rbf)
# rbf_c_50_accu = accuracy_score(training_label_rbf, rbf_c_50_predict)
# print('The SVM accuracy when c=5000 with rbf is: ', rbf_c_50_accu)
# plotSVMBoundaries.plotSVMBoundaries(training_data_rbf, training_label_rbf, rbf_c_50)

# gamma = 50
# rbf_c_50 = SVC(kernel='rbf', gamma=50)
# rbf_c_50.fit(training_data_rbf, training_label_rbf)
# rbf_c_50_predict = rbf_c_50.predict(training_data_rbf)
# rbf_c_50_accu = accuracy_score(training_label_rbf, rbf_c_50_predict)
# print('The SVM accuracy when c=5000 with rbf is: ', rbf_c_50_accu)
# plotSVMBoundaries.plotSVMBoundaries(training_data_rbf, training_label_rbf, rbf_c_50)

# gamma = 500
rbf_c_50 = SVC(kernel='rbf', gamma=500)
rbf_c_50.fit(training_data_rbf, training_label_rbf)
rbf_c_50_predict = rbf_c_50.predict(training_data_rbf)
rbf_c_50_accu = accuracy_score(training_label_rbf, rbf_c_50_predict)
print('The SVM accuracy when c=5000 with rbf is: ', rbf_c_50_accu)
plotSVMBoundaries.plotSVMBoundaries(training_data_rbf, training_label_rbf, rbf_c_50)



