from sklearn.linear_model import Perceptron
from sklearn import preprocessing
import numpy as np
from sklearn.multiclass import OneVsRestClassifier


def read_data(data_dir):
    with open(data_dir, 'r') as dataset:
        reader = dataset.readlines()
        data_set = [rows.split(',') for rows in reader]
        data_set = [list(map(float, e)) for e in data_set]
        label_set = [e.pop(-1) for e in data_set]
    return data_set, label_set


data_set, label_set = read_data("wine_train.csv")
test_data, test_label = read_data(("wine_test.csv"))

# calculate = np.array(data_set).T
# cal = [np.around([np.mean(x), np.std(x)], decimals = 3).tolist() for x in calculate]
# print(cal)

scal = preprocessing.StandardScaler().fit(np.array(data_set))
data_set_tsf = scal.transform(data_set)
test_data_tsf = scal.transform(test_data)

# data_set_tsf = data_set_tsf[:, 0:2]
# test_data_tsf = test_data_tsf[:, 0:2]

# clf = Perceptron()
# clf.fit(data_set_tsf, label_set)
# print("The weight vector w2 and w1 is : \n", clf.coef_)
# print("The weight vector w0 is: \n", clf.intercept_)
# print('The accuracy for (d) is : \n', clf.score(data_set_tsf, label_set))

err_array, weights, intercepts = [], [], []
for i in range(0, 100):
    clf = Perceptron()
    clf.fit(data_set_tsf, label_set, coef_init=np.random.rand(3, 13), intercept_init=np.random.rand(3))
    err_array.append(clf.score(data_set_tsf, label_set))
    weights.append(clf.coef_)
    intercepts.append(clf.intercept_)

least_index = err_array.index(min(err_array))
best_weight = weights[least_index]
best_intercept = intercepts[least_index]


clf_best = Perceptron()
clf_best.fit(data_set_tsf, label_set, coef_init=best_weight, intercept_init=best_intercept)
print("The best accuracy among 100 attempts with w2 and w1 is : \n", best_weight)
print("The best accuracy among 100 attempts with w0 is: \n", best_intercept)
print('The best accuracy among 100 attempts is : \n', clf_best.score(test_data_tsf, test_label))


