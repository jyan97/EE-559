from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
import numpy as np
from sklearn.multiclass import OneVsRestClassifier


class mse_binary(LinearRegression):
    def __init__(self):
        super(mse_binary, self).__init__()

    def predict(self, X):
        thr = 0.5
        y = self._decision_function(X)
        return [1 if i > thr else 0 for i in y]


def read_data(data_dir):
    with open(data_dir, 'r') as dataset:
        reader = dataset.readlines()
        data_set = [rows.split(',') for rows in reader]
        data_set = [list(map(float, e)) for e in data_set]
        label_set = list(map(int, [e.pop(-1) for e in data_set]))
    return np.array(data_set), np.array(label_set)


data_set, label_set = read_data("wine_train.csv")
test_data, test_label = read_data(("wine_test.csv"))

scal = preprocessing.StandardScaler().fit(np.array(data_set))
data_set_tsf = scal.transform(data_set)
test_data_tsf = scal.transform(test_data)

###g1
# model = mse_binary()
# clf = OneVsRestClassifier(model)
# clf.fit(data_set_tsf[:,0:2], label_set)
# # print(clf.coef_, clf.intercept_)
# print('The accuracy for (g) is :', clf.score(test_data_tsf[:,0:2], test_label))

###g2
model = mse_binary()
clf = OneVsRestClassifier(model)
clf.fit(data_set_tsf, label_set)
# print(clf.coef_, clf.intercept_)
print('The accuracy for (g) is :', clf.score(test_data_tsf, test_label))



# print(model.score(data_set_tsf, label_set))
# a = [0 if m == n else 1 for m,n in zip(yy.tolist(), label_set)]
# print(a.count(0)/len(a))



