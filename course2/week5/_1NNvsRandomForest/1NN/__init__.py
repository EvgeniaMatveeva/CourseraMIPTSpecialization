import numpy as np
from sklearn import datasets, ensemble
from _1NN import _1NN_classifier

def out(filename, str):
    with open(filename, 'w') as f:
        f.write(str)

data = datasets.load_digits()
X = data.data
y = data.target

test_split = int(X.shape[0]*0.25)
print test_split
X_train = X[:(-test_split)]
y_train = y[:(-test_split)]
X_test = X[-test_split:]
y_test = y[-test_split:]


_1NN = _1NN_classifier(X_train, y_train)
_1NN_predict = _1NN.predict(X_test)
_1NN_errors_num = len(np.where(np.array(y_test) - np.array(_1NN_predict) != 0)[0])

print "% of errors = ", float(_1NN_errors_num) / test_split * 100, '%'
out('1.txt', str(float(_1NN_errors_num) / test_split))

forest = ensemble.RandomForestClassifier(n_estimators=1000)
forest.fit(X_train, y_train)
forest_predict = forest.predict(X_test)
forest_errors_num = len(np.where(np.array(y_test) - forest_predict != 0)[0])

print "% of errors = ", float(forest_errors_num) / test_split * 100, '%'
out('2.txt', str(float(forest_errors_num) / test_split))