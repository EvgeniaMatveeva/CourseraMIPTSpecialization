import numpy as np
from sklearn import datasets, cross_validation, tree, metrics, linear_model
import matplotlib
from matplotlib import pyplot as plt

def out(filename, str):
    file = open(filename, 'w')
    file.write(str)
    file.close()

def load_data():
    data_boston = datasets.load_boston()
    X = data_boston.data
    y = data_boston.target
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=42)
    train_len = X.shape[0]*0.75
    X_train = X[:train_len, :]
    X_test = X[train_len:, :]
    y_train = y[:train_len]
    y_test = y[train_len:]
    return X_train, y_train, X_test, y_test

def gbm_predict(X, base_algorithms_list, coefficients_list):
    values = [sum([coeff * algo.predict([x])[0] for algo, coeff in zip(base_algorithms_list, coefficients_list)]) for x in X]
    return values

def get_shifts(X, y, predict):
    return -(predict(X) - y)

def get_rmse():
    y_pred = gbm_predict(X_test, base_algorithms_list, coefficients_list)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

def train_gboosting(N, maxdepth):
    base_alg_coeff = 0.9
    for i in xrange(N):
        base_alg = tree.DecisionTreeRegressor(max_depth=maxdepth, random_state=42)
        if (i == 0):
            response = np.zeros(y_train.shape[0])
            # base_alg_coeff = 1
        else:
            g = lambda x: gbm_predict(x, base_algorithms_list, coefficients_list)
            response = get_shifts(X_train, y_train, g)
            base_alg_coeff = 0.9 / (1.0 + i)
        base_alg.fit(X_train, response)
        base_algorithms_list.append(base_alg)
        coefficients_list.append(base_alg_coeff)
        error = get_rmse()
        errors.append(error)
    return errors

X_train, y_train, X_test, y_test = load_data()

base_algorithms_list = []
coefficients_list = []
N = 50
errors = []

errors = train_gboosting(N, 5)
result = errors[len(errors) - 1]
print("GBoosting test results: ", result)
#out('2.txt', str(result))
out('3.txt', str(result))

plt.plot(errors)
plt.xlabel('iteration')
plt.ylabel('RMSE')
plt.grid('on')
plt.show()

linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
linear_error = metrics.mean_squared_error(y_test, linear.predict(X_test))**0.5
print("Linear regression test results: ", linear_error)
out('5.txt', str(linear_error))

