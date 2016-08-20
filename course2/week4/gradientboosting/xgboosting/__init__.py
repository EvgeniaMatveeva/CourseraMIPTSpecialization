from sklearn import datasets, metrics
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import xgboost as xgb

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

X_train, y_train, X_test, y_test = load_data()
rmse_test_trees = []
rmse_train_trees = []
num_trees = xrange(2, 100, 1)

for n_trees in num_trees:
    bst = xgb.XGBRegressor(max_depth=5, n_estimators=n_trees)
    bst.fit(X_train, y_train)
    rmse_test_trees.append(np.sqrt(metrics.mean_squared_error(y_test, bst.predict(X_test))))
    rmse_train_trees.append(np.sqrt(metrics.mean_squared_error(y_train, bst.predict(X_train))))

rmse_test_depth = []
rmse_train_depth = []
depths = xrange(1, 20, 1)

for depth in depths:
    bst = xgb.XGBRegressor(max_depth=depth)
    bst.fit(X_train, y_train)
    rmse_test_depth.append(np.sqrt(metrics.mean_squared_error(y_test, bst.predict(X_test))))
    rmse_train_depth.append(np.sqrt(metrics.mean_squared_error(y_train, bst.predict(X_train))))

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.plot(num_trees, bst_scores_trees)
# ax1.set_title('RMSE vs num trees')
# ax2.plot(depths, bst_scores_depth)
# ax2.set_title('RMSE vs depth')

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(num_trees, rmse_test_trees, 'r', num_trees, rmse_train_trees, 'b')
plt.xlabel('num trees')
plt.ylabel('RMSE')
plt.grid('on')

plt.subplot(1, 2, 2)
plt.plot(depths, rmse_test_depth, 'r', depths, rmse_train_depth, 'b')
plt.xlabel('depth')
plt.ylabel('RMSE')
plt.grid('on')

plt.show()


