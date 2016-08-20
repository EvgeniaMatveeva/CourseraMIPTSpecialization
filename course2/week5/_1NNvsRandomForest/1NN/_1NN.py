import numpy as np


class _1NN_classifier:
    def __init__(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    def getDist(self, x):
        dists = [np.sum([(fi - fj)**2 for fi, fj in zip(t, x)]) for t, y in zip(self.train_X, self.train_y)]
        return dists

    def predict(self, X):
        result = []
        for xi in X:
            dists = self.getDist(xi)
            min_ind = np.argmin(dists)
            response = self.train_y[min_ind]
            result.append(response)
        return result