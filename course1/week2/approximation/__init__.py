import numpy as np
from scipy import linalg
from matplotlib import pylab as plt

def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


def func(x):
    return np.sin(x/5.)*np.exp(x/10.) + 5*np.exp(-x/2.)

def get_a(points):
    return np.array([[np.power(p, k) for k in xrange(len(points))] for p in points])

def get_b(points):
    return np.array([func(x) for x in points])

def get_w(a, b):
    return linalg.solve(a, b)

def plot_pn(points, coeff):
    approx = [sum([w*np.power(x, k) for (k, w) in enumerate(coeff)]) for x in points]
    plt.plot(points, approx)

def plot_true(points):
    y = [func(x) for x in points]
    plt.plot(points, y)

xs = np.arange(1, 15, 0.1)
plot_true(xs)
x1 = np.array([1, 15])
w1 = get_w(get_a(x1), get_b(x1))
plot_pn(x1, w1)

x2 = np.array([1, 8, 15])
w2 = get_w(get_a(x2), get_b(x2))
plot_pn(x2, w2)

x3 = np.array([1, 4, 10, 15])
w3 = get_w(get_a(x3), get_b(x3))
plot_pn(x3, w3)
result = ' '.join([str(w) for w in w3])
print result
out('task2.txt', result)

plt.show()