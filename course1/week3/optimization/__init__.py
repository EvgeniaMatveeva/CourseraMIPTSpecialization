import scipy
from scipy import optimize
import numpy as np
from matplotlib import pylab as plt
import timeit


def out(filename, s):
    f = open(filename, 'w')
    f.write(s)
    f.close()


def func(x):
    return np.sin(x / 5.) * np.exp(x / 10.) + 5 * np.exp(-x / 2.)


def int_func(x):
    return int(func(x))


def compare_timings():
    n = 100
    print('Gradient:')
    print timeit.timeit('minimize(lambda x: np.sin(x / 5.) * np.exp(x / 10.) + 5 * np.exp(-x / 2.), 2, method=\'BFGS\')', setup='import numpy as np; from scipy.optimize import minimize', number=n)
    print('Differential evolution:')
    print timeit.timeit('differential_evolution(lambda x: np.sin(x / 5.) * np.exp(x / 10.) + 5 * np.exp(-x / 2.), [(1, 30)])', setup='import numpy as np; from scipy.optimize import differential_evolution', number=n)


xs = np.arange(1, 30, 0.1)
y = func(xs)
plt.plot(xs, y)

min2 = scipy.optimize.minimize(func, 2, method='BFGS')
#print min2
min30 = scipy.optimize.minimize(func, 30, method='BFGS')
#print min30

s = '%.2f %.2f' % (min2.fun, min30.fun)
#print s
out('task1.txt', s)

bounds = [(1, 30)]
minfunc_evolution = scipy.optimize.differential_evolution(func, bounds)
#print minfunc_evolution
print '\n'
out('task2.txt', '%.2f' % minfunc_evolution.fun)

h = np.array([int_func(x) for x in xs])
plt.plot(xs, h)
plt.show()

minh_grad = scipy.optimize.minimize(int_func, 30, method='BFGS')
print minh_grad
minh_evolution = scipy.optimize.differential_evolution(int_func, bounds)
#print minh_evolution.fun
out('task3.txt', '%.2f %.2f' % (minh_grad.fun, minh_evolution.fun))

