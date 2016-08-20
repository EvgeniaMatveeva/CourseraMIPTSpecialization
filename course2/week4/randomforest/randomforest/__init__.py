import sklearn
import numpy as np
import scipy
from sklearn import datasets
from sklearn import cross_validation
from sklearn import ensemble, tree
import matplotlib
from matplotlib import pyplot as plt

def out(filename, str):
    f = open(filename, 'w')
    f.write(str)
    f.close()

def score(X, y, estimator):
    scores = cross_validation.cross_val_score(estimator, X, y, cv=10)
    return scores.mean()

digits = datasets.load_digits()
print digits.data.shape
X = digits.data
y = digits.target

# tree_classifier = tree.DecisionTreeClassifier()
# tree_score = score(X, y, tree_classifier)
# print "1: %f" % tree_score
# out('1.txt', str(tree_score))
#
# bagging_classifier = ensemble.BaggingClassifier(n_estimators=100)
# bagging_score = score(X, y, bagging_classifier)
# print "2: %f" % bagging_score
# out('2.txt', str(bagging_score))

d = X.shape[1]
# featuresubset_bagging_classifier = ensemble.BaggingClassifier(n_estimators=100, max_features=1 / d ** 0.5)
# featuresubset_bagging_score = score(X, y, featuresubset_bagging_classifier)
# print "3: %f" % featuresubset_bagging_score
# out('3.txt', str(featuresubset_bagging_score))


# featuresubset_tree = tree.DecisionTreeClassifier(max_features='sqrt')
# featuresubset_classifier = ensemble.BaggingClassifier(n_estimators=100, base_estimator=featuresubset_tree)
# featuresubset_score = score(X, y, featuresubset_classifier)
# print "4 manual random forest: %f" % featuresubset_score
# out('4.txt', str(featuresubset_score))

# randomforest_classifier = ensemble.RandomForestClassifier()
# randomforest_score = score(X, y, randomforest_classifier)
# print "random forest score: %f" % randomforest_score

def compare_plot(x, scores, feature):
    plt.plot(x, scores)
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(scores) - 0.05, max(scores) + 0.05)
    plt.xlabel('number of %s' % feature)
    plt.ylabel('score')
    plt.grid('on')
    plt.show()

# n_trees = xrange(1, 100, 10)
# n_scores = [score(X, y, ensemble.RandomForestClassifier(n_estimators=n)) for n in n_trees]
# compare_plot(n_trees, n_scores, 'trees')

# n_features = xrange(5, 60, 5)
# n_scores = [score(X, y, ensemble.RandomForestClassifier(max_features=n)) for n in n_features]
# compare_plot(n_features, n_scores, 'features')

randomforest_smalldepth = ensemble.RandomForestClassifier(max_depth=1)
randomforest_10depth = ensemble.RandomForestClassifier(max_depth=10)
randomforest_maxdepth= ensemble.RandomForestClassifier()
smalldepth_score = score(X, y, randomforest_smalldepth)
depth_score = score(X, y, randomforest_10depth)
maxdepth_score = score(X, y, randomforest_maxdepth)
print "random forest small depth: %f" % smalldepth_score
print "random forest 10 depth: %f" % depth_score
print "random forest max depth: %f" % maxdepth_score