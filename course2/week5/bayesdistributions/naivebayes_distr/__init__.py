from sklearn import datasets, cross_validation, naive_bayes
import numpy as np

def out(filename, str):
    with open(filename, 'w') as f:
        f.write(str)


data_digits = datasets.load_digits()
print data_digits.data[:10]

data_cancer = datasets.load_breast_cancer()
print data_cancer.data[:10]

print('Dataset digits:\n')
bernoulli = naive_bayes.BernoulliNB()
bernoulli_score = np.mean(cross_validation.cross_val_score(bernoulli, data_digits.data, data_digits.target))
print('Bernoulli score = '), bernoulli_score

multinomial = naive_bayes.MultinomialNB()
multinomial_score = np.mean(cross_validation.cross_val_score(multinomial, data_digits.data, data_digits.target))
print('Multinomial score = '), multinomial_score

gauss = naive_bayes.GaussianNB()
gauss_score = np.mean(cross_validation.cross_val_score(gauss, data_digits.data, data_digits.target))
print('Gaussian score = {}\n').format(gauss_score)
out('2.txt', str(np.max([bernoulli_score, multinomial_score, gauss_score])))


print('Dataset cancer:\n')
bernoulli = naive_bayes.BernoulliNB()
bernoulli_score = np.mean(cross_validation.cross_val_score(bernoulli, data_cancer.data, data_cancer.target))
print('Bernoulli score = '), bernoulli_score

multinomial = naive_bayes.MultinomialNB()
multinomial_score = np.mean(cross_validation.cross_val_score(multinomial, data_cancer.data, data_cancer.target))
print('Multinomial score = '), multinomial_score

gauss = naive_bayes.GaussianNB()
gauss_score = np.mean(cross_validation.cross_val_score(gauss, data_cancer.data, data_cancer.target))
print('Gaussian score = '), gauss_score
out('1.txt', str(np.max([bernoulli_score, multinomial_score, gauss_score])))
