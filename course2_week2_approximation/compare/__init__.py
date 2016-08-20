import re
import itertools
from collections import Counter
import numpy as np
import scipy
from scipy.spatial.distance import cosine


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()

with open('sentences.txt', 'r') as file:
    words = [re.split('\W', s) for s in file.readlines() if len(s) > 0]
    lines = [[word.lower() for word in row if len(word.strip()) > 0] for row in words]
    unique_words = set(list(itertools.chain.from_iterable(lines)))
    print unique_words

index = {word: i for (i, word) in enumerate(unique_words)}
print 'dict len: %s', len(index)

word_matrix = np.zeros((len(lines), len(index)))

for i, line in enumerate(lines):
    cnt = Counter(line)
    for word in line:
        j = index[word]
        word_matrix[i, index[word]] = cnt[word]

first = np.array(word_matrix[0])
dist = [scipy.spatial.distance.cosine(first, a) for a in word_matrix]
dist_indexes = [(x, dist[x]) for x in range(len(dist))]
sorted_dist = sorted(dist_indexes, key=lambda k: k[1])

print sorted_dist
two_closest = ' '.join([str(i[0]) for i in sorted_dist[1:3]])
print two_closest
out('task1.txt',  two_closest)