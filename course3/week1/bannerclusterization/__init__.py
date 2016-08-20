import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from collections import Counter


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()

raw_data = pd.read_csv('checkins.csv')

raw_data.columns = [c.strip() for c in raw_data.columns]
data = raw_data[['latitude', 'longitude']]
data.replace(r'\s+', np.nan, inplace=True, regex=True)
data = data.dropna()

subset = data[0:100000]
#print subset.shape
ms = MeanShift(bandwidth=0.1, bin_seeding=True)
ms.fit(subset)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

cnt = Counter(labels)
min_cluster_size = 15
real_cluster_labels = [c for c, n in cnt.items() if n > min_cluster_size]

real_cluster_centers = cluster_centers[real_cluster_labels]
print len(real_cluster_centers)
offices = np.array([[33.751277, -118.188740],
           [25.867736, -80.324116],
           [51.503016, -0.075479],
           [52.378894, 4.885084],
           [39.366487, 117.036146],
           [-33.868457, 151.205134]])

# def min_dist(a):
#     dist = np.array([np.linalg.norm(a - b) for b in offices])
#     return dist.min()

cluster_min_distances = [min(map(lambda a: np.linalg.norm(a-c), offices)) for c in real_cluster_centers]

print len(cluster_min_distances)

min_dist = min(cluster_min_distances)
nearest_center = real_cluster_centers[cluster_min_distances.index(min(cluster_min_distances))]

print "Smallest distance %f" % min_dist
print nearest_center
out('result.txt', str(nearest_center[0]) + ' ' + str(nearest_center[1]))