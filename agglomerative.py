import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, [0, 1]].values

model = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='ward')
model.fit(X)
labels = model.labels_

output = plt.scatter(X[:,0], X[:,1], s = 100, c = labels, marker = "o", alpha = 1, )
plt.title("Hasil Klustering Agglomerative")
plt.colorbar (output)
plt.show()
