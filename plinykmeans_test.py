import numpy as np
from sklearn.cluster import KMeans
from gensim.models.doc2vec import Doc2Vec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pliny_model = Doc2Vec.load('pliny.model')
arr = np.zeros(shape=(len(pliny_model.docvecs) - 1, 100))

for i in range(0, len(pliny_model.docvecs) - 1):
    arr[i] = pliny_model.docvecs[i]
sse = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(arr)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.plot()
plt.savefig('results.png')


