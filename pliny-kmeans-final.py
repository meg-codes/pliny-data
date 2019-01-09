import numpy as np
import pickle
from sklearn.cluster import KMeans
from gensim.models.doc2vec import Doc2Vec

pliny_model = Doc2Vec.load('pliny.model')
arr = np.zeros(shape=(len(pliny_model.docvecs) - 1, 100))

for i in range(0, len(pliny_model.docvecs) - 1):
    arr[i] = pliny_model.docvecs[i]

kmeans = KMeans(n_clusters=17, max_iter=1000).fit(arr)

groupings = {}
for i in set(kmeans.labels_):
    groupings[i] = []

for i, label in enumerate(kmeans.labels_):
    letter_raw = pliny_model.docvecs.index_to_doctag(i)
    book, letter = letter_raw.split('_')
    letter = letter.strip('.tx')
    groupings[label].append('%s.%s' % (book, letter))

with open('kmeans_groupings.pickle', 'wb') as fp:
    pickle.dump(groupings, fp)