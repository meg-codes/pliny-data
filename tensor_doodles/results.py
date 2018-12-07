import numpy as np
import pickle
import sys

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index


word2int = pickle.load(open('word2int.pickle', 'rb'))
int2word = pickle.load(open('int2word.pickle', 'rb'))
vectors = pickle.load(open('vectors.pickle', 'rb'))
print(word2int.keys())
print(int2word[find_closest(word2int[sys.argv[1]], vectors)])
