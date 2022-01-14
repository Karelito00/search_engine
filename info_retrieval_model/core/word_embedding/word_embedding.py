import numpy as np
from scipy import spatial
from sklearn.neighbors import KDTree

VECTOR_DIMENSION = 300

class WordEmbedding:
    def __init__(self, root_path = "."):
        self.embed_dict = {}
        self.vector_dimension = VECTOR_DIMENSION
        with open(f"{root_path}/glove6B/glove.6B.{VECTOR_DIMENSION}d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], 'float32')
                self.embed_dict[word] = vector
        self.kd_tree = KDTree(list(self.embed_dict.values()))
        self.keys = list(self.embed_dict.keys())

    def find_similar_word(self, embedes):
        nearest = sorted(self.embed_dict.keys(), key=lambda word: spatial.distance.euclidean(self.embed_dict[word], embedes))
        return nearest[:5]

    def find_similar_word_kdtree(self, embedes, count = 1):
        _, nearests = self.kd_tree.query([embedes], k = count)
        return list(map(lambda index: self.keys[index], nearests[0]))
