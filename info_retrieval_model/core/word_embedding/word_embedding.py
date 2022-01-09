import numpy as np
from scipy import spatial

VECTOR_DIMENSION = 100

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

    def find_similar_word(self, embedes):
        nearest = sorted(self.embed_dict.keys(), key=lambda word: spatial.distance.euclidean(self.embed_dict[word], embedes))
        return nearest[:5]
