import functools
import math
import numpy as np
from scipy import spatial
from .feedback import Feedback
from .vector import Vector
from .doc import Doc
from .word_embedding import WordEmbedding
EPS = 0.0000001

class VectorialModel:
    def __init__(self, docs):
        self.term_universe = {}
        self.docs = []
        self.feedback = Feedback()
        self.word_embedding = WordEmbedding("./core/word_embedding")
        for doc in docs:
            doc = Doc(doc)
            for term in doc.freq:
                if(self.term_universe.__contains__(term) == False):
                    self.term_universe[term] = 1
                else:
                    self.term_universe[term] += 1
            self.docs.append(doc)

        self.calculate_idf()
        self.calculate_weight_of_docs()

    def calculate_weight_of_docs(self):
        for doc in self.docs:
            doc.calculate_wi(self.idf)

    # idf = log( N / ni ) where:
    # N -> total documents
    # ni -> total documents where the term ti appears
    def calculate_idf(self):
        idf = {}
        for term in self.term_universe:
            idf[term] = math.log(len(self.docs) / self.term_universe[term], 10)
        self.idf = Vector(idf)

    def get_word_idf(self, word):
        if(self.idf.__contains__(word) == False):
            return 0
        return self.idf[word]

    # correlation calculated by cosine similarity
    def correlation(self, vector_a, vector_b):
        sum_t = 0
        max_vector = vector_a if len(vector_a) >= len(vector_b) else vector_b
        min_vector = vector_a if len(vector_a) < len(vector_b) else vector_b
        for term in min_vector:
            if(max_vector.__contains__(term)):
                sum_t += min_vector[term] * max_vector[term]

        if(sum_t < EPS):
            return 0
        if(vector_a.norm * vector_b.norm < EPS):
            return 100000
        return sum_t / (vector_a.norm * vector_b.norm)

    # The first n documents of the ranking are considered relevants
    def query(self, text, n = 20):
        query_doc = Doc(text)
        query_doc.add_terms(self.get_query_expansion(query_doc))
        query_doc.calculate_wi(self.idf)
        wi_query = self.get_feedback(query_doc)
        if(wi_query is None):
            wi_query = query_doc.wi

        self.last_query = query_doc

        ranking = []
        index = 0
        for doc in self.docs:
            rank = self.correlation(doc.wi, wi_query)
            ranking.append([rank, doc, index])
            index += 1

        ranking = sorted(ranking, reverse=True)

        return ranking[:min(n, len(ranking))]

    # IDF-AWE(q) = ( 1 / sum (IDF(wordi) ) ) * sum( IDF(wordi) * wordi_vector )
    # wordi is the word embedding of qi term of the query
    def get_idf_awe(self, wordi_array):
        AWE_vector = np.zeros(self.word_embedding.vector_dimension, dtype=float)
        sum_IDF = 0
        embed_dict = self.word_embedding.embed_dict
        for val in wordi_array:
            sum_IDF +=  self.get_word_idf(val[1])
            AWE_vector += embed_dict[val[0]] * self.get_word_idf(val[1])

        return AWE_vector * ( 1 / sum_IDF )

    def get_query_expansion(self, query_doc):
        wordi_array = []
        embed_dict = self.word_embedding.embed_dict
        for term in query_doc.terms:
            # word embeddings related with qi term
            if(embed_dict.__contains__(term)):
                wordi_array = wordi_array + [[self.word_embedding.find_similar_word_kdtree(embed_dict[term])[0], term]]
        idf_awe_vector = self.get_idf_awe(wordi_array)
        wordi_ranking = list(map(lambda val: [np.exp(spatial.distance.cosine(embed_dict[val[0]], idf_awe_vector)), val[0]], wordi_array))
        return list(map(lambda x: x[0], sorted(wordi_ranking, reverse=True)))

    # qm = q + b * d_r - y * d_nr
    # b = 0.75 / len(d_r)
    # y = 0.15 / len(d_nr)
    # d_r -> documents relevants
    # d_nr -> documents not relevants
    def get_feedback(self, query_doc):
        query_feedback = self.feedback.get_feedback(query_doc)
        if(query_feedback is None):
            return None
        relevants, no_relevants = query_feedback[0], query_feedback[1]
        qm = query_doc.wi

        sum_relev = functools.reduce(lambda a, b: a + self.docs[b].wi, relevants, Vector())
        sum_no_relev = functools.reduce(lambda a, b: a + self.docs[b].wi, no_relevants, Vector())

        b = 0 if len(relevants) == 0 else 0.75 / len(relevants)
        y = 0 if len(no_relevants) == 0 else 0.15 / len(no_relevants)
        qm = qm + (sum_relev * b) - (sum_no_relev * y)
        qm.calculate_norm()
        return qm

    # feedback_type is equal to -1 if is a negative feedback, otherwise
    # is equal to 1 and is a positive feedback
    def set_feedback(self, feedback_type, doc_index, query = None):
        if(query is None):
            query = self.last_query
        self.feedback.set_feedback(query, feedback_type, doc_index)
