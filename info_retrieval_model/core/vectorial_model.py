from .text_utils import TextPreprocessingTools
import functools
import math
A = 0.4

class Doc:
    def __init__(self, text):
        tpt = TextPreprocessingTools()
        self.terms = tpt.run_pipeline(text).split(" ")
        self.build_vector()
        self.calculate_tfi()

    def build_vector(self):
        self.vector = {}
        for term in self.terms:
            if(self.vector.__contains__(term) == False):
                self.vector[term] = 1
            else:
                self.vector[term] += 1

    def calculate_tfi(self):
        self.tfi = {}
        max_freq = max(self.vector.values())

        for term in self.vector:
            self.tfi[term] = self.vector[term] / max_freq

    def calculate_wi(self, idf, is_query = False):
        self.wi = {}
        for term in self.tfi:
            if(is_query):
                self.wi[term] =  (A + ((1 - A) * self.tfi[term])) * idf[term]
            else:
                self.wi[term] = self.tfi[term] * idf[term]

    def calculate_norm(self):
        self.norm = math.sqrt(functools.reduce(lambda a, b: a + (b * b), self.wi.values(), 0))

class VectorialModel:
    def __init__(self, docs):
        self.term_universe = {}
        self.docs = []
        for doc in docs:
            doc = Doc(doc)
            for term in doc.vector:
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
            doc.calculate_norm()

    # idf = log( N / ni ) where:
    # N -> total documents
    # ni -> total documents where the term ti appears
    def calculate_idf(self):
        self.idf = {}
        for term in self.term_universe:
            self.idf[term] = math.log(len(self.docs) / self.term_universe[term], 10)

    def correlation(self, doc_a, doc_b):
        sum_t = 0
        max_vector = doc_a.wi if len(doc_a.wi) >= len(doc_b.wi) else doc_b.wi
        min_vector = doc_a.wi if len(doc_a.wi) < len(doc_b.wi) else doc_b.wi
        for term in min_vector:
            if(max_vector.__contains__(term)):
                sum_t += min_vector[term] * max_vector[term]

        if(doc_a.norm * doc_b.norm < 0.0000001):
            return 100000
        return sum_t / (doc_a.norm * doc_b.norm)

    def query(self, text):
        query_doc = Doc(text)
        query_doc.calculate_wi(self.idf, True)
        query_doc.calculate_norm()

        ranking = []
        for doc in self.docs:
            rank = self.correlation(doc, query_doc)
            ranking.append([rank, doc])

        return sorted(ranking, reverse=True)


