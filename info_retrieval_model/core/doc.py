from .vector import Vector
from .text_utils import TextPreprocessingTools
A = 0.4

class Doc:
    def __init__(self, text):
        tpt = TextPreprocessingTools()
        self.text = text
        self.terms = tpt.run_pipeline(text).split(" ")
        self.build_freq()
        self.calculate_tfi()

    def __lt__(self, other):
        return len(self.terms) < len(other.terms)

    def build_freq(self):
        freq = {}
        for term in self.terms:
            if(freq.__contains__(term) == False):
                freq[term] = 1
            else:
                freq[term] += 1
        self.freq = Vector(freq)

    def calculate_tfi(self):
        tfi = {}
        max_freq = max(self.freq.values())

        for term in self.freq:
            tfi[term] = self.freq.vector[term] / max_freq
        self.tfi = Vector(tfi)

    def calculate_wi(self, idf, is_query = False):
        wi = {}
        for term in self.tfi:
            if(idf.__contains__(term) == False):
                continue
            if(is_query):
                wi[term] =  (A + ((1 - A) * self.tfi.vector[term])) * idf.vector[term]
            else:
                wi[term] = self.tfi.vector[term] * idf.vector[term]
        self.wi = Vector(wi, True)
