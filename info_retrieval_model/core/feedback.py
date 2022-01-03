from.vector import Vector

class Node:
    def __init__(self):
        self.mapp = {}
        self.relevants = set()
        self.no_relevants = set()

class Feedback:
    def __init__(self):
        self.first_node = Node()

    def set_feedback(self, query, feedback_type, doc_index):
        node = self.first_node
        for term in query.terms:
            if(node.mapp.__contains__(term) == False):
                node.mapp[term] = Node()
            node = node.mapp[term]

        if(feedback_type == 1):
            if(node.no_relevants.__contains__(doc_index)):
                node.no_relevants.remove(doc_index)
            node.relevants.add(doc_index)
        else:
            if(node.relevants.__contains__(doc_index)):
                node.relevants.remove(doc_index)
            node.no_relevants.add(doc_index)

    def get_feedback(self, query):
        node = self.first_node
        for term in query.terms:
            if(node.mapp.__contains__(term) == False):
                return None
            node = node.mapp[term]

        return [node.relevants, node.no_relevants]
