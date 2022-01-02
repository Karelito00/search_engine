from .vectorial_model import VectorialModel
from .file_tools import FileTools

def initialize():
    ft = FileTools("/media/karelito00/Data/University/3er año/2do semestre/SRI/Proyecto Final/Search Engine/info_retrieval_model/core/collections")

    docs = ft.get_documents()
    return  VectorialModel(map(lambda doc: ft.read_document(doc), docs))
