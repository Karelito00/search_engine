from .vectorial_model import VectorialModel
from .file_tools import FileTools

def initialize(root_path = "."):
    # Provide the collection path
    # news-group collection
    #ft = FileTools(f"{root_path}/collections/news-group")

    # docs-lisa collection
    #ft = FileTools(f"{root_path}/collections/docs-lisa")

    # npl collection
    ft = FileTools(f"{root_path}/collections/npl")

    docs = ft.get_documents()
    return  VectorialModel(map(lambda doc: ft.read_document(doc), docs))
