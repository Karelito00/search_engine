from os import walk

class FileTools:
    def __init__(self, path):
        self.path = path

    def get_documents(self, path = None, count = 1000000):
        if(path is None):
            path = self.path

        text_arr = []
        for dirpath, _, filenames in walk(path):
            text_arr.extend(map(lambda filename: dirpath + "/" + filename, filenames))

        return text_arr

    def read_document(self, path):
        f = open(path, "rb")
        return str(f.read())
