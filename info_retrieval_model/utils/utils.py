import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

class TextPreprocessingTools:
    # Expand Contractions
    # Contraction is the shortened form of a word like don’t stands
    # for do not, aren’t stands for are not. Like this, we need to expand
    # this contraction in the text data for better analysis. you can
    # easily get the dictionary of contractions on google or create
    # your own and use the re module to map the contractions.
    def expand_contractions(self, text):
        contractions_dict = {"ain't": "are not","'s":" is","aren't": "are not", "'re": "are", "'d": "had", "'ll": "will", "I'm": "I am", "let's": "let us"}
        # Regular expression for finding contractions
        contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, text)

    # For avoid that words like Ball and ball be treated differently
    def to_lower(self, text):
        return text.lower()

    def remove_punctuations(self, text):
        return "".join(['' if x in string.punctuation else x for x in text])

    def remove_words_with_digits(self, text):
        text = text.split(" ")
        return " ".join(filter(lambda x: re.search('[0-9]', x) is None, text))

    def remove_stopwords(self, text):
        text = text.split(" ")
        return " ".join(filter(lambda x: x not in stop_words, text))

    def stem_words(self, text):
        text = text.split(" ")
        return " ".join([stemmer.stem(x) for x in text])

    def lemmtize_words(self, text):
        text = text.split(" ")
        return " ".join([lemmatizer.lemmatize(x) for x in text])

    def run_pipeline(self, text):
        text = self.expand_contractions(text)
        text = self.to_lower(text)
        text = self.remove_punctuations(text)
        text = self.remove_words_with_digits(text)
        text = self.remove_stopwords(text)
        # text = self.stem_words(text)
        text = self.lemmtize_words(text)
        return text


