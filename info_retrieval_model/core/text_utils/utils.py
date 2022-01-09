import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

class TextPreprocessingTools:

    def __init__(self):
        self.contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not","can't": "can not","can't've": "cannot have",
        "'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have",
        "didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have",
        "hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
        "he'll've": "he will have","how'd": "how did","how'd'y": "how do you","how'll": "how will","i'd": "i would",
        "i'd've": "i would have","i'll": "i will","i'll've": "i will have","i'm": "i am","i've": "i have",
        "isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will","it'll've": "it will have",
        "let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not",
        "mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have",
        "needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
        "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
        "shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will",
        "she'll've": "she will have","should've": "should have","shouldn't": "should not",
        "shouldn't've": "should not have","so've": "so have","that'd": "that would","that'd've": "that would have",
        "there'd": "there would","there'd've": "there would have",
        "they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have",
        "they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we would",
        "we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
        "weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
        "what've": "what have","when've": "when have","where'd": "where did",
        "where've": "where have","who'll": "who will","who'll've": "who will have","who've": "who have",
        "why've": "why have","will've": "will have","won't": "will not","won't've": "will not have",
        "would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all",
        "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
        "you'd": "you would","you'd've": "you would have","you'll": "you will","you'll've": "you will have",
        "you're": "you are","you've": "you have"}

    # Expand Contractions
    # Contraction is the shortened form of a word like don’t stands
    # for do not, aren’t stands for are not. Like this, we need to expand
    # this contraction in the text data for better analysis. you can
    # easily get the dictionary of contractions on google or create
    # your own and use the re module to map the contractions.
    # Dictionary of english Contractions
    def expand_contractions(self, text):
        # Regular expression for finding contractions
        contractions_re = re.compile('(%s)' % '|'.join(self.contractions_dict.keys()))
        def replace(match):
            return self.contractions_dict[match.group(0)]
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


