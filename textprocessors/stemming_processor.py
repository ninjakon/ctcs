import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from .base_processor import BaseProcessor

nltk.download("punkt_tab")


class StemmingProcessor(BaseProcessor):

    porter_stemmer = None

    def __init__(self):
        BaseProcessor.__init__(self)
        self.porter_stemmer = PorterStemmer()

    def tokenize(self, prompt):
        if self.porter_stemmer is None:
            self.porter_stemmer = PorterStemmer()

        prompt_no_punctuation = prompt.translate(str.maketrans("", "", string.punctuation))
        word_tokens = word_tokenize(prompt_no_punctuation)
        stemmed_words = []
        for word in word_tokens:
            stemmed_word = self.porter_stemmer.stem(word)
            stemmed_words.append(stemmed_word)
        return ' '.join([str(s) for s in stemmed_words])
