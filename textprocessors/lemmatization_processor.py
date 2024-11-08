import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .base_processor import BaseProcessor

nltk.download("wordnet")
nltk.download("omw-1.4")


class LemmatizationProcessor(BaseProcessor):

    word_net_lemmatizer = None

    def __init__(self):
        BaseProcessor.__init__(self)
        self.word_net_lemmatizer = WordNetLemmatizer()

    def tokenize(self, prompt):
        if self.word_net_lemmatizer is None:
            self.word_net_lemmatizer = WordNetLemmatizer()

        prompt_no_punctuation = prompt.translate(str.maketrans("", "", string.punctuation))
        word_tokens = word_tokenize(prompt_no_punctuation)
        stemmed_words = []
        for word in word_tokens:
            stemmed_word = self.word_net_lemmatizer.lemmatize(word, pos="v")
            stemmed_words.append(stemmed_word)
        return ' '.join([str(s) for s in stemmed_words])
