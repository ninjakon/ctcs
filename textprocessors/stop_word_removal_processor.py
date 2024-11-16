import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Literal

# User Imports
from .base_processor import BaseProcessor

nltk.download("punkt_tab")
nltk.download("stopwords")


class StopWordRemovalProcessor(BaseProcessor):  
    """
    A class to remove all stop words from a prompt (string). 

    Attributes:
    ----------
    language : str
        The language from which the stop words should be removed from. Options include 'english' and 'german'.

    Methods:
    -------
    tokenize(prompt):
        Remove all stop words from a prompt.
    """

    def __init__(self, language: Literal['english', 'german'] = 'english'):
        BaseProcessor.__init__(self)
        self.language = language

    def tokenize(self, prompt) -> str:
        """
        Remove all stop words from a prompt. 

        Parameters:
        ----------
        prompt : str
            The prompt from which the stop words get removed.
        
        Returns:
        -------
        str
            Prompt with all stop words of the specified language removed. 
        """

        prompt_no_punctuation = prompt.translate(str.maketrans("", "", string.punctuation))
        word_tokens = word_tokenize(prompt_no_punctuation)
        stop_words = set(stopwords.words(self.language))

        filtered_tokens = [token for token in word_tokens if token.lower() not in stop_words]
        return ' '.join(filtered_tokens)