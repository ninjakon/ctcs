import string
import nltk
from nltk.tokenize import word_tokenize

from .base_processor import BaseProcessor

nltk.download("punkt_tab")


class ShortWordRemovalProcessor(BaseProcessor):  
    """
    A class to remove short words from a string. 

    Attributes:
    ----------
    min_length : int
        The minimum length a token must have to not be removed. 

    Methods:
    -------
    tokenize(prompt)
        Remove all words that are shorter than min_length
    """

    def __init__(self, min_length : int = 4):
        BaseProcessor.__init__(self)

        if not 0 <= min_length <= 50:
            raise ValueError("Min length must be a natural number and be at maximum 50.")
        
        self.min_length = min_length

    def tokenize(self, prompt):
        """
        Remove all tokens that are shorter than min_length. 

        Parameters:
        ----------
        prompt : str
            The prompt from which all short words get removed. 

        Returns:
        str
            Prompt with all short words removed. 
        """

        prompt_no_punctuation = prompt.translate(str.maketrans("", "", string.punctuation))
        word_tokens = word_tokenize(prompt_no_punctuation)

        return ' '.join(token for token in word_tokens if len(token) >= self.min_length)
