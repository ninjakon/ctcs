import random
import string
import nltk
from nltk.tokenize import word_tokenize

from .base_processor import BaseProcessor

nltk.download("punkt_tab")


class RandomRemovalProcessor(BaseProcessor):  
    """
    Randomly remove X% of tokens from a prompt. 

    Procedure:
    ---------
    - Randomly generate indices to remove 
    - Build new list with tokens that are not to be removed
    - Concatenate list to string and return

    Returns:
    -------
    str
        Prompt with X% of tokens removed
    """

    def __init__(self, percentage=20):
        BaseProcessor.__init__(self)

        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0% and 100%")
        self.percentage = percentage

    def tokenize(self, prompt):

        prompt_no_punctuation = prompt.translate(str.maketrans("", "", string.punctuation))
        word_tokens = word_tokenize(prompt_no_punctuation)
    
        indices_to_remove = random.sample(range(len(word_tokens)), len(word_tokens)*self.percentage//100)
        remaining_tokens = [token for i, token in enumerate(word_tokens) if i not in indices_to_remove]

        return ' '.join([str(s) for s in remaining_tokens])        
