"""
This py script is use for preprocess tweets
"""

import re
import csv
import pandas as pd
from nltk import tokenize

class Preprocess():
    """
	class
	"""
    def __init__(self, input_str, filename, max_length_dictionary=30000, max_length_tweet=20):
        """
    	initiate instance
    	"""
        self.filename = filename
        self.max_length_dictionary = max_length_dictionary
        self.max_length_tweet = max_length_tweet
        self.input_str = input_str
        self.cleantext = self.clean_text()
        self.tokentext = self.tokenize_text()
        self.indextext = self.replace_token_with_index(self.tokentext)
        self.pad_sequence(self.indextext)
        print(self.indextext)

    def clean_text(self):
        """
        This function is used for clean raw strings
        """
        text = re.sub(r'http://\S+.\S+', '', self.input_str)
        text = re.sub(r'@[\S]*', '', text)
        text = text.lower()
        return text

    def tokenize_text(self):
        """
        This function is used for tokenizing text
	    """
        tknzr = tokenize.TweetTokenizer(reduce_len=True)
        return tknzr.tokenize(self.cleantext)

    def replace_token_with_index(self, str_):
        """
	    This function is used for replacing tokens with indices
	    """
        dic = pd.read_csv(self.filename, sep=' ', quoting=csv.QUOTE_NONE, header=None,\
            nrows=self.max_length_dictionary)
        text = pd.DataFrame(dic[0])
        text['index'] = text.index
        dict_ = text.set_index(0).T.to_dict('int')
        newdic_ = dict_['index']
        tokened = [newdic_[word] for word in str_]
        return [tokened]

    def pad_sequence(self, text):
        """
	    This function is used for create pad sequence
	    """
        for element in text:
            if len(element) > self.max_length_tweet:
                element = element[:self.max_length_tweet]
            else:
                element = element.extend([0] * (self.max_length_tweet - len(element)))
        return text
