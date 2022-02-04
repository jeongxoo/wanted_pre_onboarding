
import re
import time
import os
import math
import numpy as np
from functools import reduce

class Tokenizer():
    def __init__(self):
        self.word_dict = {'oov': 0}
        self.fit_checker = False

    def preprocessing(self, sequences):
#         return [list(map(lambda x: x.lower, list(filter(str.isalnum, s)))) for s in sequences]
        return [list(map(lambda x: re.sub(r"[^a-zA-Z0-9]", "", x).lower(), s.split())) for s in sequences]
  
    def fit(self, sequences):
        self.fit_checker = False
        tokens = self.preprocessing(sequences)
        tokens = set(reduce(lambda x, y: x + y, tokens))
        self.word_dict = dict(self.word_dict, 
                              **{v:i+1 for i,v in enumerate(tokens)}) 
        self.fit_checker = True
    
    def transform(self, sequences):
        result = []
        tokens = self.preprocessing(sequences)
        if self.fit_checker:
            result = [list(map(lambda x: self.word_dict[x if x in self.word_dict.keys() else "oov"], t)) 
                      for t in tokens]
            return result
        else:
            raise Exception("Tokenizer instance is not fitted yet.")
      
    def fit_transform(self, sequences):
        self.fit(sequences)
        result = self.transform(sequences)
        return result
    

class TfidfVectorizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.fit_checker = False
        self.idf_matrix = []
        self.tfidf_matrix = []
  
    def fit(self, sequences):
        tokenized = self.tokenizer.fit_transform(sequences)
        
        number_token = set(reduce(lambda x, y: x + y, tokenized))
        token_count = {nt:sum([1 if nt in t else 0 for t in tokenized]) for nt in number_token}
        df = [list(map(lambda x: token_count[x], t)) for t in tokenized]

        n = len(sequences)
        self.idf_matrix = [list(map(lambda x: math.log(n / (1 + x)), d)) for d in df]
        
        self.fit_checker = True
    
    def transform(self, sequences):
        if self.fit_checker:
            tokenized = self.tokenizer.transform(sequences)
            self.tfidf_matrix = [np.array(list(map(lambda x: t.count(x), t))) * np.array(self.idf_matrix[i]) for i, t in enumerate(tokenized)]
            return self.tfidf_matrix
        else:
            raise Exception("TfidfVectorizer instance is not fitted yet.")

  
    def fit_transform(self, sequences):
        self.fit(sequences)
        return self.transform(sequences)    
