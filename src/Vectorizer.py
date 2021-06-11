# Title: Vectorizer 
# Author: Andrew Bossie
# Description: Wrapper class and helper functions for word2vec integration

from gensim.models import Word2Vec
from gensim.test.utils import datapath

# import logging

# logging.basicConfig(format=’%(asctime)s : %(levelname)s : %(message)s’, level=logging.INFO)

class Vectorizer():
    
    def __init__(self):
        pass

    @staticmethod
    def vectorize(corpus_list, file_name=None):
        
        # New Model
        if not file_name:
            v_model = Word2Vec(corpus_list, min_count=3, vector_size=1000, workers=4, window=5, epochs=40, sg=1) # 2 / 1000 / 5 / 500 / sg
            print(f"Created tensor with shape: {v_model.wv.vectors.shape}")
            print("Saving model...")
            v_model.save("../saved_models/v_model.w2v")
        else:
            # Load Saved Model
            v_model = Word2Vec.load(file_name)
            print("Done.")

        # print(v_model.wv.most_similar('president'))
        # print("----------------")
        # print(v_model.wv.most_similar('attack'))
        # print("----------------")
        # print(v_model.wv.most_similar('corruption'))
        # print("----------------")
        # print(v_model.wv.most_similar('investigation'))
        # print("----------------")
        
        return v_model

        