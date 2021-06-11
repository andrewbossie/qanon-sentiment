# Title: Importer 
# Author: Andrew Bossie
# Description: Helper functions for importing and preprocessing data

import json
import pandas as pd
import re

from gensim.parsing.preprocessing import remove_stopwords

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Importer():

    def __init__(self):
        pass

    # Import JSON and extract text
    @staticmethod
    def import_data(filename):
        print(f"Importing Data from file: {filename}...")
        with open(filename) as f:
            imported = json.load(f)
        print("Done.")
        return imported

    # Combine text into dataframe
    @staticmethod
    def generate_corpus(imported_text):
        print("Combining into dataframe...")
        corpus = pd.DataFrame(columns=['text'])
        for i in imported_text['posts']:
            if 'text' in i:
                corpus.loc[len(corpus.index)] = remove_stopwords(str(i['text']).replace("\n", "").replace("\r", "").replace("\t", "").replace('Q', '').lower())

        # Drop test posts
        corpus['text'] = corpus[~corpus['text'].str.contains("test")]
        corpus.dropna(inplace=True)

        corpus['text'] = corpus[~corpus['text'].str.contains("_")]
        corpus.dropna(inplace=True)

        # Remove Links
        corpus['text'] = corpus['text'].str.replace(r'https\S+', '', regex=True)
        corpus['text'] = corpus['text'].str.replace(r'http\S+', '', regex=True)

        # Remove Random Chars
        corpus['text'] = corpus['text'].str.replace(r'\+\+\S+', '', regex=True)
        corpus['text'] = corpus['text'].str.replace(r'\"\"\"\S+', '', regex=True)
        corpus['text'] = corpus['text'].str.replace(':', '')
        corpus['text'] = corpus['text'].str.replace('[', '')
        corpus['text'] = corpus['text'].str.replace(']', '')
        corpus['text'] = corpus['text'].str.replace('.', ' ')
        corpus['text'] = corpus['text'].str.replace(',', ' ')
        corpus['text'] = corpus['text'].str.replace('?', ' ')
        corpus['text'] = corpus['text'].str.replace('!', ' ')
        corpus['text'] = corpus['text'].str.replace('@', '')
        corpus['text'] = corpus['text'].str.replace('"', '')
        corpus['text'] = corpus['text'].str.replace("'", '')
        corpus['text'] = corpus['text'].str.replace("-", '')
        corpus['text'] = corpus['text'].str.replace(">", '')
        corpus['text'] = corpus['text'].str.replace("<", '')
        corpus['text'] = corpus['text'].str.replace("#", ' #')

        # singular 'q' signatures
        # corpus['text'] = corpus['text'].str.replace('Q', '') # TOGGLE ON / OFF

        corpus['text'] = corpus['text'].str.replace(r'\d{6,10}', '', regex=True)

        # strip left and right whitespace
        corpus['text'] = corpus['text'].str.lstrip()
        corpus['text'] = corpus['text'].str.rstrip()

        # Drop post replies
        corpus['text'] = corpus[~corpus['text'].str.contains(">>")]
        corpus.dropna(inplace=True)

        # Remove small words
        corpus['text'] = corpus['text'].apply(lambda x: ' '.join([w for w in str(x).split() if len(w)>3]))
        
        # Drop rows with len < 6
        corpus = corpus[(corpus.text.str.len() > 7)]

        corpus['text'] = corpus['text'].apply(lambda x: str(x).split())
        
        # test
        corpus.to_csv("../data/combined_dataframe_clean.csv", index=True)
  
        print("Done.")
        print(f'Corpus Shape: {corpus.shape}')
        return corpus

    # From extracted text get corpus
    @staticmethod
    def generate_unique(corpus):
        print("Generating corpus...")
        
        # for the dataframe create a set of unique words
        unique = list(corpus['text'].str.split(' ', expand=True).stack().unique())

        unique_df = pd.DataFrame(list(sorted(unique)))
        unique_df[0] = unique_df[0].apply(lambda x : [x])
        unique_df[0] = unique_df[(unique_df[0].str.len() > 3)]
        unique_df.to_csv("../data/corpus.csv", index=True)
        print("Done")
        print(f"Unique Shape: {unique_df.shape}")
        return unique_df[4:]