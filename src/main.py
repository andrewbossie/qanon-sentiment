# Title: QAnon Drop Sentiment Analysis
# Author: Andrew Bossie
# Description: Main entry point for sentiment analysis on Q-Anon drops

import sys
import os
import datetime
import math

from Importer import Importer
from Vectorizer import Vectorizer
from Classifier import Classifier

import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

if len(sys.argv) < 2:
    sys.exit("Please provide a filename!")

posts_filename = sys.argv[1]

# saved w2v model filename
if len(sys.argv) == 3:
    if sys.argv[2] != "none":   
        model_filename = sys.argv[2]
    else:
        model_filename = None
else:
    model_filename = None

# saved k_means model filename
if len(sys.argv) == 4:
    if sys.argv[3] != "none": 
        k_filename = sys.argv[3]
    else:
        k_filename = None
else:
    k_filename = None
    
# import data
imported_data = Importer.import_data(posts_filename)
corpus_df = Importer.generate_corpus(imported_data)
corpus_list = corpus_df['text'].tolist()

# word2vec
trained_vec = Vectorizer.vectorize(corpus_list, model_filename)

# word2vec into kmeans
trained_k_means = Classifier.k_means(trained_vec.wv.vectors, k_filename)

# print(trained_vec.wv.similar_by_vector(trained_k_means.cluster_centers_[0], topn=10, restrict_vocab=None))
# print("-----------------------------------------------------")
# print(trained_vec.wv.similar_by_vector(trained_k_means.cluster_centers_[1], topn=10, restrict_vocab=None))

original_stdout = sys.stdout

# with open('../data/trained_0.txt', 'w') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(trained_vec.wv.similar_by_vector(trained_k_means.cluster_centers_[0], topn=400, restrict_vocab=None))

# with open('../data/trained_1.txt', 'w') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(trained_vec.wv.similar_by_vector(trained_k_means.cluster_centers_[1], topn=400, restrict_vocab=None))
#     sys.stdout = original_stdout

# with open('../data/trained_2.txt', 'w') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(trained_vec.wv.similar_by_vector(trained_k_means.cluster_centers_[2], topn=50, restrict_vocab=None))
#     sys.stdout = original_stdout
    
# Which clusters are positive and negative
positive_center = trained_k_means.cluster_centers_[1]
negative_center = trained_k_means.cluster_centers_[0]

# Weight words in clusters based on distance to center and populate pandas dataframe
print("Generating weights...")
sentiment_corpus = pd.DataFrame(trained_vec.wv.index_to_key)
sentiment_corpus.columns = ['key']
sentiment_corpus['vectors'] = sentiment_corpus['key'].apply(lambda x: trained_vec.wv[f'{x}'])
sentiment_corpus['cluster'] = sentiment_corpus['vectors'].apply(lambda x: trained_k_means.predict([np.array(x)]))
sentiment_corpus.cluster = sentiment_corpus['cluster'].apply(lambda x: x[0])
sentiment_corpus['cluster_signed'] = [-1 if i==0 else 1 for i in sentiment_corpus['cluster']]

# Lets determine HOW positive or negative each word is...
sentiment_corpus['min_value'] = sentiment_corpus.apply(lambda x: 1/(trained_k_means.transform([x.vectors]).min()), axis=1)
sentiment_corpus['distance'] = sentiment_corpus['min_value'] * sentiment_corpus['cluster_signed']

sentiment_corpus.to_csv("../data/word_sentiment.csv", columns = ["key", "cluster", "cluster_signed", "min_value", "distance"], index=False)
print("Done.")

# Print split words to file
# sentiment_corpus[['key', 'distance']].loc[sentiment_corpus['cluster_signed'] == 1].to_csv('../data/words_1.txt', index=False)
# sentiment_corpus[['key', 'distance']].loc[sentiment_corpus['cluster_signed'] == -1].to_csv('../data/words_-1.txt', index=False)

# exit()

# Sanity Check
negative_word_count = len(sentiment_corpus[sentiment_corpus['cluster_signed'] == -1])
positive_word_count = len(sentiment_corpus[sentiment_corpus['cluster_signed'] == 1])

print(f"Number of positive words: {positive_word_count}")
print(f"Number of negative words: {negative_word_count}")

# Replace cleaned df words with scores
score_df = pd.DataFrame()

# Remove k_means results on words lower than w2v's min_count threshold.
print("Generating Sentence Sentiment Dataframe...")
score_df['split'] = corpus_df['text'].map(lambda x: [sentiment_corpus[sentiment_corpus['key'] == j]['distance'].values for i, j in enumerate(x)])
score_df['split'] = score_df['split'].map(lambda x: [j for j in x if j.size > 0])
score_df['split'] = score_df['split'].map(lambda x: [j[0] for i, j in enumerate(x)])
score_df['line_score'] = score_df['split'].map(lambda x: sum(x))

score_df['line_score'].to_csv("../data/score_dataframe.csv")
print("Done.")

# Total 
positive_drops_count = len(score_df[score_df['line_score'] > 0])
negative_drops_count = len(score_df[score_df['line_score'] < 0])

print(f"Number of positive drops: {positive_drops_count}")
print(f"Number of negative drops: {negative_drops_count}")

# Group positive and negative drops
positive_sentiment_drops_df = pd.DataFrame()
negative_sentiment_drops_df = pd.DataFrame()

positive_sentiment_drops_df['positive_drops'] = score_df[score_df['line_score'] > 0]['line_score']
negative_sentiment_drops_df['negative_drops'] = score_df[score_df['line_score'] < 0]['line_score']

# Drop Sentiment Raw Bar Chart
bar_chart = plt.figure(2)
plt.bar(["Positive", "Negative"], [positive_drops_count, negative_drops_count], width=0.5)
plt.xlabel("Sentiment")
plt.ylabel("No. of drops")
plt.savefig("../plots/drop_bar_chart.png")

# Histogram with mean and std
positive_mean = positive_sentiment_drops_df['positive_drops'].mean()
negative_mean = negative_sentiment_drops_df['negative_drops'].mean()

positive_std = positive_sentiment_drops_df['positive_drops'].std()
negative_std = negative_sentiment_drops_df['negative_drops'].std()

print(f"Positive collection mean {positive_mean}, deviation: {positive_std}")
print(f"Negative collection mean {negative_mean}, deviation: {negative_std}")

positive_sentiment_hist = plt.figure(3)
plt.hist(positive_sentiment_drops_df['positive_drops'], 100, range=[0, 10], density = True, histtype ='bar')
plt.xlabel("Occurances")
plt.ylabel("Weighted Distance")
plt.savefig("../plots/positive_sentiment_hist.png")
# exit()

negative_sentiment_hist = plt.figure(4)
plt.hist(negative_sentiment_drops_df['negative_drops'], 100, range=[-17, 0],density = True, histtype ='bar')
plt.xlabel("Occurances")
plt.ylabel("Weighted Distance")
plt.savefig("../plots/negative_sentiment_hist.png")

sys.exit("Bye.")