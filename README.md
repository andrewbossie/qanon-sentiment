# Q-Anon Unsupervised Sentiment Analysis

## Motivation
Learning a new concept in any field can be a daunting task. A typical approach is to find an application that is relevant or at least interesting to you, the learner. In teaching myself unsupervised machine learning, I was intrigued by QAnon drops; anonymous internet posts that have been shaking up the socio-political world. Are there interpretations that ML can make on QAnon drops? Do these drops lean towards a specific sentiment?

## The Project
With the preceding motivation I set out to create a python-based unsupervised approach to describing the sentiment of QAnon drops. In the following write up we will use word2vec and K-means cluster, to implement a quick and dirty method of extracting positive or negative sentiment from these posts.

## Tech and the Code
Python 3+, Pandas, Numpy, Scikit-Learn, Gensim and Matplotlib

# Run
`python(3) main.py <path-to-datafile> <path-to-w2vfile> <path-to-kmeansfile>`

* Args 2 and 3 are optional