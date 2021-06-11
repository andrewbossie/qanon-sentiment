import numpy as np
import matplotlib.pyplot as plt
import pickle

import sklearn
from sklearn.cluster import KMeans

class Classifier:

    # K-Means Clustering
    @staticmethod
    def k_means(corpus_vec, file_name=None):
        
        if not file_name:
            print("Generating new k_means model...")
            k_means = KMeans(n_clusters=2, max_iter=10, random_state=True, n_init=50).fit(corpus_vec.astype('double'))
            print("Saving Model...")
            with open("../saved_models/k_means.pkl", "wb") as f:
                pickle.dump(k_means, f)
        else:
            print("Loading previous k_means model...")
            with open(file_name, "rb") as f:
                k_means = pickle.load(f)

        print("Done.")

        y_kmeans = k_means.predict(corpus_vec.astype('double'))
        k_means_plot = plt.figure(1)
        plt.scatter(corpus_vec[:, 0], corpus_vec[:, 1], c=y_kmeans, alpha=0.3, s=50, cmap='viridis')
        centers = k_means.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.8)
        plt.savefig('../plots/k_means.png')
        
        return k_means