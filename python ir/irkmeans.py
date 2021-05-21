from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

from sklearn.cluster import KMeans;
import numpy as np

###############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
#categories = None

print ("Loading 20 newsgroups dataset for categories:")
print (categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print ("%d documents" % len(dataset.data))
print ("%d categories" % len(dataset.target_names))
print

labels = dataset.target
true_k = np.unique(labels).shape[0]

print ("Extracting features from the training dataset using a sparse vectorizer")
vectorizer = CountVectorizer(max_df=0.5, max_features=10000, stop_words='english')
X = vectorizer.fit_transform(dataset.data)

print ("n_samples: %d, n_features: %d" % X.shape)
print


###############################################################################
# Do the actual clustering

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

#print ("Clustering sparse data with %s" % km)
km.fit(X)


print ("Purity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print ("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print ("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print ("Adjusted Rand-Index: %.3f" % \
    metrics.adjusted_rand_score(labels, km.labels_))
print ("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(
    X, labels, sample_size=1000))

print
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
      print("Cluster %d:" % i, end='')
      for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
      print()