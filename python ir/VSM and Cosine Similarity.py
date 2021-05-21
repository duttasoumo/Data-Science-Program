#!/usr/bin/env python
# coding: utf-8

# In[67]:


from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
import nltk

#Reading Files into Corpus
dir = 'docs\\ex2\\'
filelist = os.listdir(dir)
N = len(filelist)
corpus = []
for fn in filelist:
    f = open(dir+fn, 'r')
    text = f.read().lower()
    corpus.append(text)
print(corpus)

#Term-Doc Freq Matrix
vectorizer  = CountVectorizer()
X = vectorizer.fit_transform(corpus)
X = X.toarray().T
print(X)

vocab = vectorizer.get_feature_names()
print(vocab)

#IDF Computation
df = np.reshape(np.count_nonzero(X, axis=1), (len(vocab), 1))
print(df)
idf = np.log10(N/df)
print(idf)

#Weight Matrix
wm = X * idf
print(wm)

#Preparing the query
query = ['gold silver truck']
q = vectorizer.transform(query).toarray()
q = q * idf.T
print("Query:", q)


#Cosine Similarity
q_dot_d = np.dot(q, wm)
print(q_dot_d)
d_norms = np.sqrt(np.sum(wm**2, axis=0))
q_norm = np.sqrt(np.sum(q**2))
print(d_norms)
print(q_norm)

cos_sim = q_dot_d/(d_norms * q_norm) 
print(cos_sim)


# In[74]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
X = vectorizer.fit_transform(corpus)
X = tfidf.fit_transform(X)
print(X.toarray())

