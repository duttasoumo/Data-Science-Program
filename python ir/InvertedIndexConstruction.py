#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

dirs = 'docs\\ex2\\'
filelist = os.listdir(dirs)
print("File List:n", filelist)

corpus = []
for fn in filelist:
    f = open(dirs + fn, 'r')
    text = f.read()
    corpus.append(text)

V = CountVectorizer()
tdm = V.fit_transform(corpus).toarray().T
vocab = V.get_feature_names()

print("TDM:\n", tdm)
print("Vocabulary:\n", vocab)

iid = dict()
for (i, word) in enumerate(vocab):
    iid[word] = list(np.where(tdm[i] > 0)[0])

print("IID:")
for key in iid.keys():
    print(key,": ", iid[key])

