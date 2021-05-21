#!/usr/bin/env python
# coding: utf-8

# In[30]:

import string
import nltk
from nltk.corpus import movie_reviews as dataset
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Read document list
doclist = dataset.fileids()
np.random.shuffle(doclist)
#print(doc_list)


# In[33]:


#build corpus and vocab
corpus = []
Y = []
for fn in doclist:
    corpus.append(dataset.raw(fn))
    Y.append(dataset.categories(fn))
    
#StopWords Removal
sw = stopwords.words('english') + list(string.punctuation)
all_words = dataset.words()
vocab = [x for x in all_words if x not in sw]
#Getting most frequent words
vocab_top = list(dict(nltk.FreqDist(vocab).most_common()).keys())[:3000]
print(vocab_top)
vec = CountVectorizer(vocabulary=vocab_top)
X = vec.fit_transform(corpus).toarray()
Y = np.array(Y)

# print(X[:5])
# print(Y[:5])
#print(len(vocab_top))
# print(X)

#tf-idf weights
tfidf = TfidfTransformer()
X_w = tfidf.fit_transform(X)


#Making the training and testing sets
(m, n) = X_w.shape
print(m, n)

X_train = X_w[:1800, :]
Y_train = Y[:1800, :]

X_test = X_w[1800:, :]
Y_test = Y[1800:, :]


# In[34]:


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, np.squeeze(Y_train))

Y_pred = clf.predict(X_test)


# In[35]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

correct = len(np.where(np.squeeze(Y_test)==Y_pred)[0])
print(correct/200)


# ## -----------------------------------

# In[ ]:


from nltk.corpus import stopwords

#StopWords Removal
sw = stopwords.words('english') + list(string.punctuation)
all_words = dataset.words()
vocab = [x for x in all_words if x not in sw]


# In[27]:


#Getting Most Frequent Words
#print(list(dict(nltk.FreqDist(vocab).most_common()).keys())[:10])
fd = list(dict(nltk.FreqDist(vocab).most_common()).keys())[:3000]

