#!/usr/bin/env python
# coding: utf-8

# In[25]:


from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np

#Reading Files into Corpus
dir = 'docs\\'
filelist = os.listdir(dir)
N = len(filelist)
corpus = []
for fn in filelist:
    f = open(dir+fn, 'r')
    text = f.read().lower()
    corpus.append(text)
print(corpus)


# In[31]:


vectorizer  = CountVectorizer()
X = vectorizer.fit_transform(corpus)
X = X.toarray().T
X


# In[29]:


vocab = vectorizer.get_feature_names()
vocab


# In[5]:


df = np.reshape(np.count_nonzero(X, axis=1), (len(vocab), 1))
#print(df)
idf = np.log10(N/df)
idf


# In[6]:


wm = X * idf
wm


# In[12]:


d1_norms=np.sqrt(np.sum(wm[:,0]**2, axis=0))
d2_norms=np.sqrt(np.sum(wm[:,1]**2, axis=0))
d3_norms=np.sqrt(np.sum(wm[:,2]**2, axis=0))
d4_norms=np.sqrt(np.sum(wm[:,3]**2, axis=0))
d3_norms


# In[20]:


d1_dot_d2 = np.dot(wm[:,0], wm[:,1])
d1_dot_d3 = np.dot(wm[:,0], wm[:,2])
d2_dot_d3 = np.dot(wm[:,1], wm[:,2])
d2_dot_d4 = np.dot(wm[:,1], wm[:,3])
d1_dot_d2


# In[23]:


cos_sim_d1_d2= d1_dot_d2/(d1_norms * d2_norms) 
cos_sim_d1_d3= d1_dot_d3/(d1_norms * d3_norms) 
cos_sim_d2_d3= d2_dot_d3/(d2_norms * d3_norms) 
cos_sim_d2_d4= d2_dot_d4/(d2_norms * d4_norms) 
cos_sim_d2_d4


# In[22]:


print('Cosine Similary of D1 and D2 : {}'.format(cos_sim_d1_d2))
print('Cosine Similary of D1 and D3 : {}'.format(cos_sim_d1_d3))
print('Cosine Similary of D2 and D3 : {}'.format(cos_sim_d2_d3))
print('Cosine Similary of D2 and D4 : {}'.format(cos_sim_d2_d4))


# In[ ]:




