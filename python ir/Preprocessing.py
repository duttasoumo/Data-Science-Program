#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import string
dataset = nltk.corpus.gutenberg
text = dataset.words('austen-emma.txt')

print("Original Text:\n", text[:100])


# In[3]:


#Stopword and Punctuation removal
sw = nltk.corpus.stopwords.words('english') + list(string.punctuation)
text_1 = [w for w in text if w not in sw]

print("Stopwords and Punctuation Removed:\n", text_1[:100])


# In[6]:


#Stemming
stemmer = nltk.stem.PorterStemmer()

text_2 =  [stemmer.stem(w) for w in text_1]
print("Stemmed:\n", text_2[:100])

