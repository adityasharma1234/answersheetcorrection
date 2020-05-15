#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = "../input/automated-essay-scoring-dataset/"
s_directory = './'
glove = './glove.6B/'


# In[34]:


X = pd.read_csv(os.path.join(data, 'training_set_rel3.tsv'), sep='\t', encoding='ISO-8859-1')
y = X['domain1_score']
X = X.dropna(axis=1)
X = X.drop(columns=['rater1_domain1', 'rater2_domain1'])
import os
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
minimum_scores = [-1, 2, 1, 0, 0, 0, 0, 0, 0]
maximum_scores = [-1, 12, 6, 3, 3, 4, 4, 30, 60]


def toword(es, remove_stopwords):

    es = re.sub("[^a-zA-Z]", " ", es)
    words = es.lower().split()
    if remove_stopwords:
        stop = set(stopwords.words("english"))
        w = [w for w in words if not w in stop]
    return (w)


def tosentence(es, remove_stopwords):

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    initialsentence = tokenizer.tokenize(es.strip())
    s = []
    for r in initialsentence:
        if len(r) > 0:
            s.append(toword(r, remove_stopwords))
    return s


def make_feature(w, model, numberoffeaturess):
    f_vector = np.zeros((numberoffeaturess,),dtype="float32")
    numberofwords = 0.
    indexset = set(model.wv.index2word)
    for word in w :
        if word in indexset:
            numberofwords += 1
            f_vector = np.add(f_vector,model[word])
    f_vector = np.divide(f_vector,numberofwords)
    return f_vector

def AverageFeature(essays, model, num_features):
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = make_feature(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs
