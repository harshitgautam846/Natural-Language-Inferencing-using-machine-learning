#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:43:05 2019

@author: jack
"""

import nltk
import re
import math
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from fuzzywuzzy import fuzz
from nltk.util import ngrams

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('contradiction.txt', delimiter="\t", header=None, names=['num','sentence1','sentence2','rating','label'])
df.drop(['num','rating'], axis=1,inplace=True)

df['label'] = df['label'].apply(lambda label: 'contradiction' if label=='CONTRADICTION' else 'not contradiction')


contraction = { 'isn\'t':'is not','aren\'t' : 'are not','wasn\'t':'was not','weren\'t':'were not',\
               'haven\'t':'have not', 'hasn\'t' :'has not','hadn\'t' :'had not','won\'t' :'will not',\
               'wouldn\'t':'would not' ,'don\'t':'do not' ,'doesn\'t':'does not','didn\'t':'did not',\
               'can\'t':'can not','couldn\'t':'could not','shouldn\'t':'should not','mightn\'t':'might not',\
               'mustn\'t':'must not'}

le = preprocessing.LabelEncoder()
df['label'] = le.fit_transform(df['label'])

df_majority = df[df['label'] == 1]
df_minority = df[df['label'] == 0]
df_minority_upsampled = resample(df_minority, replace=True, n_samples = 3500)
df = pd.concat([df_majority, df_minority_upsampled])


def exp(sent):
    new_sent=''
    words =sent.split()
    for i in words:
        if i in contraction:
            new_sent+=' '+contraction[i]
        else:
            new_sent+=' '+i
    return new_sent.strip()

def prep(sentence):
    sentence = exp(sentence)
    sentence = re.sub('[^a-zA-Z\']',' ', sentence)
    sentence = sentence.lower()
    sentence = sentence.split()
    sentence = ' '.join(sentence)
    return sentence

df['sentence1'] = df['sentence1'].apply(lambda x: prep(x))
df['sentence2'] = df['sentence2'].apply(lambda x: prep(x))

def jaccard_similarity(sent1, sent2):
    return nltk.jaccard_distance(set(sent1) ,set(sent2))

def edit(sent1 ,sent2):
    return nltk.edit_distance(sent1,sent2)

def get_cosine(vec1, vec2):
    
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

WORD = re.compile(r'\w+')

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)
 
def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return len(lcs_set)

def l2Distance(v1 ,v2):
    return math.sqrt(sum((v1[k] - v2[k])**2 for k in set(v1.keys()).intersection(set(v2.keys()))))
        
def fuzzy(sent1 ,sent2):
    return fuzz.ratio(sent1 , sent2)/100

def speechCount(sent):
    tokens = nltk.word_tokenize(sent)
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    
    res ={'NN':0 ,'JJ' :0 , 'VB' :0}
    #counts = Counter(tag for word,tag in tags)
    
    for word ,tag in tags:
        if tag in ['NN','NNS','NNP','NNPS']:
            res['NN']+=1;
        elif tag in ['JJ','JJR','JJS']:
            res['JJ']+=1
        elif tag in ['VB','VBD','VBG','VBN','VBP','VBZ']:
            res['VB']+=1
    
    total = sum(res.values())
    return dict((word, float(count)/total) for word,count in res.items())
    
def genNgrams(sentence, n):
    l =[token for token in sentence.split()]
    return list(ngrams(l,n))

def cosine_ngrams(a, b):
    vec1 = Counter(a)
    vec2 = Counter(b)
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator

def jaccard_distance(a, b):
    a = set(a)
    b = set(b)
    if len(a|b)==0:
        return 0
    
    return 1.0 * len(a&b)/len(a|b)

tag=[]
for x in range(df.shape[0]):
    d1 = speechCount(df['sentence1'].iloc[x])
    d2 = speechCount(df['sentence2'].iloc[x])
    
    d3 = [ abs(d1[key] - d2.get(key, 0)) for key in d1.keys()]
    tag.append(d3)

df['jacc'] = df.apply(lambda x:jaccard_similarity(x['sentence1'],x['sentence2']),axis=1)
df['edit'] = df.apply(lambda x:edit(x['sentence1'] ,x['sentence2']),axis=1)
df['cosine'] = df.apply(lambda x : get_cosine(text_to_vector(x['sentence1']) ,text_to_vector(x['sentence2']) ),axis =1)
df['lcs'] = df.apply(lambda x:lcs(x['sentence1'], x['sentence2']),axis=1)
df['euclid'] = df.apply(lambda x: l2Distance(text_to_vector(x['sentence1']) ,text_to_vector(x['sentence2'])), axis=1)
df['fuzz'] = df.apply(lambda x: fuzzy(x['sentence1'] ,x['sentence2']), axis=1)
df[['noun','adjective','verb']] = pd.DataFrame((x[0],x[1],x[2]) for x in tag)
df['cos_gram_2'] = df.apply(lambda x: cosine_ngrams(genNgrams(x['sentence1'],2) ,genNgrams(x['sentence2'],2)), axis=1)
df['cos_gram_3'] = df.apply(lambda x: cosine_ngrams(genNgrams(x['sentence1'],3) ,genNgrams(x['sentence2'],3)), axis=1)
df['jacc_gram_2'] = df.apply(lambda x: jaccard_distance(genNgrams(x['sentence1'],2) ,genNgrams(x['sentence2'],2)), axis=1)
df['jacc_gram_3'] = df.apply(lambda x: jaccard_distance(genNgrams(x['sentence1'],3) ,genNgrams(x['sentence2'],3)), axis=1)


X = df.iloc[:,3:]

min_max = MinMaxScaler()
min_max.fit(X)
X = min_max.transform(X)

x_train,x_test,y_train,y_test = train_test_split(X,df['label'],test_size = 0.4,random_state=1)

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 score:",metrics.f1_score(y_test, y_pred))

model = DecisionTreeClassifier(max_depth=5)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 score:",metrics.f1_score(y_test, y_pred))


rforest = RandomForestClassifier(n_estimators=100)
rforest.fit(x_train,y_train)
y_pred = rforest.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
feature_importances = pd.DataFrame(rforest.feature_importances_,index = x_train.columns,columns=['importance']).sort_values('importance',ascending=False)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 score:",metrics.f1_score(y_test, y_pred))

sent1 ='i am a boy'
sent2 ='i am not a boy'

f =[]
f.append(jaccard_similarity(sent1,sent2))
f.append(edit(sent1,sent2))
f.append(get_cosine(text_to_vector(sent1),text_to_vector(sent2)))
f.append(lcs(sent1,sent2))
f.append(l2Distance(text_to_vector(sent1),text_to_vector(sent2)))
f.append(fuzzy(sent1,sent2))
d1 =speechCount(sent1)
d2 =speechCount(sent2)
d3 =[ abs(d1[key] - d2.get(key, 0)) for key in d1.keys()]
f.append(d3[0])
f.append(d3[1])
f.append(d3[2])
f.append(cosine_ngrams(genNgrams(sent1,2) ,genNgrams(sent2,2)))
f.append(cosine_ngrams(genNgrams(sent1,3) ,genNgrams(sent2,3)))
f.append(jaccard_distance(genNgrams(sent1,2) ,genNgrams(sent2,2)))
f.append(jaccard_distance(genNgrams(sent1,2) ,genNgrams(sent2,2)))
f = np.asarray(f)
f = f.reshape(1,-1)
le.inverse_transform(model.predict(f))
#contra['length'].plot.box(grid='True')