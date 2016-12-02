# coding: utf-8
import re
from itertools import product
from collections import defaultdict
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import requests
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix

tweets={}
out =Counter()
instance1=''
instance2=''
instance3=''
def get_census_names():
    """ Fetch a list of common male/female names from the census.
    For ambiguous names, we select the more frequent gender."""
    males = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.male.first').text.split('\n')
    females = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.female.first').text.split('\n')
    males_pct = dict([(m.split()[0].lower(), float(m.split()[1]))
                  for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1]))
                    for f in females if f])
    male_names = set([m for m in males_pct if m not in females_pct or
                  males_pct[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pct or
                  females_pct[f] > males_pct[f]])    
    return male_names, female_names	

def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()
			
def tokenize(string, lowercase, keep_punctuation, prefix,
             collapse_urls, collapse_mentions):
    """ Split a tweet into tokens."""
    if not string:
        return []
    if lowercase:
        string = string.lower()
    tokens = []
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)
    if keep_punctuation:
        tokens = string.split()
    else:
        tokens = re.sub('\W+', ' ', string).split()
    if prefix:
        tokens = ['%s%s' % (prefix, t) for t in tokens]
    return tokens
def tweet2tokens(tweet, use_descr=True, lowercase=True,
                 keep_punctuation=True, descr_prefix='d=',
                 collapse_urls=True, collapse_mentions=True):
    """ Convert a tweet into a list of tokens, from the tweet text and optionally the
    user description. """
    tokens = tokenize(tweet['text'], lowercase, keep_punctuation, None,
                       collapse_urls, collapse_mentions)
    if use_descr:
        tokens.extend(tokenize(tweet['user']['description'], lowercase,
                               keep_punctuation, descr_prefix,
                               collapse_urls, collapse_mentions))
    return tokens

def make_vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary))  # If term not present, assign next int.
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]  # looking up a key; defaultdict takes care of assigning it a value.
    print('%d unique terms in vocabulary' % len(vocabulary))
    return vocabulary
	
def make_feature_matrix(tokens_list, vocabulary):
    X = lil_matrix((len(tweets), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        #print ("i= "+str(i))
        for token in tokens:
            j = vocabulary[token]
            X[i,j] += 1
    return X.tocsr()  
	
def get_gender(tweet, male_names, female_names):
    name = get_first_name(tweet)
    if name in female_names:
        return 1
    elif name in male_names:
        return 0
    else:
        return -1
    
def do_cross_val(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    global out,instance1, instance2,instance3
    cv = KFold(len(y), nfolds)
    accuracies = []
    
    for train_idx, test_idx in cv:
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        
        acc = accuracy_score(y[test_idx], predicted)
        #print("-------------------------")
        #print(Counter(y[test_idx]))
        out+=Counter(predicted)
        accuracies.append(acc)
	
    print('predicted'+str(Counter(out)))
    #print(predicted)	
    for i,value in enumerate(predicted):
        if value==0:
            instance1='male tweet::'+ str(tweets[i]['text'].encode("utf-8"))
            break
    for i,value in enumerate(predicted):
        if value==1:
            instance2='female tweet::'+ str(tweets[i]['text'].encode("utf-8"))
            break
    for i,value in enumerate(predicted):
        if value==-1:
            instance3='female tweet::'+ str(tweets[i]['text'].encode("utf-8"))
            break
    avg = np.mean(accuracies)
    return avg

def main():
    global tweets
    male_names, female_names = get_census_names()
    tweets=pickle.load(open('tweets.pkl', 'rb'))
    tokens_list = [tweet2tokens(item, use_descr=True, lowercase=True,
                            keep_punctuation=False, descr_prefix='d=',
                            collapse_urls=True, collapse_mentions=True)
							for item in tweets]
    #print(tokens_list)
    print("-------------------------")		  
    vocabulary = make_vocabulary(tokens_list)
    X = make_feature_matrix(tokens_list, vocabulary)
    print('shape of X:', X.shape)
    index2term = {i: t for t, i in vocabulary.items()}
    beta = np.ones(len(vocabulary))  # assume Beta = vector of 1s
    z = np.zeros(len(tweets))
    for i in range(len(tweets)):  # for each row.
        for j in range(X.indptr[i], X.indptr[i+1]): # for each col.
            colidx = X.indices[j]
            z[i] += beta[colidx] * X.data[j]
    print('X * beta for tweet 200=', z[200])
    print('which is the same as the sum %.1f, since beta=[1...1]' % X[200].sum())
    y = np.array([get_gender(t, male_names, female_names) for t in tweets])
    print('gender labels:', Counter(y))	
    print('avg accuracy', do_cross_val(X, y, 5))
    #print('gender predict:', Counter(y))	
    #for t in tweets:
        #name = get_first_name(t)
        #print(name)
        #if name in female_names:
            #print('female tweet::'+ str(t['text'].encode("utf-8")))
            #break
    #for t in tweets:
        #name = get_first_name(t)
        #if name in male_names:
            #print('male tweet::'+ str(t['text'].encode("utf-8")))
            #break
        
if __name__ == '__main__':
    main()