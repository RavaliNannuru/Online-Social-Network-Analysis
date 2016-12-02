from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import json
import os
import csv
import pickle
import requests
from TwitterAPI import TwitterAPI

a=0
b=0
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

def getTwitterConnection():
    consumer_Key = 'Hi4BX6SFEBj73Kv3MsPnmuNgy'
    consumer_Secret ='CgFmAQQ7hbbSIU55BNGCt2OrWyngLeBTvMu4BuBVooNlTnWgVd'
    access_token = '773590590323822592-nZeIvwchyN1EDFV6ljNIWLedhnjxS3Q'
    access_token_Secret ='zbPjOcIoMb5UXInUqJT2xx5QHxnW8WDXgj7VHBvv4wyzc'
    twitter = TwitterAPI(consumer_Key,consumer_Secret,access_token,access_token_Secret)
    return twitter
	
def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)
def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()

def sample_tweets(twitter, limit, male_names, female_names):
    tweets = []
    while True:
        try:
            # Restrict to U.S.
            for response in twitter.request('statuses/filter',
                        {'locations':'-124.637,24.548,-66.993,48.9974'}):
                if 'user' in response:
                    name = get_first_name(response)
                    if name in male_names or name in female_names:
                        tweets.append(response)
                        if len(tweets) % 100 == 0:
                            print('found %d tweets' % len(tweets))
                        if len(tweets) >= limit:
                            return tweets
        except:
            print("Unexpected error:", sys.exc_info()[0])
    return tweets
	
def read_screen_names(filename):
    return [line.strip('\n') for line in open(filename)]

def get_users(twitter, screen_names):
    return robust_request(twitter,'users/lookup',{'screen_name':screen_names})

def get_friends(twitter, screen_names):
    uid = []
    request = robust_request(twitter,'friends/ids',{'screen_name':screen_names})
    for r in request:
        uid.append(r)
    print(uid)    
    return sorted(uid)   
def add_all_friends(twitter, users):
    filename = "friends.txt"
    file=open(filename,'w')
    for i in users:
        res = get_friends(twitter, i['screen_name'])
        d=res
        for friend in d:
            f=i['screen_name']+ ","+str(friend)+"\n"
            file.write(str(f))
def count_friends(users):
    userlist=[]
    for i in users:
        userlist+=i['friends']
    count= Counter(userlist)
    return count 
def create_graph(users, friend_counts):
    G=nx.Graph()
    for r in users:
        G.add_node(r['screen_name'])
        for l in r['friends']:
            if friend_counts[l] > 1:
                G.add_edge(r['screen_name'],l)
    return G
	
def  main():
    global a,b
    male_names, female_names = get_census_names()
    print('census_names returned')
    twitter = getTwitterConnection()
    print("Connected")
    tweets = sample_tweets(twitter, 1000, male_names, female_names)
    pickle.dump(tweets, open('tweets.pkl', 'wb'))
    print('saved')
    screen_names = read_screen_names('candidates.txt')
    #screen_names
    print("screen names saved")
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    a =len(users)
    b=len(tweets)
    print("user files are built")
    add_all_friends(twitter, users)
    print("Connected3")


if __name__ == '__main__':
    main()