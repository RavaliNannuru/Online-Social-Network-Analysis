from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import json
from TwitterAPI import TwitterAPI

def getTwitterConnection():
    consumer_Key = 'Hi4BX6SFEBj73Kv3MsPnmuNgy'
    consumer_Secret ='CgFmAQQ7hbbSIU55BNGCt2OrWyngLeBTvMu4BuBVooNlTnWgVd'
    access_token = '773590590323822592-nZeIvwchyN1EDFV6ljNIWLedhnjxS3Q'
    access_token_Secret ='zbPjOcIoMb5UXInUqJT2xx5QHxnW8WDXgj7VHBvv4wyzc'
    twitter = TwitterAPI(consumer_Key,consumer_Secret,access_token,access_token_Secret)
    return twitter

twitter = getTwitterConnection()

def robust_request(twitter, resource, params, max_tries=5):
   for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print >> sys.stderr, 'Got error:', request.text, '\nsleeping for 15 minutes.'
            sys.stderr.flush()
            time.sleep(61 * 15)

def getTweets(query):
    response = robust_request(twitter,'search/tweets',{'q':query,'count':10000})
    data = json.loads(response.text)
    tweetsList = []
    for tweet in data['statuses']:
        #print(tweet['text'])
        tweetsList.append(tweet['text'])
    return tweetsList

tweets = getTweets('iphone7')
print(tweets)



