#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:04:50 2019

@author: jing
"""
import pandas as pd
#from pandas import DataFrame, Series
import re
import string
 #import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

class topic_modeling():
   

    #######data preprocessing....

    def data_load():
       #loading dataset
       all_tweets = pd.read_csv('all_tweets.csv',encoding = "ISO-8859-1",error_bad_lines=False)
       #null hashtag row count
       all_tweets['hashtags_c'].isnull().value_counts()
       #delete rows with null hashtag
       all_tweets = all_tweets.dropna(subset=["hashtags_c"])
       #check the rows after deleting null hashtag
       all_tweets['hashtags_c'].isnull().value_counts()
       print('null hashtag removed')
       #data selection
       col_n=['id','user','text','retweet_count']
       all_tweet=pd.DataFrame(all_tweets,columns=col_n)
       return all_tweet



    #remove url from tweets
    def url_rem(s):
        s = re.sub(r'http://[a-zA-Z0-9.?/&=:]*',"",s)
        return s
    #remove punctuations from tweets
    def remove_punctuation(s):
        s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
        return s

    def stopwords_removal(s):
        custom_stop_words = ['say', 'saying', 'sayings',
                     'says', 'us', 'un', 'it', 'would',
                     'let', 'just', 'said', 'is','as',
                     'not','a','by','on','ture','go',
                     'goes','went','going','none','can',
                    'for','of','may','have','get']
        stop = stopwords.words('english')+custom_stop_words
        s['text'].apply(lambda x: [item for item in x if item not in stop])

    #stemming
    def lemmatize_stemming(text):
        stemmer=SnowballStemmer('english')
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result
        print(' preprocessed')


    def data_preprocess(t):
        #t['text']=url_rem(t['text'])        
        t['text']=t['text'].apply(topic_modeling.url_rem)
        print('url removed')
   
        t['text']=t['text'].apply(topic_modeling.remove_punctuation)
        print('punctuation removed')
        #convert to lowercase
        t['text']=t['text'].str.lower()
        print('convert to lowercase')
        return t

if __name__ == '__main__':

    all_tweet=topic_modeling.data_load()
    all_user=all_tweet.drop(['id','retweet_count'],axis=1)
    print('user selected')     
    processed_user=topic_modeling.data_preprocess(all_user)
    processed_user['text']=processed_user['text'].map(topic_modeling.preprocess)
    users_merge =processed_user.groupby(['user']).sum().reset_index('user')
    print('merge done')
    # create training and testing vars
    train, test = train_test_split(users_merge, test_size=0.2)
    print (train.shape)
    print (test.shape)
    print('training testing splited')
    print('training set has been saved to trian.csv file') 
    dictionary = gensim.corpora.Dictionary(train['text'])
    print('dictionary generated')
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    print('dictionary filtered')
    #create corpus
    bow_corpus = [dictionary.doc2bow(doc) for doc in train['text']]
    print('corpus generated')
    #train  the model
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    lda_model.save('lda.model')
    print('lda model trained')
    #topic modeling and save to csv file
    topic_matrix=pd.DataFrame(index=[train['user']],columns=('0','1','2','3','4','5','6','7','8','9'))
    topic_matrix=pd.DataFrame(columns=('0','1','2','3','4','5','6','7','8','9'))
    topic_matrix.insert(0, 'user_id', train['user'])
    topic_matrix_r=topic_matrix.reset_index(drop=True)
    #save train to a csv file
    train.to_csv("train.csv",sep=',')
    #this function takes long time
    for i in range(202201,len(bow_corpus)):
        for index,score in lda_model[bow_corpus[i]]:
            print("start writing to line {}..".format(i))
            topic_matrix_r.loc[[i],[str(index)]]=score
    topic_matrix_r.to_csv("topic_m.csv",sep=',')



    

