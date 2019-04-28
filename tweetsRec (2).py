
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd

from pandas import DataFrame
#from sklearn.linear_model import LogisticRegression


# In[13]:


#loading dataset
all_tweets = pd.read_csv('all_tweets.csv',encoding = "ISO-8859-1",error_bad_lines=False)
all_tweets.head()


# In[5]:


#计算hashtag是空值的行数
all_tweets['hashtags_c'].isnull().value_counts()


# In[14]:


#删除hasgtag是空值的行
all_tweets = all_tweets.dropna(subset=["hashtags_c"])


# In[327]:


#查看空值行已删除
all_tweets['hashtags_c'].isnull().value_counts()


# In[328]:


all_tweets.tail()


# In[15]:


#列筛选以便做相似用户
col_n=['id','user','text','retweet_count']
all_tweet=pd.DataFrame(all_tweets,columns=col_n)
all_tweet.head()


# In[16]:


all_user=all_tweet.drop(['id','retweet_count'],axis=1)


# In[17]:


import re
def url_rem(s):
    s = re.sub(r'http://[a-zA-Z0-9.?/&=:]*',"",s)
    return s
#results=re.compile(r'http[s]://[a-zA-Z0-9.?/&=:]*',re.S)
#dd=re.sub(r'http[s]://[a-zA-Z0-9.?/&=:]*',"",s)

all_user['text']=all_user['text'].apply(url_rem)




# In[19]:


#install gensim using 'pip install gensim' in advance
import gensim
#from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')

nltk.download('stopwords')
from nltk.corpus import stopwords

# In[14]:


#define stopwords
custom_stop_words = ['say', 'saying', 'sayings',
                     'says', 'us', 'un', 'it', 'would',
                     'let', 'just', 'said', 'is','as',
                     'not','a','by','on','ture','go',
                     'goes','went','going','none','can',
                    'for','of','may','have','get']
                    
stop = stopwords.words('english')+custom_stop_words


# In[26]:


import string
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s


# In[16]:


#删除标点符号
all_user['text']=all_user['text'].apply(remove_punctuation)


# In[17]:


#全部转成小写
all_user['text']=all_user['text'].str.lower()
all_user.head()


# In[27]:


from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
stemmer=SnowballStemmer('english')


# In[28]:



def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[20]:


#doc_sample = all_user[all_user['user'] == 69030191].values[0][1]




processed_user=all_user
processed_user['text']=processed_user['text'].map(preprocess)


# In[31]:


users_merge =processed_user.groupby(['user']).sum().reset_index('user')



# In[34]:


from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# create training and testing vars
train, test = train_test_split(users_merge, test_size=0.2)
print (train.shape)
print (test.shape)



# In[35]:


dictionary = gensim.corpora.Dictionary(train['text'])


# In[33]:


count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[36]:


dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


# In[37]:


bow_corpus = [dictionary.doc2bow(doc) for doc in train['text']]


# In[38]:


lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
lda_model.save('lda.model')


# In[40]:


for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))



# In[42]:


x=[]
for i in range(10):
        x.append(lda_model[bow_corpus[1]][i][1])
x



# In[45]:


bow_doc_0 = bow_corpus[0]
for i in range(len(bow_doc_0)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_0[i][0], 
                                               dictionary[bow_doc_0[i][0]], 
bow_doc_0[i][1]))


# In[47]:


for index, score in sorted(lda_model[bow_corpus[1]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic{}: {}".format(score,index, lda_model.print_topic(index, 10)))




# In[665]:

#
##df =pd.DataFrame(columns=('user','topic0','topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9'))
##dic2={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
##topic_matrix=pd.DataFrame(index=[train['user']],columns=('0','1','2','3','4','5','6','7','8','9'))
#topic_matrix=pd.DataFrame(columns=('0','1','2','3','4','5','6','7','8','9'))
#topic_matrix.insert(0, 'user_id', train['user'])
#topic_matrix_r=topic_matrix.reset_index(drop=True)
##save train to a csv file
#train.to_csv("train.csv",sep=',')
#
#for i in range(202201,len(bow_corpus)):
#    for index,score in lda_model[bow_corpus[i]]:
#        #print("start writing to line {}..".format(i))
#        topic_matrix_r.loc[[i],[str(index)]]=score
#topic_matrix_r.to_csv("topic_m.csv",sep=',')


# In[47]:


topic_matrix_r=pd.read_csv('topic_m.csv')

# In[42]:


#from gensim import corpora, models, similarities
#tfidf = models.TfidfModel(bow_corpus)
#corpus_tfidf = tfidf[bow_corpus]
#from pprint import pprint
#for doc in corpus_tfidf:
#    pprint(doc)
#    break
#lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
#for idx, topic in lda_model_tfidf.print_topics(-1):
#    print('Topic: {} Word: {}'.format(idx, topic))
# In[732]:


#for index, score in sorted(lda_model_tfidf[corpus_tfidf[1]], key=lambda tup: -1*tup[1]):
#    for i in range(10):
#        #vec=lda_model_tfidf.print_topic(index)
#        vec=score

# In[929]:

##generate data and target for SVM model
#data=[]
#target=[]
#for s in range(2000):
#    length=len(lda_model[bow_corpus[s]])
#    temp=[]
#    for i in range(length):
#        temp.append(lda_model[bow_corpus[s]][i][1])
#        #data.append(temp)
#        idx=temp.index(max(temp))
#        t=lda_model[bow_corpus[s]][idx][0]
#    data.append(temp)
#    target.append(t)  




# In[52]:

from sklearn import metrics


# In[53]:
topic_M=topic_matrix_r[:10]
topic_M=topic_M.where(topic_M.notnull(), 0)
#topic_m=topic_M.drop(['user_id'],axis=1)
topic_m=topic_M.set_index('user_id')

# In[ ]:
#K(X, Y) = <X, Y> / (||X||*||Y||)

saveAllConSim=metrics.pairwise.cosine_similarity(topic_m, Y=None, dense_output=True)

#get target user id 
def GetUserId(data):
    idx=data.index.tolist()[0]
    user=topic_m.index[idx]
    return user
def RelativeUserId(x):
    for i in range(4):
        idx=x.index.tolist()[i+1]
        user=topic_m.index[idx]
        x['similar user'][idx]=user
def SimUserOutput(x):
    n=0
    for a in x.index:       
        if n==0:
            print('Tartget User:' + str(topic_m.index[a]) + '\n')
        else:
            print('\t\tSimilar users:' + str(topic_m.index[a]) + ' \n')            
        n=n+1 
for i in saveAllConSim:
    
    #将数据变为DataFrame
    newDataFrame = DataFrame(i)   
    newDataFrame.columns = ['conSim']
    newDataFrame.columns.name = 'user_idx'
    sortedConSim = newDataFrame.sort_values(by='conSim',ascending=False)[:5]
    User=GetUserId(sortedConSim)
    sortedConSim.insert(1,'similar user',User)
    RelativeUserId(sortedConSim)
    sortedConSim.to_csv("{0}.csv".format(User))
    SimUserOutput(sortedConSim)

Target=pd.read_csv("716984586.csv",index_col=0)
target=Target.reset_index(drop=True)



def GetTweets(all_tweet,target):
    indicate_rec=[]
    for i in range(1,5):
        indicate_rec.append(all_tweet[all_tweet['user'].isin([target['similar user'][i]])])
    return indicate_rec


def GetText(all_tweet,target):
    a=[]
    x=all_tweet['user'].tolist()
    for i in range(1,5):
        y=x.index(target['similar user'][i])
        indicate_rec=all_tweet['text'].iloc[y]
        a.append(indicate_rec)
    return a
    
Document=GetText(all_tweet,target)


        
        