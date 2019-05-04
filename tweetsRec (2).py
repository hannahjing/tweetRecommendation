
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import heapq   
import re
import string
from pandas import DataFrame, Series
#install gensim using 'pip install gensim' in advance
import gensim

#from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
np.random.seed(2018)
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from matplotlib import pyplot as plt
from sklearn import metrics
# In[13]:
#######data preprocessing....

#loading dataset
all_tweets = pd.read_csv('all_tweets.csv',encoding = "ISO-8859-1",error_bad_lines=False)
all_tweets.head()
#计算hashtag是空值的行数
all_tweets['hashtags_c'].isnull().value_counts()

#删除hasgtag是空值的行
all_tweets = all_tweets.dropna(subset=["hashtags_c"])

#查看空值行已删除
all_tweets['hashtags_c'].isnull().value_counts()
print('null hashtag removed')
#列筛选以便做相似用户
col_n=['id','user','text','retweet_count']
all_tweet=pd.DataFrame(all_tweets,columns=col_n)
all_user=all_tweet.drop(['id','retweet_count'],axis=1)
all_tweet=all_tweet.set_index(['user'],drop=True)
print('user selected')
#remove url from text
def url_rem(s):
    s = re.sub(r'http://[a-zA-Z0-9.?/&=:]*',"",s)
    return s

#define stopwords
custom_stop_words = ['say', 'saying', 'sayings',
                     'says', 'us', 'un', 'it', 'would',
                     'let', 'just', 'said', 'is','as',
                     'not','a','by','on','ture','go',
                     'goes','went','going','none','can',
                    'for','of','may','have','get']
                    
stop = stopwords.words('english')+custom_stop_words

def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s


stemmer=SnowballStemmer('english')


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
    print(' preprocessed')


def data_preprocess(all_user):
    #remove url from text
    all_user['text']=all_user['text'].apply(url_rem)
    print('url removed')
    #删除标点符号
    all_user['text']=all_user['text'].apply(remove_punctuation)
    print('punctuation removed')

    #全部转成小写
    all_user['text']=all_user['text'].str.lower()
    print('convert to lowercase')
    return all_user
    

    
    



# In[33]:


#count = 0
#for k, v in dictionary.iteritems():
#    print(k, v)
#    count += 1
#    if count > 10:
#        break
#

# In[36]:





# In[40]:

#
#for idx, topic in lda_model.print_topics(-1):
#    print('Topic: {} \nWords: {}'.format(idx, topic))
#
#
#
## In[42]:
#
#
#x=[]
#for i in range(10):
#        x.append(lda_model[bow_corpus[1]][i][1])
#
#
#
#
## In[45]:
#
#
#bow_doc_0 = bow_corpus[0]
#for i in range(len(bow_doc_0)):
#    print("Word {} (\"{}\") appears {} time.".format(bow_doc_0[i][0], 
#                                               dictionary[bow_doc_0[i][0]], 
#bow_doc_0[i][1]))
#
#
## In[47]:
#
#
#for index, score in sorted(lda_model[bow_corpus[1]], key=lambda tup: -1*tup[1]):
#    print("\nScore: {}\t \nTopic{}: {}".format(score,index, lda_model.print_topic(index, 10)))
#



# In[665]:

#topic modeling and save to csv file
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




# In[42]:
###可以删除
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



    processed_user=data_preprocess(all_user)
    processed_user['text']=processed_user['text'].map(preprocess)
    users_merge =processed_user.groupby(['user']).sum().reset_index('user')
    print('merge done')

    # create training and testing vars
    train, test = train_test_split(users_merge, test_size=0.2)
    print (train.shape)
    print (test.shape)
    print('training testing splited')


# In[53]:
    
    
 
    
    
topic_matrix_r=pd.read_csv('topic_m.csv')
topic_M=topic_matrix_r[:5000]
topic_M=topic_M.where(topic_M.notnull(), 0)
#topic_m=topic_M.drop(['user_id'],axis=1)
topic_m=topic_M.set_index('user_id')

#K(X, Y) = <X, Y> / (||X||*||Y||)

saveAllConSim=metrics.pairwise.cosine_similarity(topic_m, Y=None, dense_output=True)

#get target user id 
#def GetUserId(data):
#    idx=data.index.tolist()[0]
#    user=topic_m.index[idx]
#    return user
#get 19 similar user id
#def RelativeUserId(x):
#    for i in range(19):
#        idx=x.index.tolist()[i+1]
#        user=topic_m.index[idx]
#        x['similar user'][idx]=user
#output the similar user
#def SimUserOutput(x):
#    n=0
#    for a in x.index:       
#        if n==0:
#            #print('Tartget User:' + str(topic_m.index[a]) + '\n')
#            print('Tartget User:' + str(a) + '\n')
#        else:
#            #print('\t\tSimilar users:' + str(topic_m.index[a]) + ' \n')
#            print('\t\tSimilar users:' + str(a) + ' \n')             
#        n=n+1 
#save all the cosine similarity       
for i in saveAllConSim:
    
    #将数据变为DataFrame
    newDataFrame = DataFrame(i)   
    newDataFrame.columns = ['conSim']
    newDataFrame=newDataFrame.set_index(topic_m.index,drop=True)
    #newDataFrame.columns.name = 'user_idx'
    sortedConSim = newDataFrame.sort_values(by='conSim',ascending=False)[:20]
    #User=GetUserId(sortedConSim)
    User=sortedConSim.index[0]
    #sortedConSim.insert(1,'similar user',User)
    #RelativeUserId(sortedConSim)
    sortedConSim.to_csv("TargetUsers/{0}.csv".format(User))
    #SimUserOutput(sortedConSim)

def GetText(all_t,tar):
    idx=tar.index[0]
    tar.drop([idx])
    y=tar.index.tolist()
    a=all_tweet[['text']][all_tweet.index.isin(y)]
    return a
#    a=[]
#    #x=all_t['user'].tolist()
#    for i in range(1,20):
#        #y=x.index(tar.index[i])
#        y=tar.index[i]
#        if(all_t.index=y):
##        indicate_rec=all_t['text'].loc[y]
##        a.append(indicate_rec)
#            a.append(all_tweet['text'].loc[y])
#    return a


def TargetTxtGen(user):
    Document=GetText(all_tweet,Target)
   # Document=Series.tolist(Document)
    #Document.append(all_tweet['text'].loc[all_tweet.index.isin([716984586])])
    #Target_Text=all_tweet['text'].loc[user]
    Target_Text=all_tweet['text'].loc[all_tweet.index.isin([user])]
    Target_Text=Target_Text.groupby([Target_Text.index]).sum().to_frame()
    Document.append(Target_Text)
    return Document
    #Target_Text=Series.tolist(Target_Text)
#    if len(Target_Text)>1:
#        Target_Text=' '.join(Target_Text)
#    #for i in range(len(Target_Text)):
#        #Document.append(Target_Text[i])  
#   # return Document
#    Document.append(Target_Text)
#    return Document
    
    

    #将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频

 
   #该类会统计每个词语的tf-idf权值
#transformer = TfidfTransformer()
 
    #第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
#tfidf = transformer.fit_transform(vectorizer.fit_transform(Document))
 
    #获取词袋模型中的所有词语  
#word = vectorizer.get_feature_names()
 
    #将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
#weight = tfidf.toarray()
 
    #打印特征向量文本内容
#print ('Features length: ' + str(len(word)))
#for j in range(len(word)):
#    print( word[j])
 
    #打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
#for i in range(len(weight)):
#    for j in range(len(word)):
#        print( weight[i][j],)
#    print ('\n')
    #打印特征向量文本内容
#print ('Features length: ' + str(len(word)))
#for j in range(len(word)):
#    print( word[j], )
#    #打印每类文本词频矩阵
#print ('TF Weight: ')
#for i in range(len(Weight)):
#    for j in range(len(Word)):
#        print (Weight[i][j],)
#    print ('\n')
    
#Calculating cosine similarity
def CosValues(weight):
    Numb = 0
    Agen = 0
    Bgen = 0
    cos=[]
    for i in range(len(weight)-1):
        for v, k in zip(weight[i], weight[len(weight)-1]):
            Numb += v * k
            Agen += v**2
            Bgen += k**2
        CosSim= Numb / (math.sqrt(Agen) * math.sqrt(Bgen))
        cos.append(CosSim)
    print(cos)
    return cos
def MaxCosIdx(weight):
    cos=CosValues(weight)
    #找到最相似的三条tweets的index
    temp=[]
    Inf = 0
    for i in range(10):
        temp.append(cos.index(max(cos)))
        cos[cos.index(max(cos))]=Inf
    temp.sort()
    print(temp)
    return temp
   
def Recommend(user): 
    vectorizer = CountVectorizer()
    Document=TargetTxtGen(user)   
    X = vectorizer.fit_transform(Document['text'])
    Word = vectorizer.get_feature_names()
    Analyze = vectorizer.build_analyzer()
    Weight = X.toarray()  
    temp=MaxCosIdx(Weight)
    Document=Document.reset_index(drop=False)
    for i in temp:
        RecText=Document.loc[i]['text']
        RecUser=Document.loc[i]['user']
        print('{0}\n'.format(i)+'userID: '+str(RecUser)+'\n'+str(RecText)+'\n\n')
        
  
 
if __name__ == '__main__':
    #identify target user


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

    print('opening target user csv...')
    path=r'TargetUsers/27089302.csv'
    o=open(path)
    print('Recommending...')
    Target=pd.read_csv(o,index_col=0)
    print('CSV file read')
    Recommend(27089302)
        
    
    





        
        