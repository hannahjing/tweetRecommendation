
# coding: utf-8

# In[12]:
import topic_modeling
from topic_modeling import *
import numpy as np
import pandas as pd
#install gensim using 'pip install gensim' in advance
from pandas import DataFrame
#from gensim.utils import simple_preprocess
np.random.seed(2018)
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import sty
import copy
from sty import fg, bg, ef, rs, RgbFg
def GetText(all_t,tar):
#    idx=tar.index[0]
#    s=tar.drop([idx])
#    y=s.index.tolist()
    y=tar.index.tolist()
    a=all_t[['text']][all_t.index.isin(y)]
   # a.insert(1,'id',all_t[['tweet_id']][all_t.index.isin(y)])
    a.insert(1,'tweet_id',all_t[['id']][all_t.index.isin(y)])
    return a



#def TargetText(user):
#    Target_Text=all_tweet[['text']].loc[all_tweet.index.isin([user])]
##    Target_Text=Target_Text.groupby([Target_Text.index]).sum().to_frame()
#    return Target_Text
    
def WeightGen(data,target,doc):
    vectorizer = CountVectorizer()
#    Document=GetText(all_tweet,Target)
    D=copy.deepcopy(Document)
    D['text']=topic_modeling.data_preprocess(D) 
#    doc['text']=topic_modeling.data_preprocess(doc)
    X = vectorizer.fit_transform(doc['text'])
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
#    Word = vectorizer.get_feature_names()
    #Analyze = vectorizer.build_analyzer()
    Weight = tfidf.toarray()
    return Weight

def TargetTfIdf(w,d,user):
    tfidf=w[d.index.isin([user])]
    tfidf_mean=np.mean(tfidf,axis=0)
    return tfidf_mean
    
    
    
    
    
#Calculating cosine similarity
def CosValues(doc,weight,user):
    Numb = 0
    Agen = 0
    Bgen = 0
    cos=[]
    idx=doc.index.isin([user])
    tfidf=weight[idx]
    tfidf_mean=np.mean(tfidf,axis=0)
    l=len(tfidf)
    w=weight[~(idx)]
    for i in range(len(weight)-l):
        for v, k in zip(w[i], tfidf_mean):
            Numb += v * k
            Agen += v**2
            Bgen += k**2
        CosSim= Numb / (math.sqrt(Agen) * math.sqrt(Bgen))
        cos.append(CosSim)
#    print(cos)
    return cos
def MaxCosIdx(doc,weight,user):
    cos=CosValues(doc,weight,user)
    temp=[]
    Inf = 0
    for i in range(10):
        temp.append(cos.index(max(cos)))
        cos[cos.index(max(cos))]=Inf
    temp.sort()
#    print(temp)
    return temp
   
def Recommend(data,weight,target,doc,user): 
    END = '\033[0m'
    temp=MaxCosIdx(doc,weight,user)
    doc=doc.reset_index(drop=False)
    print("\033[1;43m Recommendation for user:\t"+str(user)+END)
    A=[]
    B=[]
    C=[]
    for i in temp:
        RecText=doc.iloc[i]['text']
        RecUser=doc.iloc[i]['user']
        RecId=doc.iloc[i]['tweet_id']
        A.append(RecUser)
        B.append(RecId)
        C.append(RecText)
        
#        print(ef.italic + '{0}\n'.format(i)+fg(255, 10, 10)+'userID: '+str(RecUser)+'\n [tweet_id:'+str(RecId)+']  '+fg.rs+str(RecText)+'\n\n'+ rs.italic)     
    dataframe = pd.DataFrame({'user id':A,'tweet id':B,'tweets':C})
    dataframe.to_csv("Recommendations/{0}.csv".format(user))

 
if __name__ == '__main__':
    all_tweet=topic_modeling.data_load()    
    all_tweet=all_tweet.set_index(['user'],drop=True)
    
    topic_matrix_r=pd.read_csv('topic_m.csv')
    topic_M=topic_matrix_r[:10000]
    topic_M=topic_M.where(topic_M.notnull(), 0)
    topic_m=topic_M.set_index('user_id')

    #K(X, Y) = <X, Y> / (||X||*||Y||)

#    saveAllConSim=metrics.pairwise.cosine_similarity(topic_m, Y=None, dense_output=True)
#
#    
#    for i in saveAllConSim:
#        #将数据变为DataFrame
#        newDataFrame = DataFrame(i)   
#        newDataFrame.columns = ['conSim']
#        newDataFrame=newDataFrame.set_index(topic_m.index,drop=True)
#        sortedConSim = newDataFrame.sort_values(by='conSim',ascending=False)[:20]
#        User=sortedConSim.index[0]
#        sortedConSim.to_csv("TargetUsers/{0}.csv".format(User))
#    
#    print('opening target user csv...')
#    path=r'TargetUsers/27089302.csv'
#    o=open(path)
    print('Recommending...')
   # Target=pd.read_csv(o,index_col=0)
    Target=cluster_m['cluster_id'].to_frame()

    Document=GetText(all_tweet,Target)
    Weight=WeightGen(all_tweet,Target,Document)
    for i in range(len(Target)):
        userID=Target.index[i]
        Recommend(all_tweet,Weight,Target,Document,userID)
        
#    userID=Target.index[80]
#    Recommend(all_tweet,Weight,Target,Document,userID)
 #   Recommend(27089302)
        
    
    





        
        