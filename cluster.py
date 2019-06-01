#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:58:14 2019

@author: jing
"""
import tweetsRec(2)
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=bow_corpus, texts=train['text'], start=5, limit=20, step=6)
# Show graph
import matplotlib.pyplot as plt
limit=20; start=5; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()



from sklearn.cluster import KMeans

# random state, we will use 42 instead of 10 for a change
rs = 42

# set the random state. different random state seeds might result in different centroids locations
model = KMeans(n_clusters=10, random_state=rs)
model.fit(topic_m)

# sum of intra-cluster distances
print("Sum of intra-cluster distance:", model.inertia_)

print("Centroid locations:")
for centroid in model.cluster_centers_:
    print(centroid)
    
    
r = pd.concat([topic_m, pd.Series(model.labels_, index = topic_m.index)], axis = 1)  #详细输出每个样本对应的类别
r.columns = list(topic_m.columns) + [u'聚类类别'] #重命名表
r.to_csv('cluster')




import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


#不同类别用不同颜色和样式绘图
d = tsne[r[u'user number'] == 0]
plt.plot(d[0], d[1], 'r.')
d = tsne[r[u'user number'] == 1]
plt.plot(d[0], d[1], 'go')
d = tsne[r[u'聚类类别'] == 2]
plt.plot(d[0], d[1], 'b*')
plt.show()
