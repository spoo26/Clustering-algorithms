
# coding: utf-8

# In[3]:

import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('mall.csv')
dataset.head(3)


# **clustering based on annual incoming and  Spending Score
# 

# In[4]:

X= dataset.iloc[:,[3,4]].values


# In[5]:

print ("AI  and SS\n",X)


# **Elbow method
# 

# In[6]:

from sklearn.cluster import KMeans
wcss =[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    #inertia is used for finding with in sum of squares in kmeans clustering method


# In[7]:

wcss


# In[8]:

plt.plot(range(1,11),wcss)
plt.title("the elbow method")
plt.xlabel("The number of clusters")
plt.ylabel("wcss")
plt.show()


# In[9]:

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        
y_kmeans=kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "sensible")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "cyan", label = "standard")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "magenta", label = "target")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "green", label = "careless")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "orange", label = "careful")

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="yellow", label = "Centroids")
plt.title("CLusters of clients")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score (1–100)")
plt.legend()
plt.show()


# In[ ]:



