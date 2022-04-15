#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[3]:


wine=datasets.load_wine()


# In[4]:


print(wine.feature_names)


# In[5]:


print(wine.target_names)


# In[6]:


print(wine.data[0:5])


# In[7]:


print(wine.target)


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3)


# In[9]:


knn=KNeighborsClassifier(n_neighbors=1)
model=knn.fit(X_train,y_train)
result=model.predict(X_test)
Percentage=accuracy_score(result,y_test)
print("Accuracy is ",Percentage*100)


# In[17]:


from sklearn import tree
clf=tree.DecisionTreeClassifier()
model=clf.fit(X_train,y_train)
result=model.predict(X_test)
Percentage=accuracy_score(result,y_test)
print("Accuracy is",Percentage*100)


# In[12]:


from sklearn.linear_model import LogisticRegression 


# In[16]:


clf=LogisticRegression()
model=clf.fit(X_train,y_train)
result=model.predict(X_test)
Percentage=accuracy_score(result,y_test)
print("Accuracy is :",Percentage*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




