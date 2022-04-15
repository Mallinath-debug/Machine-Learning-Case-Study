#!/usr/bin/env python
# coding: utf-8

# # Loading some important libraries for loading data

# In[ ]:


################################
## Name - Mallinath Elekar
## Data Set -
## Independent Variable - Wether 
## Dependent Variable - Play
## We have to predict whether he can play or not
## df is used as variable to store the data
## This dataset has no null values
## 1)Load the data using pandas
## 2)Preprocessing,Data Analysis
## 3)Train the data
## 4)Test the data
## 5) Improve the accuracy
##
##
##
################################


# In[ ]:





# In[14]:


import pandas as pd


# # Load the data

# In[15]:


df=pd.read_csv('Play.csv')
df.head()


# # Data Analysis

# In[16]:


#How big is the data
df.shape


# In[17]:


#datatype of columns
df.info()


# In[18]:


#is there any missing value
df.isnull().sum()


# In[19]:


#how does the data look mathematically
df.describe()


# In[20]:


#are there any duplicate values
df.duplicated().sum()


# # Generating Profile Report For Data Analysis 

# In[21]:


from pandas_profiling import ProfileReport
prof=ProfileReport(df)
prof.to_file(output_file='output.html')


# In[22]:


df.head()


# # Converting Categorical Data Into Numeric Using Label Encoder

# In[23]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[24]:


df['Wether_N']=le.fit_transform(df['Wether'])
df.head()


# In[25]:


df['Temperature_N']=le.fit_transform(df['Temperature'])
df.head()


# In[26]:


df['Play_N']=le.fit_transform(df['Play'])
df.head()


# # Dropping the old columns

# In[27]:


df1=df.drop(['Wether','Temperature','Play'],axis=1)
df1.head()


# In[31]:


X=df[['Wether_N','Temperature_N']]
Y=df[['Play_N']]


# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[33]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5)


# In[34]:


from sklearn import tree
clf=tree.DecisionTreeClassifier()
model=clf.fit(X_train,Y_train)
result=model.predict(X_test)
Percentage=accuracy_score(result,Y_test)
print('Accuarcy is',Percentage*100)


# In[35]:


from sklearn.neighbors import KNeighborsClassifier


# In[39]:


clf=KNeighborsClassifier()
model=clf.fit(X_train,Y_train)
result=model.predict(X_test)
percentage=accuracy_score(result,Y_test)
print("Accuracy is",percentage*100)


# In[ ]:




