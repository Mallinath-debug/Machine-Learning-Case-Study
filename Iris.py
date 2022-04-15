#!/usr/bin/env python
# coding: utf-8

# In[7]:


#######################################################################
## Dataset - Iris Dataset
## Independent Variable - there are four independent variable i.e SepalLength SepalWidth PetalLength,Petal Width
## Dependent Variable -Species
## Dependent Variable has 3 types i.e Iris Sentosa ,Iris Versicolor,Iris Verginica
## We have to decide which flower it is
##
##
##
##
##
##
##
##
##########################################################################


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


df=pd.read_csv('Iris.csv')


# In[9]:


df.sample(5)


# In[10]:


df=df.drop(columns=['Id'])


# In[11]:


df.sample(5)


# In[12]:


df.describe()


# In[13]:


df.info()


# In[14]:


df['Species'].value_counts()


# # Preprocessing the Data

# In[15]:


df.isnull().sum()


# # Exploratory Data Analysis

# In[16]:


df['SepalLengthCm'].hist()


# In[17]:


df['SepalWidthCm'].hist()


# In[18]:


df['PetalLengthCm'].hist()


# In[19]:


df['PetalWidthCm'].hist()


# In[20]:


#scatterplot
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[21]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[22]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[23]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[24]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# In[25]:


df.corr()['SepalLengthCm']


# In[26]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# In[27]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[28]:


df['Species']=le.fit_transform(df['Species'])
df.head()


# In[30]:


from sklearn.model_selection import train_test_split
X=df.drop(columns=['Species'])
Y=df['Species']


# In[32]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


# In[41]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[45]:


clf=LogisticRegression()
model=clf.fit(X_train,Y_train)
result=model.predict(X_test)
Percentage=accuracy_score(result,Y_test)
print('Accuracy is ',Percentage*100)


# In[47]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
model=clf.fit(X_train,Y_train)
result=model.predict(X_test)
Percentage=accuracy_score(result,Y_test)
print('Accuracy is',Percentage*100)


# In[51]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
model=clf.fit(X_train,Y_train)
result=model.predict(X_test)
Percentage=accuracy_score(result,Y_test)
print(Percentage*100)


# In[ ]:


from sklearn import 

