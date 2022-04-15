#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


df=pd.read_csv('titanic1.csv')


# In[3]:


df.head()


# In[4]:


#How big is the data
df.shape


# In[5]:


#what is the data type of cols
df.info()


# In[6]:


#Are there any missing values
df.isnull().sum()


# In[7]:


#How does the data look matheatically
df.describe()


# In[8]:


#Are there any duplicated values
df.duplicated().sum()


# In[9]:


df=df.drop(columns='Cabin',axis=1)


# In[10]:


#replacing the missing values in age columns with mean value
df['Age'].fillna(df['Age'].mean(), inplace=True)


# In[11]:


#finding the mode value of embarked column
print(df['Embarked'].mode())


# In[12]:


print(df['Embarked'].mode()[0])


# In[13]:


#replacing the missing values in "Embarked" column with mode value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


# In[14]:


df.isnull().sum()


# # DATA ANALYSIS

# In[15]:


#getting some statistical measures about the data
df.describe()


# In[16]:


#finding the number of people survived and not survived
#0-Dead
#1-Survived
df['Survived'].value_counts()


# # Data Visualization

# In[17]:


sns.countplot('Survived',data=df)


# In[18]:


df['Sex'].value_counts()


# In[19]:


sns.countplot('Sex',data=df)


# In[20]:


sns.countplot('Sex',hue='Survived',data=df)


# In[21]:


sns.countplot('Pclass',data=df)


# In[22]:


sns.countplot('Pclass',hue='Survived',data=df)


# In[23]:


#Encoding the Categorical columns
#converting categorical columns
df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[24]:


df.head()


# In[25]:


#Separating Features and Targets
X=df.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
Y=df['Survived']


# In[26]:


print(X)


# In[27]:


print(Y)


# In[28]:


#Splitting the data into training data and testing data


# In[29]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[30]:


print(X.shape,X_train.shape,X_test.shape)


# In[31]:


#Model Training
#Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[33]:


model=LogisticRegression()
model.fit(X_train,Y_train)


# In[34]:


#accuracy on training data
from sklearn.metrics import accuracy_score
X_train_prediction=model.predict(X_train)


# In[35]:


print(X_train_prediction)


# In[38]:


training_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print('Accuracy score of trainign data:',training_data_accuracy*100)


# In[39]:


#accuracy on testing data

X_test_prediction=model.predict(X_test)


# In[40]:


print(X_test_prediction)


# In[41]:


test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print('Accuracy score of test data:',test_data_accuracy*100)


# In[ ]:




