#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('diabetes.csv')
df.head()


# In[3]:


#How big is the data
#Count number of rows and columns
df.shape


# In[4]:


#what is the data type of columns
df.info()


# In[5]:


#Are there any missing value
df.isnull().sum()


# In[6]:


##How does the data look mathematically
#getting statistical data
df.describe()


# In[7]:


#Are the any duplicated values
df.duplicated().sum()


# In[8]:


#count the number of diabetic patients
#0----> Non-Diabetic
#1----> Diabetic
df['Outcome'].value_counts()


# In[9]:


df.groupby('Outcome').mean()


# In[10]:


sns.countplot(df['Outcome'],label='count')


# In[11]:


plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:7].corr(),annot=True,fmt='.0%')


# In[12]:


#separating data and labels
X=df.drop(columns='Outcome',axis=1)
Y=df['Outcome']


# In[13]:


print(X)


# In[14]:


print(Y)


# In[15]:


#Data Standarization


# In[16]:


scaler=StandardScaler()


# In[17]:


scaler.fit(X)


# In[18]:


standardized_data=scaler.transform(X)


# In[19]:


print(standardized_data)


# In[20]:


X=standardized_data
Y=df['Outcome']


# In[21]:


print(X,Y)


# In[22]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[23]:


print(X.shape,X_train.shape,X_test.shape)


# In[24]:


#Training the model


# In[25]:


from sklearn import svm
clf=svm.SVC(kernel='linear')


# In[26]:


clf.fit(X_train,Y_train)


# In[27]:


from sklearn.metrics import accuracy_score


# In[28]:


#accuracy on the training data
X_train_predicition=clf.predict(X_train)
training_data_accuracy=accuracy_score(X_train_predicition,Y_train)


# In[29]:


print('Accuracy score of the training data',training_data_accuracy*100)


# In[30]:


#accuracy on the test data
X_test_predicition=clf.predict(X_test)
test_data_accuracy=accuracy_score(X_test_predicition,Y_test)


# In[31]:


print('Accuracy score of the test data',test_data_accuracy*100)


# # Making a predicitve system

# In[37]:


input_data=(4,110,92,0,0,37.6,0.191,30)

#changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the arrayas we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#standarize the input data
std_data=scaler.transform(input_data_reshaped)

print(std_data)

prediction=clf.predict(std_data)
print(prediction)


# In[ ]:




