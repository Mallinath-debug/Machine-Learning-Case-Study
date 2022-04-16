#!/usr/bin/env python
# coding: utf-8

# 
# # IMPORT IMPORTANT LIBRARIES

# In[1]:


#COLUMN INFO
#1)Protein 1 -The activator protein-1 (AP-1) transcription factor is believed to be important in tumorigenesis and altered AP-1 activity was associated with cell transformation.
#2)Protein 2 -Abstract. BMP-2 is involved in the fetal and postnatal development of the mammary gland but has also been detected in breast cancer cells.
#3)Protein 3 - Metastasis-associated protein 3 (MTA3) is a cell type-specific subunit of the Mi-2/NuRD transcriptional corepressor complex. 
#4) Protein 4- HER2-positive breast cancer is a breast cancer that tests positive for a protein called human epidermal growth factor receptor 2 (HER2)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


breast_cancer=pd.read_csv('breast_cancer.csv')


# In[4]:


breast_cancer.head()


# In[5]:


breast_cancer.sample(5)


# # BASIC QUESTIONS WE SHOULD AS K TO DATASET

# In[6]:


#How does the data look like
breast_cancer.shape


# In[7]:


#Information of every columns
breast_cancer.columns


# In[8]:


#Information of the data set
breast_cancer.info()


# In[9]:


#Describe the data set
breast_cancer.describe()


# In[10]:


#Are there any null values
breast_cancer.isnull().sum()


# In[11]:


breast_cancer.dropna(inplace=True)
#dropna removes the rows tha contain null values


# In[12]:


breast_cancer.isnull().sum()


# In[13]:


#Unique values
breast_cancer.nunique()


# In[14]:


breast_cancer.Gender.unique()


# In[15]:


breast_cancer.Gender.value_counts()


# In[16]:


#USING MATPLOTLIB AND SEABORN LIBRARY TO PLOT GRAPH BETWEEN MALE AND FEMALE


# In[17]:


plt.figure(figsize=(15,6))
sns.countplot('Gender',data=breast_cancer)
plt.xticks(rotation=0)
plt.show()


# In[18]:


#Plotting graph for ages and checking which age has higher number of chances of getting breast cancer


# In[19]:


bins=list(range(20,105,5))
plt.figure(figsize=(8,5))
plt.hist(breast_cancer['Age'].astype(int),width=4,align='mid',bins=bins,color='blue',edgecolor='black')
plt.xticks(bins)
plt.xlabel('Ages')
plt.title('Ages in dataset')
plt.yticks(np.arange(0,65,5))
plt.show()


# In[20]:


breast_cancer.Histology.unique()


# In[21]:


#TYPES OF BREAST CANCER
#1)Infiltrating Ductal Carcinoma - Very Common
#2)Mucinous Carcinoma - Very Very Rare
#3) Infiltrating Lobular Carcinoma - Very Rare


#Infiltrating Ductal Carcinoma -Invasive ductal carcinoma (IDC) begins in the lining of a breast duct (milk duct) and spreads outside the duct to other tissues in the breast. It can also spread through the blood and lymph system to other parts of the body. IDC is the most common type of invasive breast cancer.
#The five-year survival rate for localized invasive ductal carcinoma is high — nearly 100% when treated early on. If the cancer has spread to other tissues in the region, the five-year survival rate is 86%
#Mucinous Carcinoa - Mucinous carcinomas, defined as tumors in which at least 50% of the cells are mucinous, can occur in the endometrium and are similar to those that arise from the endocervix.
#Infiltrating Lobular Carcinoma - Invasive lobular carcinoma is a type of breast cancer that begins in the milk-producing glands (lobules) of the breast. Invasive cancer means the cancer cells have broken out of the lobule where they began and have the potential to spread to the lymph nodes and other areas of the body.


# In[22]:


breast_cancer.Histology.value_counts()


# In[23]:


plt.figure(figsize=(15,6))
sns.countplot('Histology',data=breast_cancer)
plt.xticks(rotation=0)
plt.show()


# In[24]:


#Tumour Stage


# In[25]:


breast_cancer.Tumour_Stage.unique()


# In[26]:


breast_cancer.Tumour_Stage.value_counts()


# In[27]:


plt.figure(figsize=(15,6))
sns.countplot('Tumour_Stage',data=breast_cancer)
plt.xticks(rotation=0)
plt.show()


# In[28]:


breast_cancer_type_stage=(breast_cancer.groupby(['Histology','Tumour_Stage'],as_index=False).agg(Total=('Age','count')))


# In[29]:


breast_cancer_type_stage


# In[30]:


plt.figure(figsize=(15,6))
sns.barplot(x='Histology',hue='Tumour_Stage',y='Total',data=breast_cancer_type_stage)
plt.xticks(rotation=0)
plt.show()


# In[31]:


breast_cancer['Age']=pd.cut(breast_cancer['Age'],bins=5)


# In[32]:


breast_cancer['Age'].head()


# In[33]:


plt.figure(figsize=(15,6))
sns.countplot(x='Age',hue='Tumour_Stage',data=breast_cancer)
plt.xticks(rotation=0)
plt.show()


# In[34]:


plt.figure(figsize=(15,6))
sns.countplot(x='Age',hue='Histology',data=breast_cancer)
plt.xticks(rotation=0)
plt.show()


# In[35]:


protein_types=breast_cancer[['Protein1','Protein2','Protein3','Protein4']]


# In[36]:


for i in protein_types.columns:
    sns.boxplot(x=protein_types[i],orient='h',palette='Set2')
    plt.show()


# In[37]:


breast_cancer_type_protein=breast_cancer[['Histology','Protein1','Protein2','Protein3','Protein4']]


# In[38]:


breast_cancer_type_protein.head()


# In[39]:


plt.figure(figsize=(15,6))
sns.barplot(x='Histology',y='Protein1',data=breast_cancer_type_protein)
plt.xticks(rotation=0)
plt.show()


# In[40]:


plt.figure(figsize=(15,6))
sns.barplot(x='Histology',y='Protein2',data=breast_cancer_type_protein)
plt.xticks(rotation=0)
plt.show()


# In[41]:


plt.figure(figsize=(15,6))
sns.barplot(x='Histology',y='Protein3',data=breast_cancer_type_protein)
plt.xticks(rotation=0)
plt.show()


# In[42]:


plt.figure(figsize=(15,6))
sns.barplot(x='Histology',y='Protein4',data=breast_cancer_type_protein)
plt.xticks(rotation=0)
plt.show()


# In[43]:


breast_cancer_stage_protein=breast_cancer[['Tumour_Stage','Protein1','Protein2','Protein3','Protein4']]


# In[44]:


breast_cancer_stage_protein.head()


# In[45]:


plt.figure(figsize=(15,6))
sns.barplot(x='Tumour_Stage',y='Protein1',data=breast_cancer_stage_protein)
plt.xticks(rotation=0)
plt.show()


# In[46]:


plt.figure(figsize=(15,6))
sns.barplot(x='Tumour_Stage',y='Protein2',data=breast_cancer_stage_protein)
plt.xticks(rotation=0)
plt.show()


# In[47]:


plt.figure(figsize=(15,6))
sns.barplot(x='Tumour_Stage',y='Protein3',data=breast_cancer_stage_protein)
plt.xticks(rotation=0)
plt.show()


# In[48]:


plt.figure(figsize=(15,6))
sns.barplot(x = 'Tumour_Stage',y = 'Protein4', data = breast_cancer_stage_protein)
plt.xticks(rotation=0)
plt.show()


# In[49]:


breast_cancer_age_protein=breast_cancer[['Age','Protein1','Protein2','Protein3','Protein4']]


# In[50]:


breast_cancer_age_protein.head()


# In[51]:


plt.figure(figsize=(15,6))
sns.barplot(x='Age',y='Protein1',data=breast_cancer_age_protein)
plt.xticks(rotation=0)
plt.show()


# In[52]:


plt.figure(figsize=(15,6))
sns.barplot(x='Age',y='Protein2',data=breast_cancer_age_protein)
plt.xticks(rotation=0)
plt.show()


# In[53]:


plt.figure(figsize=(15,6))
sns.barplot(x='Age',y='Protein3',data=breast_cancer_age_protein)
plt.xticks(rotation=0)
plt.show()


# In[54]:


#ER -> Estrogen receptor (ER) positive. The cells of this type of breast cancer have receptors that allow them to use the hormone estrogen to grow. Treatment with anti-estrogen hormone (endocrine) therapy can block the growth of the cancer cells.\
#Meaning of ER

#Meaning of pr
#PR -> If breast cancer cells have progesterone receptors, the cancer is called PR-positive breast cancer. 


# In[55]:


breast_cancer_markers=breast_cancer[['Histology','ER status','PR status','HER2 status']]


# In[56]:


breast_cancer_markers.head()


# In[57]:


breast_cancer_markers['ER status'].unique()


# In[58]:


breast_cancer_markers['ER status'].value_counts()


# In[59]:


breast_cancer_markers['PR status'].value_counts()


# In[60]:


breast_cancer_markers['HER2 status'].value_counts()


# In[61]:


plt.figure(figsize=(15,6))
sns.countplot(x='ER status',hue='Patient_Status',data=breast_cancer)
plt.xticks(rotation=0)
plt.show()


# In[62]:


plt.figure(figsize=(15,6))
sns.countplot(x='PR status',hue='Patient_Status',data=breast_cancer)
plt.xticks(rotation=0)
plt.show()


# In[63]:


plt.figure(figsize=(15,6))
sns.countplot(x='HER2 status',hue='Patient_Status',data=breast_cancer)
plt.xticks(rotation=0)
plt.show()


# In[64]:


plt.figure(figsize=(15,6))
sns.countplot(x='Age',hue='Patient_Status',data=breast_cancer)
plt.xticks(rotation=0)
plt.show()


# In[65]:


plt.figure(figsize=(15,6))
sns.countplot(x='Tumour_Stage',hue='Patient_Status',data=breast_cancer)
plt.xticks(rotation=0)
plt.show()


# In[66]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()


# In[67]:


breast_cancer['Tumour_Stage']=label_encoder.fit_transform(breast_cancer['Tumour_Stage'])


# In[68]:


breast_cancer['Histology']=label_encoder.fit_transform(breast_cancer['Histology'])


# In[69]:


breast_cancer['ER status']=label_encoder.fit_transform(breast_cancer['ER status'])


# In[70]:


breast_cancer['PR status']=label_encoder.fit_transform(breast_cancer['PR status'])


# In[71]:


breast_cancer['HER2 status']=label_encoder.fit_transform(breast_cancer['HER2 status'])


# In[72]:


breast_cancer['Surgery_type']=label_encoder.fit_transform(breast_cancer['Surgery_type'])


# In[73]:


breast_cancer['Patient_Status']=label_encoder.fit_transform(breast_cancer['Patient_Status'])


# In[74]:


x=breast_cancer.drop(['Patient_ID','Age','Gender','Date_of_Surgery','Date_of_Last_Visit','Patient_Status'],axis=1)
y=breast_cancer.Patient_Status


# In[75]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[76]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[77]:


model=LogisticRegression()
model.fit(X_train,y_train)


# In[78]:


y_pred=model.predict(X_test)


# In[79]:


print("Accuracy Of Training is :",model.score(X_train,y_train)*100)
print("Accuracy Of Testing is :",model.score(X_test,y_test)*100)


# In[ ]:




