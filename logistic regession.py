#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#getting data

input_data=pd.read_csv("C:/Users/boobathi/Desktop/DATA SCIENCE/supervised learning/logistic regressin/train.csv")
print(input_data.shape)
print(input_data.head())

print("No of passanger travelled in ship:",str(len(input_data.index)))


# In[4]:


#visualizing survived rate

sns.countplot(x="Survived",data=input_data)
plt.show()
#


# In[5]:


#viewing survived rate for each sex

sns.countplot(x="Survived",hue="Sex",data=input_data)
plt.show()
#


# In[6]:


#viewing survived rate for pcclass

sns.countplot(x="Survived",hue="Pclass",data=input_data)
plt.show()
#


# In[11]:


#viewing age using histogram

plt.hist(input_data['Age'])
plt.xlabel('age')
plt.ylabel('density')
plt.show()
#


# In[12]:


#viewing null values

print(input_data.isnull())
print(input_data.isnull().sum())
#


# In[19]:


#viewing null values through heatmap

sns.heatmap(input_data.isnull(),yticklabels=False)
plt.show()
#


# In[20]:


#dropping cabin

input_data.drop("Cabin",axis=1,inplace=True)
print(input_data.head())
#


# In[21]:


#dropping null values

input_data.dropna(inplace=True)
print(input_data.head())
#


# In[ ]:


print(input_data.isnull())
#


# In[22]:


sns.heatmap(input_data.isnull(),yticklabels=False)
plt.show()
print(input_data.isnull().sum())
#


# In[23]:


print(input_data.head())
#


# In[24]:


#setting dummy values

sex=pd.get_dummies(input_data["Sex"],drop_first=True)
print(sex.head())
#


# In[25]:


pclass=pd.get_dummies(input_data["Pclass"],drop_first=True)
print(pclass.head())


# In[26]:


embarked=pd.get_dummies(input_data["Embarked"],drop_first=True)
print(embarked.head())
#


# In[27]:


input_data=pd.concat([input_data,sex,pclass,embarked],axis=1)
print(input_data.head())
#


# In[28]:


input_data.drop(["PassengerId","Pclass","Name","Ticket","Embarked","Sex"],axis=1,inplace=True)
print(input_data.head())
x=input_data.drop("Survived",axis=1)
y=input_data["Survived"]
#input_data.drop(["PassengerId","Pclass","Name","Ticket","Embarked","Sex"],axis=1,inplace=True)
print(input_data.head())
x=input_data.drop("Survived",axis=1)
y=input_data["Survived"]
#


# In[29]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
#
from sklearn.metrics import classification_report
a=classification_report(y_test,predictions)
print(a)
#
from sklearn.metrics import confusion_matrix
b=confusion_matrix(y_test,predictions)
print(b)
#
from sklearn.metrics import accuracy_score
c=accuracy_score(y_test,predictions)
print(c)


# In[ ]:




