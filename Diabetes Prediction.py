#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import svm 


# ## EDA

# In[2]:


dd=pd.read_csv(r"C:\Users\rpsie\Downloads\Projects\ML Projects\diabetes_data.csv")
dd.head(10)


# In[3]:


dd.shape


# In[4]:


dd.info()


# In[5]:


dd.describe()


# In[6]:


dd['Outcome'].value_counts()


# In[7]:


result=dd.groupby('Outcome').mean()
result


# In[8]:


x=dd.drop("Outcome",axis=1)
y=dd["Outcome"]
print(x,y)


# In[9]:


plt.figure(figsize=(12,5))
plt.bar(dd['Age'],dd['Glucose'],color='red')
plt.title('Glucose vs Age')
plt.xlabel('Age')
plt.ylabel('Glucose')
plt.show()


# In[10]:


plt.scatter(dd['BMI'],dd['Insulin'],color='g',alpha=0.5)
plt.show()


# In[11]:


plt.figure(figsize=(12,5))
plt.bar(dd['Pregnancies'],dd['BloodPressure'],color='orange')
plt.show()


# In[12]:


sns.barplot('Outcome','DiabetesPedigreeFunction',data=dd)
plt.show()


# ## Model Training

# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40,stratify=y)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(x_test))


# In[15]:


model=svm.SVC()
model.fit(x_train,y_train)


# In[18]:


from sklearn.metrics import accuracy_score,classification_report
y_train_pred=model.predict(x_train)
accuracy=accuracy_score(y_train_pred,y_train)
print('accuracy of training set:',accuracy)
y_test_pred=model.predict(x_test)
accuracy=accuracy_score(y_test_pred,y_test)
print('accuracy of testing set:',accuracy)
report=classification_report(y_train_pred,y_train)
print('report of training set:',report)
report=classification_report(y_test_pred,y_test)
print('report of testing set:',report)


# ## Check Model Prediction

# In[36]:


# reshape is done such that to told the model that we only want to predict only for one given input data as it is train for whole data set.
input=(2,141,58,34,128,25.4,0.699,24)
array=np.array(input)
reshaped_array=array.reshape(1,-1)
prediction=model.predict(reshaped_array)
print(prediction)

if (prediction==0):
    print('Person is non_dibetic')
else:
    print('Person is dibetic')


# In[ ]:




