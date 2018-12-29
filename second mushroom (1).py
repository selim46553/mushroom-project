
# coding: utf-8

# In[2]:


cd /Users/SelimBilgin/Desktop/denemeler/


# In[3]:


import pandas as pd
mantar= pd.read_csv("mushrooms.csv")


# In[4]:


import numpy as np


# In[5]:


mantar.isnull().sum()


# In[6]:


X = mantar.drop("class", axis= 1)


# In[8]:


Y = mantar["class"]


# In[10]:


Y.head()


# In[11]:


X.head()


# In[12]:


X.columns


# In[15]:


from sklearn.preprocessing import LabelEncoder


# In[17]:


for i in X.columns:
    X[i] = LabelEncoder().fit_transform(X[i])
    Y = LabelEncoder().fit_transform(Y)


# In[18]:


Y


# In[20]:


X.head()


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)


# In[22]:


from sklearn.tree import DecisionTreeClassifier


# In[24]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(X_train,Y_train)


# In[27]:


sonuc= classifier.predict(X_test)


# In[28]:


print(sonuc)


# In[29]:


from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(sonuc, Y_test)


# In[30]:


print(val_mae)

