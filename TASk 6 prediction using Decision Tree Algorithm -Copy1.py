#!/usr/bin/env python
# coding: utf-8

# # sahaya henisha malar TSF task 6 mar 2021
# #Prediction using Decision Tree algorithm
# #create the decision tree classifier and visualize it graphically.
# 
# 

# In[1]:


#importing all the requried libraries
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import seaborn as sns
import matplotlib.pyplot as mt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# # loading data set
# 

# In[17]:


df = pd.read_csv('F:\Iris.csv')
df.head()


# # Data Inspection

# In[18]:


df.info()


# In[19]:


df.describe().T


# In[20]:


df.describe()


# # input data visualization
# 

# In[22]:


sns.pairplot(df, hue='Species')


# ## we can observe the speciesv "iris Setosa" makes a distinctive cluster in every parameter ,while other two species overlap a bit each other
# 
#   

# # finding  the correlation matrix
# 

# In[23]:


df.corr()


# In[24]:


sns.heatmap(df.corr())

#using heatmap to visulaize data


# In[ ]:


#We observed that: (i)Petal length is highly related to petal width (ii)Sepal length is not related to sepal width


# # data cleaning

# In[25]:


df.isnull().sum()


# In[26]:


df.drop('Id', axis=1, inplace=True)


# # data preprocessing

# In[28]:


target=df['Species']
df1=df.copy()
df1=df1.drop('Species', axis=1)
df1.shape


# In[29]:


#defingi the attributes and labels
X=df.iloc[:, [0,1,2,3]].values
le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])
y=df['Species'].values
df.shape


# # Training the model

# In[30]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Traingin split:",X_train.shape)
print("Testin spllit:",X_test.shape)


# In[ ]:


##we split the data into train and test 


# In[ ]:


##defing Decision Tree algorithm


# In[31]:


dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
print("Decision Tree Classifier created!")


# # Classification Report and Confusion Matrix
# 

# In[32]:


y_pred=dtree.predict(X_test)
print("Classification report:\n",classification_report(y_test,y_pred))


# In[33]:


print("Accuracy:",sm.accuracy_score(y_test,y_pred))


# In[ ]:


##The accuracy is 1 or 100% since i took all the 4 features of the iris dataset.


# In[34]:


#confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm


# # Visualization of trained model

# In[35]:


#visualizing the graph
mt.figure(figsize=(20,10))
tree=plot_tree(dtree,feature_names=df.columns,precision=2,rounded=True,filled=True,class_names=target.values)


# In[ ]:


##The Descision Tree Classifier is created and is visaulized graphically. Also the prediction was calculated using decision tree algorithm and accuracy of the model was evaluated.

