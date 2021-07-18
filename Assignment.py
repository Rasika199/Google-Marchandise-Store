#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Importing the Libraries
import pandas as pd
import numpy as np

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import pydotplus
from IPython.display import Image


# In[37]:


# Loading the data
data = pd.read_csv('C:/Users/Desktop/sample_user_data.csv')


# In[38]:


# first five rows of the data
data.head()


# In[39]:


#Looking for imbalance in the data
data.info()


# In[40]:


# Summary of the data
data.describe()


# In[41]:


# Checking the missing values in the data
data.isnull().sum()


# In[42]:


# Deleting the fullVisitorID, date and mobileDeviceModelbecause these variables are not required to analysis.
data = data.drop(['fullVisitorId','Date','mobileDeviceModel'],axis = 1)


# In[43]:


# Converting the VisitStartTime in date and time
data['VisitStartTime'] = pd.to_datetime(data['VisitStartTime'],unit='s')
data.head()


# In[44]:


# Extracting year, month, day, hour, minute from date
data['year']=data['VisitStartTime'].dt.year
data['month']=data['VisitStartTime'].dt.month
data['day']=data['VisitStartTime'].dt.day
data['hour']=data['VisitStartTime'].dt.hour
data['minute']=data['VisitStartTime'].dt.minute


# In[45]:


# Filling the missing values from the data
# if the bounces = 1, there will be no transaction; else a transaction is made. So we replace NaN with 0
data['bounces'].fillna(0,inplace = True)
# if the pageviews = NaN, there will be no transaction; else a transaction is made. So we replace NaN with 0
data['pageviews'].fillna(0,inplace = True)
# if the timeOnSite = NaN, there will be no transaction; else a transaction is made. So we replace NaN with 0
data['timeOnSite'].fillna(0,inplace = True)


# In[46]:


data.head()


# In[47]:


# Count the city names
data['city'].value_counts()


# In[48]:


# Replacing "not available in demo dataset" and "(not set)" by "Unavailable"
data['city'].replace('not available in demo dataset', 'Unavailable',inplace=True)
data['city'].replace('(not set)', 'Unavailable',inplace=True)
data.head()


# In[49]:


# totalTransactionRevenue and transactionns are shows NaN values, so we replace NaN with 0
data['totalTransactionRevenue'].fillna(0,inplace=True)
data['transactions'].fillna(0,inplace=True)


# In[50]:


# Target variable is 'transaction', so 1 = transaction, 0 = no transaction
data['transaction'] = np.where(data['transactions'] == 0,'0','1')
data['transaction'] = data['transaction'].astype('str')


# In[51]:


# TransactionRevenue is totalTransactionRevenue / 1000000
data['transactionRevenue'] = data['totalTransactionRevenue']/1000000


# In[52]:


data.head()


# In[53]:


# Removing the variables which is not needed
data = data.drop(['VisitStartTime','totalTransactionRevenue','year'],axis = 1)


# In[54]:


data.head()


# In[55]:


# Plotting the correlation matrix
corr = plt.figure(figsize=(10,10));
plt.title('Correlation Matrix')
data.corr()
sns.heatmap(data.corr(),vmin=-1,vmax=1,cmap='Blues',annot=True);


# In[56]:


data.info()


# In[57]:


# Conveting the categorical valriable into interger using label encoder
labEnc = LabelEncoder()
cols = ["source","medium","campaign","operatingSystem","city","ChannelGrouping"]
label = ["deviceCategory"]

for col in cols:
    data[col]=labEnc.fit_transform(data[col])

data = pd.get_dummies(data,columns = label)


# In[58]:


data.info()


# In[59]:


# Separating independent and dependent variables
features = ["VisitNumber","bounces","pageviews","timeOnSite","source","medium","campaign","operatingSystem",
           "city","ChannelGrouping","month","day","hour","deviceCategory_desktop","deviceCategory_mobile"]
X = data[features]
y = data.transaction


# In[60]:


# Creating the train and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[61]:


# Defining the Decision Tree algorithm
dtree=DecisionTreeClassifier(max_depth=8, max_leaf_nodes = 25, random_state=10)
dtree.fit(X_train,y_train)
print("Decision Tree Classifier Created")


# In[62]:


# Predicting the values of test data
predictions=dtree.predict(X_test)
predictions


# In[63]:


# Constructing the confusion matrix and Accuracy of the model
print("Accuracy is:", metrics.accuracy_score(y_test,predictions))
print("confusion_matrix :-\n",confusion_matrix(y_test,predictions))


# In[64]:


decision_tree = tree.export_graphviz(dtree,out_file='decision.dot',feature_names=features,class_names = ['0','1'],filled=True)


# In[65]:


get_ipython().system('dot -Tpng decision.dot -o decision.png')


# In[66]:


# Viwe the image
image = plt.imread('decision.png')
plt.figure(figsize = (30,30))
plt.imshow(image)

