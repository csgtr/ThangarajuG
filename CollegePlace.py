#!/usr/bin/env python
# coding: utf-8

# In[67]:


#Importing the Libraries
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC  


# In[ ]:


#Data Collection and Preparation
#Read The Data Set
df=pd.read_csv("E:\\NMDS\collegePlace.csv")
df.head()


# In[3]:


df['Stream'].value_counts()


# In[10]:


#Handling Mising Values
df.info()


# In[2]:


#Handling Mising Values
df.isnull().sum()


# In[13]:


#Handling Categorical Values
df.describe()


# In[12]:


#Handling Outliers
def transformationplot(feature):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.displot(feature)
    transformationplot(np.log(df['Age']))
    plt.show()


# In[14]:


# I tried all the columns and find out that only age column has some outliers.

plt.figure(figsize = (10, 6), dpi = 100)
sns.boxplot(x = "Age", data = df)


# In[15]:


#Removing Outlier
max_thresold = df['Age'].quantile(0.95)
print(max_thresold)

min_thresold = df['Age'].quantile(0.01)
print(min_thresold)

df = df[(df['Age']<max_thresold) & (df['Age']>min_thresold)]


# In[16]:


#Handling Categorical values
#Lable Encoding
df=df.replace(['Male'],[0])
df=df.replace(['Female'],[1])
df=df.replace(['Computer Science','Information Technology','Electronics And Communication','Mechanical','Electrical','Civil'],[0,1,2,4,3,5])
df.drop(['Hostel'],axis=1)


# In[25]:


#Exploratory Data Abnliysis
#Data Visualization count of stream
plt.figure(figsize = (10, 6), dpi = 100)
color_palette = sns.color_palette("Accent_r")
sns.set_palette(color_palette)
sns.countplot(x = "Stream", data = df)


# In[27]:


#Data Visualization count of Internships
plt.figure(figsize = (10, 6), dpi = 100)
color_palette = sns.color_palette("cool")
sns.set_palette(color_palette)
sns.countplot(x = "Internships", data = df)
plt.show()


# In[5]:


#Data Visualization  Distribution of CGPA
plt.figure(figsize = (10, 6), dpi = 100)
grp = dict(df.groupby('CGPA').groups)

m = {}

for key, val in grp.items():
    
    if key in m:
        m[key] += len(val)
        
    else:
        m[key] = len(val)

    
plt.title("Distribution of CGPA")
plt.pie(m.values(), labels = m.keys())
plt.show()


# In[24]:


#Univariate Analysis
df.hist()
plt.show()


# In[31]:


plt.figure(figsize = (10, 6), dpi = 100)
# setting the different color palette
color_palette = sns.color_palette("Accent_r")
sns.set_palette(color_palette)

sns.countplot(x = "Gender", data = df)

plt.show()


# In[33]:


#Relationships
#Multivariate Analysis
plt.figure(figsize = (10, 6), dpi = 100)
# setting the different color palette
color_palette = sns.color_palette("plasma")
sns.set_palette(color_palette)
sns.barplot(x = "PlacedOrNot", y = "Gender", data = df)
plt.show()


# In[33]:


#Multivariate Analysis
plt.figure(figsize = (10, 6), dpi = 100)
# setting the different color palette
color_palette = sns.color_palette("magma")
sns.set_palette(color_palette)
sns.barplot(x = "Stream", y = "PlacedOrNot", data = df)
plt.show()


# In[35]:


# How many placed
plt.figure(figsize = (10, 6), dpi = 100)


# setting the different color palette
color_palette = sns.color_palette("BuGn_r")
sns.set_palette(color_palette)

sns.countplot(x = "PlacedOrNot", data = df)

plt.show()


# In[6]:


#Scaling the Data
#Correllation
plt.figure(figsize = (10, 6), dpi = 100)
color = sns.color_palette("BuGn_r")
sns.heatmap(df.corr(), vmax=0.9, annot=True,cmap = color)


# In[8]:


#Model Building
#mode1:SVM
classifier=svm.SVC(kernel='linear')


# In[18]:


#mode1:SVM
import numpy as np
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.model_selection import train_test_split
 classes = 4
X,t= make_classification(100, 5, n_classes = classes, random_state= 40, n_informative = 2, n_clusters_per_class = 1)
#%%
X_train, X_test, y_train, y_test=  train_test_split(X, t , test_size=0.50)
#%%
model = svm.SVC(kernel = 'linear', random_state = 0, C=1.0)
#%%
model.fit(X_train, y_train)
#%%
y=model.predict(X_test)
y2=model.predict(X_train)
#%%
from sklearn.metrics import accuracy_score
score =accuracy_score(y, y_test)
print(score)
score2 =accuracy_score(y2, y_train)
print(score2)
#%%
import matplotlib.pyplot as plt
color = ['black' if c == 0 else 'lightgrey' for c in y]
plt.scatter(X_train[:,0], X_train[:,1], c=color)
 
# Create the hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (model.intercept_[0]) / w[1]
 
# Plot the hyperplane
plt.plot(xx, yy)
plt.axis("off"), plt.show();


# In[43]:


# Spot-Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
  #  kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, scoring='accuracy')
results.append(cv_results)
names.append(name)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)


# In[63]:


# KNN Classification
#Handling Categorical values
#Lable Encoding
df=pd.read_csv("E:\\NMDS\collegePlace.csv")
df=df.replace(['Male'],[0])
df=df.replace(['Female'],[1])
df=df.replace(['Computer Science','Information Technology','Electronics And Communication','Mechanical','Electrical','Civil'],[0,1,2,4,3,5])
df.drop(['Hostel'],axis=1)
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
#df = read_csv(filename, names=names)
array = df.values
X = array[:,0:7]
Y = array[:,7]
#num_folds = 5
#kfold = KFold(n_splits=5, random_state=3)
model = KNeighborsClassifier()
results = cross_val_score(model,X,Y)
print(results.mean())


# In[66]:


import pikle
import joblib
pickle.dump((knn,open("placement.pkl","wb"))
model=pickle.load(open("placement.pkl","rb")))


# In[ ]:


from flask 


# In[ ]:




