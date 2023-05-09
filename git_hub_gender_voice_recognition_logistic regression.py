#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np                                                        # For pre-preocessing data
import pandas as pd                                                       # For pre-preocessing data
import matplotlib.pyplot as plt                                           # For visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns                                                     # For visualisation

                      # For training our Logistic Regression model
import scipy.stats as stats                                               # For training our model using Statsmodels
import statsmodels.api as sm                                              # For training our model using Statsmodels
from sklearn.metrics import classification_report,confusion_matrix        # For Performance metrics 
from sklearn.metrics import ConfusionMatrixDisplay                        # For plotting confusion matrix
#from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_validate                        # For cross validation scores
from sklearn.model_selection import cross_val_score                       # For cross validation scores
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
                                                                          # For Performance metrics 
from statsmodels.stats.outliers_influence import variance_inflation_factor 
                                                                          # For Feature Selection
from sklearn.metrics import roc_auc_score                                 # For ROC AUC 
from sklearn.metrics import roc_curve                                     # For plotting ROC 
from sklearn.metrics import precision_recall_curve                        # For plotting Precision and Recall 

import os                                                                 # For changing home directory
from sklearn.model_selection import train_test_split                      # For train test split


pd.set_option('display.max_rows', 250)                                    # to show upto 250 rows in output
pd.set_option('display.max_colwidth',250)                                 # to show upto 250 cols in output
pd.set_option('display.float_format', lambda x: '%.5f' % x)               # customised format for pandas dataframe output


import warnings
warnings.filterwarnings('ignore')                                        # To supress warnings


plt.style.use('ggplot')     


# In[24]:


#Data ingestion
file_path='D:/DS/resume projects/voice recodnition/voice.csv'

voiceor=pd.read_csv(file_path)
voicem=voiceor
# original_data=voiceor.copy()
# print(f'We have {customers_data.shape[0]} rows and {customers_data.shape[1]} columns in the data') # fstring 


# In[25]:


plt.figure(figsize=(10,10))
sns.heatmap(voicem.corr(),annot=True)


# In[26]:


voicem.corr().style.background_gradient("cool")


# In[27]:


voicem


# In[28]:


voicem.isna().sum()


# In[29]:


# making ytest data more model friendly
voicem['label']=voicem['label'].replace({'male':0,'female':1})


# In[30]:


voicem


# In[31]:


#splitting data into 80/20
x=voicem.drop(['label'],axis=1)
y=voicem.label
xtn,xts,ytn,yts=train_test_split(x,y,train_size=0.7)
from sklearn.preprocessing import *
std=StandardScaler()
xstdtn=std.fit_transform(xtn)
xstdftn=pd.DataFrame(xstdtn,columns=x.columns)
xstdts=std.transform(xts)
xstdfts=pd.DataFrame(xstdts,columns=x.columns)


# In[32]:


xstdftn


# In[33]:


from sklearn.linear_model import LogisticRegression 
logr=LogisticRegression()
logr=logr.fit(xstdftn,ytn)
yptn1=logr.predict(xstdftn)
ypts1=logr.predict(xstdfts)


# In[34]:


yp2=logr.predict_proba(xstdfts)  # for eoc curve


# In[35]:


#confusion metrix
cnfmt=confusion_matrix(yts,ypts1)
cnfmt=pd.DataFrame(cnfmt,columns=['Actually mail','Actually female'],index=['Predicted male','Predicted female'])
sns.heatmap(cnfmt,annot=True)
print(cnfmt)


# In[36]:


# accuracy
accn=accuracy_score(ytn,yptn1)
accs=accuracy_score(yts,ypts1)
print('Train',str(accn*100)+'%')
print('test',str(accs*100)+'%')


# In[37]:


from sklearn.metrics import recall_score
gmts=classification_report(yts,ypts1)
gmtn=classification_report(ytn,yptn1)
print('Test:-\n',gmts,'\n','Train:-\n',gmtn)


# In[45]:


#roc curve 
from sklearn.metrics import roc_curve
ypredictonxtest=logr.predict_proba(xstdfts)
ts_y_p_df=pd.DataFrame(ypredictonxtest)
#pd.concat((ts_y_p_df,pd.DataFrame(yts)),axis=1)


# In[41]:


sen,sep,threshold1=roc_curve(yts,ts_y_p_df[1])


# In[43]:


plt.plot(sen,sep,color='green')
plt.plot([0,1],[0,1],'--')
plt.title('ROC curve')
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.show()


# In[ ]:


#Summary
# 1.)4.)Finally we have our  model ready which can predict Gender as per the given input with 97% accuracy

