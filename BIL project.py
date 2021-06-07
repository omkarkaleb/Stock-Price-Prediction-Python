#!/usr/bin/env python
# coding: utf-8

# In[262]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import time
import datetime as dt
import numpy as np
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[263]:


bse = pd.read_csv(r"C:\Users\omkar kale\Downloads\bse.csv")
bse.head()


# In[264]:


bse['Date'] = pd.to_datetime(bse['Date'], format='%Y-%m-%d')


# In[265]:


bse.iloc[::-1]


# In[266]:


bse.describe()


# In[267]:


print ("Rows     : " ,bse.shape[0])
print ("Columns  : " ,bse.shape[1])
print ("\nFeatures : \n" ,bse.columns.tolist())
print ("\nMissing values :  ", bse.isnull().sum().values.sum())
print ("\nUnique values :  \n",bse.nunique())


# In[268]:



bse.info()


# In[333]:


X = bse[['Date','Open','High','Low']]
# X = bse[['Date','Open']]
y = bse['Close']
X_train = X[140:639]
X_test = X[0:139]
y_train = y[140:639]
y_test = y[0:139]


# In[334]:


# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# In[335]:


from sklearn.linear_model import LinearRegression
X_trainn = X_train.apply(pd.to_numeric, errors='coerce')
y_trainn = y_train.apply(pd.to_numeric, errors='coerce')
X_trainn.fillna(0, inplace=True)
y_trainn.fillna(0, inplace=True)
lr = LinearRegression()
lr.fit(X_trainn, y_trainn)


# In[336]:


X_testn = X_test.apply(pd.to_numeric, errors='coerce')
y_testn = y_test.apply(pd.to_numeric, errors='coerce')
X_testn.fillna(0, inplace=True)
y_testn.fillna(0, inplace=True)
prediction = lr.predict(X_testn)
lr_score = format(lr.score(X_testn,y_testn)*100)
print("Accuracy {:.2f}%".format(lr.score(X_testn,y_testn)*100))


# In[337]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 200)
rf.fit(X_trainn, y_trainn)


# In[338]:


pred = rf.predict(X_testn)
rf_score = format(rf.score(X_testn,y_testn)*100)
print("Accuracy {:.2f}%".format(rf.score(X_testn,y_testn)*100))


# In[339]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
X_t = X_trainn.astype('int')
y_t = y_trainn.astype('int')
model.fit(X_t, y_t)


# In[340]:


NB = model.predict(X_testn)
model_score = format(model.score(X_t,y_t)*100)
print("Accuracy {:.2f}%".format(model.score(X_t,y_t)*100))


# In[341]:


lr_score = round(float(lr_score))
rf_score = round(float(rf_score))
model_score = round(float(model_score))
algo = ['Linear regression', 'Random Forest', 'Naive Bayes']
accuracy = [lr_score, rf_score, model_score]
plt.bar(algo, accuracy) 
plt.xlabel('Algorithm') 
plt.ylabel('y')
plt.yticks(np.arange(0, 110, 10))
plt.show()


# In[342]:


x = X_test
plt.figure(figsize=(18,9))
plt.title('Actual vs Predicted')
plt.plot(bse['Date'], bse['Close'], label = "Actual")
plt.plot(x['Date'], prediction, label = "LR")
plt.plot(x['Date'], pred, label = "RF")
plt.plot(x['Date'], NB, label = "NB")
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price in INR', fontsize = 18)
plt.legend()
plt.show()


# In[343]:


# importance = rf.feature_importances_
importance = lr.coef_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.2f' % (i,v))
plt.title('Feature Importance for Linear Regression')
plt.bar([x for x in range(len(importance))], importance)

plt.show()


# In[344]:


importance = rf.feature_importances_
# importance = lr.coef_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.2f' % (i,v))
plt.title('Feature Importance for Random Forest')
plt.bar([x for x in range(len(importance))], importance)

plt.show()


# In[345]:


df = pd.DataFrame({'Date': X_test['Date'], 'Actual': y_test, 'Predicted by LR': prediction, 'Predicted by RF': pred, 'Predicted by NB': NB})
df.iloc[::-1]


# In[332]:


x = bse[['Date','Open']]
x = x[0:139]
x = x.apply(pd.to_numeric, errors='coerce')
x.fillna(0, inplace=True)
plt.figure(figsize=(15,7.5))
plt.scatter(X_test['Open'], y_testn, color = "red")
plt.plot(X_test['Open'], lr.predict(x), color = "green")
plt.title("Visualization for Linear Regression")
plt.show()

