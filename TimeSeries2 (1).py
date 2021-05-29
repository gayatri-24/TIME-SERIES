#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
A = pd.read_csv("C:/Users/Gayatri/Downloads/AirPassengers.csv")


# In[2]:


A.head(2)


# In[3]:


A.Date = pd.to_datetime(A.Month,format="%Y-%m")


# In[4]:


A.head(2)


# In[5]:


A.index=A.Date


# In[6]:


A.head(2)


# In[7]:


A = A.drop(labels=["Month"],axis=1)


# In[8]:


A.head(2)


# In[9]:


A.plot()


# In[10]:


from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt
fig = seasonal_decompose(A,model="multiplicative")
fig.plot()
plt.show()


# In[11]:


A


# In[13]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit_transform(A[['#Passengers']])


# In[14]:


A[["#Passengers"]].mean()


# In[15]:


import numpy as np
log_pass = np.log(A[["#Passengers"]])
rm = log_pass.rolling(window=12).mean()
A['diff']=log_pass-rm


# In[17]:


B=  A[A['diff'].isnull()==False]


# In[18]:


B=B.drop(labels=["#Passengers"],axis=1)


# In[19]:


B.columns=["Pass"]


# In[20]:


B.head(2)


# In[21]:


from statsmodels.tsa.stattools import adfuller
x = adfuller(pd.Series(B['Pass']))
if(x[1]<0.05):
    print("Stationary")
else:
    print("Not Stationary")


# In[22]:


from statsmodels.tsa.stattools import adfuller
x = adfuller(pd.Series(A['#Passengers']))
if(x[1]<0.05):
    print("Stationary")
else:
    print("Not Stationary")


# In[26]:


trd = B[B.index.year<=1958]
tsd = B[B.index.year>1958]

from statsmodels.tsa.ar_model import AR
model = AR(trd).fit()
pred = model.predict(start="1959-01-01",end="1960-12-01")
tsd['Forecasted_from_AR']=pred


# In[27]:


tsd


# In[ ]:




