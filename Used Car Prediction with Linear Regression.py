#!/usr/bin/env python
# coding: utf-8

# In[234]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[235]:


data0=pd.read_csv("Used Car Info.csv")


# In[236]:


data0.describe(include="all")


# In[101]:


# eliminate the column that is relatively irrelevant and difficult to analyze


# In[237]:


data1=data0.drop(["Model"],axis=1)


# In[238]:


data1.isnull().sum()


# In[239]:


data2=data1.dropna(axis=0)


# In[240]:


data2.describe(include="all")


# In[241]:


# deal with each individual column


# In[246]:


sns.distplot(data2["Price"])


# In[247]:


q1=data2["Price"].quantile(0.95)


# In[248]:


data3=data2[data2["Price"]<q1]


# In[249]:


sns.distplot(data3["Price"])


# In[250]:


sns.distplot(data3["Mileage"])


# In[251]:


q2=data3["Mileage"].quantile(0.95)


# In[252]:


data4=data3[data3["Mileage"]<q2]


# In[253]:


sns.distplot(data4["Mileage"])


# In[254]:


sns.distplot(data4["EngineV"])


# In[255]:


data5=data4[data4['EngineV']<5]


# In[256]:


sns.distplot(data5["EngineV"])


# In[257]:


sns.distplot(data5["Year"])


# In[258]:


q3=data5["Year"].quantile(0.05)


# In[259]:


data6=data5[data5["Year"]>q3]


# In[260]:


sns.distplot(data6["Year"])


# In[261]:


data6.describe(include="all")


# In[262]:


# create the relation chart between column "price" and other three Numeric columns: "Year", "EngineV", "Mileage"


# In[263]:


data6.describe(include="all")


# In[264]:


# create relation chart that between column "Price" and other three that exclude "Brand"


# In[265]:


f, (ax1, ax2, ax3)=plt.subplots(1,3, sharey=True, figsize=(15,3))
ax1.scatter(data6["Year"],data6["Price"])
ax1.set_title("Year and Price")
ax2.scatter(data6["EngineV"],data6["Price"])
ax2.set_title("EngineV and Price")
ax3.scatter(data6["Mileage"],data6["Price"])
ax3.set_title("Mileage and Price")
plt.show()


# In[266]:


# revise the relation charts


# In[267]:


log_price=np.log(data6["Price"])
data6["Log Price"]=log_price


# In[268]:


f, (ax1, ax2, ax3)=plt.subplots(1,3, sharey=True, figsize=(15,3))
ax1.scatter(data6["Year"],data6["Log Price"])
ax1.set_title("Year and Log Price")
ax2.scatter(data6["EngineV"],data6["Log Price"])
ax2.set_title("EngineV and Log Price")
ax3.scatter(data6["Mileage"],data6["Log Price"])
ax3.set_title("Mileage and Log Price")
plt.show()


# In[269]:


data7=data6.drop(["Price"],axis=1)


# In[270]:


data7.describe(include="all")


# In[271]:


# check multicollinearity


# In[272]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[273]:


var=data7[["Mileage","EngineV","Year"]]
vif=pd.DataFrame()
vif["VIF"]=[variance_inflation_factor(var.values,i) for i in range (var.shape[1])]
vif["Columns"]=var.columns


# In[274]:


vif


# In[275]:


# deal with the categorical values 


# In[276]:


data8=pd.get_dummies(data7,drop_first=True)


# In[277]:


data8.describe(include="all")


# In[278]:


data8.columns.values


# In[280]:


# prepare linear regression 


# In[281]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# In[282]:


y=data8["Log Price"]


# In[283]:


x=data8.drop(["Log Price"],axis=1)


# In[284]:


scaler=StandardScaler()


# In[285]:


scaler.fit(x)


# In[286]:


x_scaled=scaler.transform(x)


# In[287]:


# create the train/test set


# In[288]:


from sklearn.model_selection import train_test_split


# In[289]:


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=365)


# In[290]:


# create the linear regression


# In[291]:


reg=LinearRegression()


# In[292]:


reg.fit(x_train,y_train)


# In[293]:


reg.score(x_train,y_train)


# In[294]:


reg.intercept_


# In[295]:


reg.coef_


# In[296]:


reg_summary=pd.DataFrame()
reg_summary["Weights"]=reg.coef_
reg_summary["Features"]=x.columns.values
reg_summary


# In[297]:


# validate the model by testing


# In[298]:


y_hat=reg.predict(x_test)


# In[299]:


sns.distplot(y_test-y_hat)


# In[300]:


y_test=y_test.reset_index(drop=True)


# In[306]:


test_summary=pd.DataFrame()
test_summary["Prediction"]=np.exp(y_hat)
test_summary["Real Result"]=
test_summary["Residual"]=np.exp(y_test)-np.exp(y_hat)


# In[310]:


test_summary["Difference%"]=np.absolute(test_summary["Residual"]/np.exp(y_test)*100)


# In[312]:


test_summary.describe()

