#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


CustOrders = pd.read_csv('/Users/harishsrinivasan/Downloads/stout/casestudy.csv')


# In[3]:


CustOrders.head()


# In[109]:


CustOrders


# In[4]:


CustOrders.describe()


# ## Total Revenue for current year

# In[5]:


CustOrders2015 = CustOrders[CustOrders['year']==2015]
CustOrders2016 = CustOrders[CustOrders['year']==2016]
CustOrders2017 = CustOrders[CustOrders['year']==2017]


# In[8]:


CustOrders2017


# In[9]:


TotRev2015 = CustOrders2015['net_revenue'].sum()
TotRev2016 = CustOrders2016['net_revenue'].sum()
TotRev2017 = CustOrders2017['net_revenue'].sum()


# In[11]:


Revenue = [TotRev2015,TotRev2016,TotRev2017]
RevenueYear = ['2015', '2016', '2017']


# In[12]:


print("Total revenue for the current year: " + str(TotRev2017))


# ## New Customer Revenue

# In[13]:


NewCustOrders2016 = pd.merge(CustOrders2015, CustOrders2016, on='customer_email', how='outer',indicator=True)
NewCustOrders2016 = NewCustOrders2016[NewCustOrders2016['year_y']==2016]
NewCustOrders2016 = NewCustOrders2016[NewCustOrders2016['year_x'].isna()]
NewCustRev2016 = NewCustOrders2016['net_revenue_y'].sum()


# In[14]:


NewCustOrders2016


# In[16]:


print("New Customer Revenue for 2016: " + str(NewCustRev2016)) 


# In[17]:


NewCustOrders2017 = pd.merge(CustOrders2016, CustOrders2017, on='customer_email', how='outer',indicator=True)
NewCustOrders2017 = NewCustOrders2017[NewCustOrders2017['year_y']==2017]
NewCustOrders2017 = NewCustOrders2017[NewCustOrders2017['year_x'].isna()]
NewCustRev2017 = NewCustOrders2017['net_revenue_y'].sum()


# In[19]:


NewCustOrders2017


# In[18]:


print("New Customer Revenue for 2017: " + str(NewCustRev2017)) 


# ## Existing Customer Growth

# In[20]:


NewCustOrders2017 = pd.merge(CustOrders2016, CustOrders2017, on='customer_email', how='outer',indicator=True)


# In[21]:


NewCustOrders2017 = NewCustOrders2017[NewCustOrders2017['_merge']=='both']


# In[22]:


Growth2017 = pd.DataFrame(columns=['customer_email','net_growth'])
Growth2017['net_growth'] = NewCustOrders2017['net_revenue_y'] - NewCustOrders2017['net_revenue_x']
Growth2017['customer_email'] = NewCustOrders2017['customer_email']


# In[23]:


NewCustOrders2017


# In[24]:


Growth2017


# In[25]:


NewCustOrders2016 = pd.merge(CustOrders2015, CustOrders2016, on='customer_email', how='outer',indicator=True)
NewCustOrders2016 = NewCustOrders2016[NewCustOrders2016['_merge']=='both']
NewCustOrders2016


# In[26]:


Growth2016 = pd.DataFrame(columns=['customer_email','net_growth'])
Growth2016['net_growth'] = NewCustOrders2016['net_revenue_y'] - NewCustOrders2016['net_revenue_x']
Growth2016['customer_email'] = NewCustOrders2016['customer_email']


# In[27]:


Growth2016


# ## Revenue lost from attrition

# In[28]:


Attrition2017 = Growth2017['net_growth'].sum()
Attrition2016 = Growth2016['net_growth'].sum()


# In[29]:


Attrition2017


# In[30]:


Attrition2016


# ## Existing Customer Revenue Current Year

# In[31]:


Ex_custrev_2017 = NewCustOrders2017['net_revenue_y'].sum()
Ex_custrev_2016 = NewCustOrders2017['net_revenue_x'].sum()


# In[33]:


print('Existing Customer Revenue Current Year: ' + str(Ex_custrev_2017))


# ## Existing Customer Revenue Prior Year

# In[34]:


print('Existing Customer Revenue Prior Year: ' + str(Ex_custrev_2016))


# ## Total Customers Current Year

# In[37]:


print('Total Customers Current Year: ' + str(len(CustOrders2017))) 


# ## Total Customers Prior Year

# In[38]:


print('Total Customers Prior Year: ' + str(len(CustOrders2016))) 


# ## New Customers

# In[48]:


Custlist2017 = pd.merge(CustOrders2016, CustOrders2017, on='customer_email', how='outer',indicator=True)
Custlist2016 = pd.merge(CustOrders2015, CustOrders2016, on='customer_email', how='outer',indicator=True)


# In[41]:


NewCustlist2017 = Custlist2017[Custlist2017['_merge']=='right_only']


# In[43]:


print('New Customers in 2017: ' + str(len(NewCustlist2017))) 


# In[108]:


NewCustlist2016 = Custlist2016[Custlist2016['_merge']=='right_only']
print('New Customers in 2016: ' + str(len(NewCustlist2016))) 


# ## Lost Customers

# In[51]:


LostCustlist2017 = Custlist2017[Custlist2017['_merge']=='left_only']
LostCustlist2016 = Custlist2016[Custlist2016['_merge']=='left_only']


# In[54]:


print('Lost Customers in 2017: ' + str(len(LostCustlist2017))) 
print('Lost Customers in 2016: ' + str(len(LostCustlist2016))) 


# ## Visualisations

# In[73]:


import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from functools import reduce


# In[64]:


colors = ['yellowgreen', 'gold', 'lightskyblue']
plt.title('Total Revenue per Year')
plt.pie(Revenue, labels=RevenueYear, startangle = 90, radius=2, autopct='%1.1f%%', shadow=True,colors=colors)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show() 


# In[67]:


X = ['2016','2017']
X_axis = np.arange(len(X))
plt.figure(figsize=(10,8))  
plt.bar(X_axis - 0.2, (len(NewCustlist2016), len(NewCustlist2017)), 0.4, label = 'New Customers')
plt.bar(X_axis + 0.2, (len(CustOrders2016), len(CustOrders2017)), 0.4, label = 'Total Customers')
  
plt.xticks(X_axis, X)
plt.xlabel("Years")
plt.ylabel("Number of Customers")
plt.title("Total Customers vs New customers by Year")
plt.legend()
plt.show()


# In[70]:


X = ['2016','2017']
X_axis = np.arange(len(X))
plt.figure(figsize=(10,8))    
plt.bar(X_axis - 0.2, (Ex_custrev_2016, Ex_custrev_2017), 0.4, label = 'Existing Customer Revenue')
plt.bar(X_axis + 0.2, (TotRev2016, TotRev2017), 0.4, label = 'Total Revenue')
  
plt.xticks(X_axis, X)
plt.xlabel("Years")
plt.ylabel("Revenue Earned")
plt.title("Total Revenue vs Existing Customer Revenue by Year")
plt.legend()
plt.show()


# In[72]:


X = ['2016','2017']
X_axis = np.arange(len(X))
plt.figure(figsize=(10,8))  
plt.bar(X_axis - 0.2, (NewCustRev2016, NewCustRev2017), 0.4, label = 'New Customer Revenue')
plt.bar(X_axis + 0.2, (TotRev2016, TotRev2017), 0.4, label = 'Total Revenue')
  
plt.xticks(X_axis, X)
plt.xlabel("Years")
plt.ylabel("Revenue Earned")
plt.title("Total Revenue vs New Customer Revenue by Year")
plt.legend()
plt.show()


# In[92]:


dfs = [CustOrders2015, CustOrders2016, CustOrders2017]
df_common = reduce(lambda left,right: pd.merge(left,right,on='customer_email'), dfs)


# In[93]:


df_common


# In[94]:


LoyalCustRev2015 = df_common['net_revenue_x'].sum()
LoyalCustRev2016 = df_common['net_revenue_y'].sum()
LoyalCustRev2017 = df_common['net_revenue'].sum()


# In[106]:


X = ['2015', '2016','2017']

plt.figure(figsize=(10,8)) 
plt.bar(X, (LoyalCustRev2015, LoyalCustRev2016, LoyalCustRev2017))

plt.xlabel("Years")
plt.ylabel("Revenue Earned")
plt.title("Total Revenue from existing customer for 3 years by Year")
plt.legend()
plt.show()


# In[82]:


CustOrders2015.describe()


# In[83]:


CustOrders2016.describe()


# In[84]:


CustOrders2015.describe()


# In[107]:


plt.figure(figsize=(10,7))
sns.boxplot(x='year',y='net_revenue',data=CustOrders, palette='rainbow')
plt.title("Net_revenue by Year")


# In[101]:


len(Growth2017[Growth2017['net_growth'] < 0])


# In[102]:


len(Growth2017[Growth2017['net_growth'] > 0])


# In[105]:


X = ['2016','2017']
X_axis = np.arange(len(X))
plt.figure(figsize=(10,8))  
plt.bar(X_axis - 0.2, (len(Growth2017[Growth2017['net_growth'] > 0]), len(Growth2016[Growth2016['net_growth'] > 0])), 0.4, label = 'Customers who bought more than last year')
plt.bar(X_axis + 0.2, (len(Growth2017[Growth2017['net_growth'] < 0]), len(Growth2016[Growth2016['net_growth'] < 0])), 0.4, label = 'Customers who bought less than last year')
  
plt.xticks(X_axis, X)
plt.xlabel("Years")
plt.ylabel("Number of Customers")
plt.title("Growth over years")
plt.legend()
plt.show()


# In[ ]:




