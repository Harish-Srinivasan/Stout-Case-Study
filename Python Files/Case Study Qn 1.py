#!/usr/bin/env python
# coding: utf-8

# In[135]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import plotly.express as px
import seaborn as sns
from matplotlib import rcParams


# In[136]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from itertools import chain, combinations
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV


# In[4]:


df = pd.read_csv("/Users/harishsrinivasan/Downloads/stout/loans_full_schema.csv")


# In[5]:


df


# In[6]:


df.describe()


# # Data Preprocessing

# In[7]:


# Checking null values in columns
df.isnull().sum().sort_values(ascending=False)


# In[19]:


# Checking types for data columns
df.dtypes


# ## Exploratory Data Analysis

# In[110]:


lst1 = df.groupby(['state'])['loan_amount'].mean().values.tolist()
lst2 = df.groupby(['state'])['loan_amount'].mean().keys().tolist()


# In[111]:


df_state_loan = pd.DataFrame(list(zip(lst1, lst2)),
              columns=['Mean Loan Amount', 'state'])


# In[115]:


lst3 = df.groupby(['state'])['interest_rate'].mean().values.tolist()


# In[158]:


df_state_intr = pd.DataFrame(list(zip(lst3, lst2)),
              columns=['Mean Interest Rate', 'state'])


# In[170]:


df.groupby(['state'])['loan_amount'].mean().keys()


# In[114]:


fig = px.choropleth(df_state_loan,
                    locations='state', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='Mean Loan Amount',
                    color_continuous_scale="Viridis_r", 
                    )
fig.update_layout(
      title_text = 'Mean Loan Amount by State',
      title_font_family="Times New Roman",
      title_font_size = 22,
      title_font_color="black", 
      title_x=0.45, 
         )
fig.show()


# In[159]:


fig = px.choropleth(df_state_intr,
                    locations='state', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='Mean Interest Rate',
                    color_continuous_scale="Viridis_r", 
                    )
fig.update_layout(
      title_text = 'Mean Interest Rate by State',
      title_font_family="Times New Roman",
      title_font_size = 22,
      title_font_color="black", 
      title_x=0.45, 
         )
fig.show()


# In[160]:


plt.figure(figsize=(15,8))

plt.subplot(121)
g = sns.distplot(df["loan_amount"])
g.set_xlabel("", fontsize=12)
g.set_ylabel("Frequency Dist", fontsize=12)
g.set_title("Frequency Distribuition of Loan Amount", fontsize=20)
plt.savefig("/Users/harishsrinivasan/Desktop/UTD/Code/Stout-Case-Study-1/images/loanfreq.png")


# In[137]:


plt.figure(figsize=(15,8))

plt.subplot(121)
g = sns.distplot(df["interest_rate"])
g.set_xlabel("", fontsize=12)
g.set_ylabel("Frequency Dist", fontsize=12)
g.set_title("Frequency Distribuition of interest rate", fontsize=20)
plt.savefig("/Users/harishsrinivasan/Desktop/UTD/Code/Stout-Case-Study-1/images/intfreq.png")


# In[94]:


df['int_round'] = df['interest_rate'].round(0).astype(int)

plt.figure(figsize = (10,8))
g1 = sns.countplot(x="int_round",data=df, 
                   palette="Set1")
g1.set_xlabel("Int Rate", fontsize=12)
g1.set_ylabel("Count", fontsize=12)
g1.set_title("Interest Rate Normal Distribuition", fontsize=20)


# In[138]:


plt.figure(figsize = (10,8))
g = sns.countplot(x="loan_status", data=df)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Count", fontsize=15)
g.set_title("Loan Status Count", fontsize=20)


# In[123]:


loanpurp = pd.DataFrame()
loanpurp = df.groupby(['term','loan_purpose']).size().sort_values()
loanp = loanpurp.unstack()
loanp


# In[124]:


sns.set(style='white')
plt.figure(figsize=(20, 10))
plt.title('Total Loan by term & Loan Purpose')
ax =sns.heatmap(loanp.T,mask= loanp.T.isnull(),annot=True,fmt='g');
ax


# In[126]:


loans_by_purpose = df.groupby('loan_purpose')
plt.figure(figsize = (10,8))
loans_by_purpose['loan_purpose'].count().plot(kind='bar')
plt.title("Loan purpose count")


# In[133]:


plt.figure(figsize=(8,5))
sns.boxplot(x='homeownership',y='interest_rate',data=df2, palette='rainbow')
plt.title("Interest Rate by Homeownership")


# In[139]:


plt.figure(figsize=(8,5))
sns.boxplot(x='verified_income',y='interest_rate',data=df2, palette='rainbow')
plt.title("Interest Rate by Verified Income")


# In[140]:


#df_1 = df['verified_income_joint'].dropna()

plt.figure(figsize=(8,5))
sns.boxplot(x='verification_income_joint',y='interest_rate',data=df2, palette='rainbow')
plt.title("Interest Rate by verification_income_joint")


# In[141]:


plt.figure(figsize=(25,10))
sns.boxplot(x='loan_purpose',y='interest_rate',data=df, palette='rainbow')
plt.title("Interest Rate by loan_purpose")


# In[143]:


plt.figure(figsize=(8,5))
sns.boxplot(x='public_record_bankrupt',y='interest_rate',data=df2, palette='rainbow')
plt.title("Interest Rate by public_record_bankrupt")


# In[144]:



plt.figure(figsize=(8,5))
sns.boxplot(x='public_record_bankrupt',y='loan_amount',data=df, palette='rainbow')
plt.title("loan_amount by public_record_bankrupt")


# ## Data Preprocessing

# In[128]:


# convert na values to 0 
nan_to_0_cols = ['emp_length', 'annual_income_joint', 'debt_to_income_joint', 'debt_to_income', 'months_since_last_delinq', 
                 'months_since_90d_late', 'months_since_last_credit_inquiry', 'num_accounts_120d_past_due']
df[nan_to_0_cols] = df[nan_to_0_cols].fillna(0)
#df1 = pd.get_dummies(df)


# In[129]:


df


# In[130]:


#removing unnecessary features before data manipulation
cols = ['emp_length', 'homeownership', 'annual_income',
       'verified_income', 'debt_to_income', 'annual_income_joint',
       'verification_income_joint', 'debt_to_income_joint', 'delinq_2y',
       'months_since_last_delinq', 'earliest_credit_line',
       'inquiries_last_12m', 'total_credit_lines', 'open_credit_lines',
       'total_credit_limit', 'total_credit_utilized',
       'num_collections_last_12m', 'num_historical_failed_to_pay',
       'months_since_90d_late', 'current_accounts_delinq',
       'total_collection_amount_ever', 'current_installment_accounts',
       'accounts_opened_24m', 'months_since_last_credit_inquiry',
       'num_satisfactory_accounts', 'num_accounts_120d_past_due',
       'num_accounts_30d_past_due', 'num_active_debit_accounts',
       'total_debit_limit', 'num_total_cc_accounts', 'num_open_cc_accounts',
       'num_cc_carrying_balance', 'num_mort_accounts',
       'account_never_delinq_percent', 'tax_liens', 'public_record_bankrupt',
        'application_type',
       'interest_rate', 'installment',
       'balance', 'paid_total', 'paid_principal', 'paid_interest',
       'paid_late_fees']


# In[131]:


#cols2 = ['homeownership', 'verified_income']
df2 = pd.DataFrame()
df2 = df[cols]
df3 = pd.get_dummies(df2)


# In[132]:


df3


# In[145]:


size = df3.shape[0]
rs = 1
Train, Test = train_test_split(df3, test_size= 0.2, random_state= rs)
CV, Test = train_test_split(Test, test_size=0.5, random_state = rs)
print(Train.shape, CV.shape, Test.shape)


# In[146]:


Train_y = np.array(Train["interest_rate"])
CV_y = np.array(CV["interest_rate"])
Test_y = np.array(Test["interest_rate"])
Train_x = Train.drop(["interest_rate"], axis = 1)
CV_x = CV.drop(["interest_rate"], axis = 1)
Test_x = Test.drop(["interest_rate"], axis = 1)


# In[168]:


#lassocv for feature selection
modellasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0001, 10, 1000]).fit(Train_x, Train_y)
lassopred = modellasso.predict(CV_x)
print("RMSE of Lasso: ", np.sqrt(mean_squared_error(lassopred, CV_y)))

coeff = modellasso.coef_

x = list(Train_x)
x_pos = [i for i, _ in enumerate(x)]


plt.figure(figsize = (10,40))
plt.barh(x_pos, coeff, color='green')
plt.ylabel("Features -->")
plt.xlabel("Coefficents -->")
plt.title("Coefficents from Lasso")
plt.yticks(x_pos, x)
plt.savefig("/Users/harishsrinivasan/Desktop/UTD/Code/Stout-Case-Study-1/images/feat_selec1.png", bbox_inches = "tight")
plt.show()


# ## Predicting Interest Rate

# In[149]:


df4 = df3.drop(["interest_rate"], axis = 1)
y = df3['interest_rate']


# In[150]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df4, y, test_size = 0.25,
                                                                           random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Testing Features Shape:', test_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[164]:


def evaluate(model,X_train,X_test,Y_train,Y_test):
    
    model.fit(X_train,Y_train)
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)    
    errors_train = abs(predictions_train - Y_train)
    errors_test = abs(predictions_test - Y_test)
    
    mape_train = 100 * np.mean(errors_train / Y_train)
    mape_test = 100 * np.mean(errors_test / Y_test)
    
    accuracy_train = 100 - mape_train
    accuracy_test = 100 - mape_test
    print('Model Performance')
    
    print('Accuracy(Train Data) = {:0.2f}%.'.format(accuracy_train))
    print('Accuracy(Test Data) = {:0.2f}%.'.format(accuracy_test))
    plt.figure(figsize = (10,10))
    plt.scatter(predictions_train,(predictions_train - Y_train),c='g',s=40,alpha=0.5)
    plt.scatter(predictions_test,(predictions_test - Y_test),c='b',s=40,alpha=0.5)
    plt.hlines(y=0,xmin=0,xmax=30)
    plt.title('residual plot: Blue - test data and Green - train data')
    plt.ylabel('residuals')
    plt.savefig("/Users/harishsrinivasan/Desktop/UTD/Code/Stout-Case-Study-1/images/lr1.png")
    return accuracy_train,accuracy_test


# In[163]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators= 10, random_state=42)

evaluate(rf, train_features, test_features, train_labels, test_labels)


# In[165]:


#train the linear regression model
lm = LinearRegression()

evaluate(lm, train_features, test_features, train_labels, test_labels)


# In[ ]:




