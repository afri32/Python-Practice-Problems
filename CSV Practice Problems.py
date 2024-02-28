#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
df=pd.read_csv('hypothetical_dataset - hypothetical_dataset.csv',delimiter=',')
df.head(10)


# In[2]:


df.describe()


# In[3]:


df.dtypes


# In[4]:


df.isnull().sum()


# In[5]:


df = df.fillna(df.mean())
df


# In[6]:


df = df.dropna()
df


# In[7]:


df['Education Level'] = df['Education Level'].astype('category')
df


# In[8]:


corr = df.corr()
print("Correlation Matrix:")
print(corr)


# In[9]:


df.corr()


# In[10]:


import matplotlib.pyplot as plt
x = 'Age'
y= 'Education Level'
plt.scatter(df[x], df[y])
plt.title(f'Scatter Plot: {x} vs {y}')
plt.xlabel(x)
plt.ylabel(y)
plt.show()


# In[11]:


x = 'Age'
plt.hist(df[x], bins=20, color='skyblue', edgecolor='black')
plt.title(f'Histogram: {x}')
plt.xlabel(x)
plt.ylabel('Frequency')
plt.show()


# In[12]:


age_column = 'Age'
age_bins = [0, 10, 20, 30, 40, 50, 100]  
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51+']

df['Age Group'] = pd.cut(df[age_column], bins=age_bins, labels=age_labels)

health_score_column = 'Health Score'
average_health_by_age = df.groupby('Age Group')[health_score_column].mean()

income_column = 'Income'
average_income_by_age = df.groupby('Age Group')[income_column].mean()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

axes[0].bar(average_health_by_age.index, average_health_by_age, color='r')
axes[0].set_title(' Health Score ')
axes[0].set_xlabel('Age')
axes[0].set_ylabel(' Score')
axes[1].bar(average_income_by_age.index, average_income_by_age, color='g')
axes[1].set_title('Average Income ')
axes[1].set_xlabel('Age ')
axes[1].set_ylabel(' Income')
axes[2].bar(average_income_by_age.index, average_income_by_age, color='b')
axes[2].set_title('average Income ')
axes[2].set_xlabel('age ')
axes[2].set_ylabel(' income')
plt.show()


# In[13]:


age_column = 'Age'
age_bins = [0, 18, 30, 40, 50, 60, 100]
age_labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61+']
df['AgeGroup'] = pd.cut(df[age_column], bins=age_bins, labels=age_labels)
df['Combine'] = df['Health Score'] +df['Income']
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
average_health_by_age = df.groupby('AgeGroup')['Health Score'].mean()
axes[0].bar(average_health_by_age.index, average_health_by_age, color='purple')
axes[0].set_title('Average Health ')
axes[0].set_xlabel('Age')
axes[0].set_ylabel(' Health Score')
average_combined_feature_by_age = df.groupby('AgeGroup')['Combine'].mean()
axes[1].bar(average_combined_feature_by_age.index, average_combined_feature_by_age, color='green')
axes[1].set_title('Average Combined Feature by Age Group')
axes[1].set_xlabel('Age Group')
axes[1].set_ylabel('Average Combined Feature')
plt.show()
# Calculate correlation matrix
correlation_matrix = df[['Age', 'Health Score', 'Income', 'Combine']].corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


# In[25]:


import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot(df['Health Score'], bins=20, kde=True, color='green', edgecolor='blue', alpha=0.9)
plt.title('Histogram: Health Score')
plt.xlabel('HealthScore')
plt.ylabel('Frequency')
plt.show()


# In[28]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df.describe()

plt.figure(figsize=(12, 6))

plt.title(' Prices ')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

mean_returns = df.groupby('Age')['Health Score'].mean()
plt.figure(figsize=(10, 6))
mean_returns.plot(kind='bar', color='skyblue')
plt.title('Mean ')
plt.xlabel('data')
plt.ylabel('Mean ')
plt.show()
correlation_matrix = df[['Age', 'Income', 'Health Score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()
df.to_csv('cleaned_and_transformed_data.csv', index=False)


# In[ ]:




