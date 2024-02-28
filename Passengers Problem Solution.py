#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
df = pd.read_csv('C:\\Users\F223119\Downloads\Titanic-Dataset.csv',delimiter=',')
df


# In[2]:


x = df['Fare'].max()
info = df[(df['Fare'] == x)]
info


# In[11]:


new_y = df['Survived']
new=df[(df['Survived'] == 1)]
new


# In[12]:


pcounts= df['Pclass'].value_counts()
pcounts


# In[5]:


missing_val = df.isnull().sum()
missing_val


# In[13]:


df.dropna()
missing_value= df['Age'].fillna( df['Age'].mean())
missing_value


# In[14]:


import matplotlib.pyplot as plt
import pandas as pd
plt.plot(df['Age'])
plt.show()


# In[15]:


df.dropna(subset=['Age','Cabin'])


# In[18]:


df['Family_size']=df['SibSp']+df['Parch']


# In[19]:


fare_above=df[df['Fare']>df['Fare'].mean()]
fare_above


# In[23]:


import matplotlib.pyplot as plt
import pandas as pd
s=df.groupby('Pclass')['Survived'].mean()
plt.bar(s.index, s)
plt.title("survived rate")
plt.xlabel('Ticket')
plt.ylabel('Rate')
plt.show()


# In[33]:


import matplotlib.pyplot as plt
import pandas as pd
plt.hist(df['Age'],df['PassengerId'])
plt.title("survived rate")
plt.xlabel('ages')
plt.ylabel('Rate')
plt.show()


# In[34]:


import matplotlib.pyplot as plt
import pandas as pd
plt.bar(df['Age'].mean(),df['PassengerId'].max())
plt.title("survived rate")
plt.xlabel('ages')
plt.ylabel('Rate')
plt.show()


# In[35]:


df.sort_values(by=['Age'])


# In[38]:


df.groupby('PassengerId').mean()


# In[39]:


df.groupby('Age').mean()


# In[40]:


df.groupby('Age')['PassengerId'].mean()


# In[42]:


df.groupby('Fare').aggregate(['min', np.median, max])


# In[43]:


pcounts= df['Pclass'].value_counts()
pcounts


# In[44]:


pcounts= df['Survived'].value_counts()
pcounts


# In[ ]:




