#!/usr/bin/env python
# coding: utf-8

# # Excercise Exploratory Data Analysis

# For this lab, we'll explore some data from a very useful source, the UC Irvine machine learning data repository.
# 
# Especially, we will be using this dataset, https://archive.ics.uci.edu/ml/datasets/Heart+Disease. Here our goal is to know whether a disease is present or not.
# 
# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0). 
# 
# **Goal: Can you find nice relationships, and do exploratory analysis to come up some data supported conclusions**

# In[1]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10.0, 8.0)


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df=pd.read_csv('heart.csv')  
df.columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']                                


# > TODO: How many rows are there in the dataset?
# 

# In[5]:


print(df.dtypes) # what can you conclude...


# ## Data Cleaning

# First we have to clean and sanitize the data. This data is pretty clean and is mostly numeric but contains some '?' in some fields.  To make it easier to handle, we convert those fields to ```None```. For convenience, you should define a function "safefloat" that takes a string argument, and returns None if the argument is '?', otherwise the float value of the string. 

# In[6]:


# TODO write safefloat
def safefloat(x):
    return 


# Find which fields contain '?' and try applying safefloat to fields containing '?' to make sure it is working. After that it can be used, when required.

# In[ ]:





# #### Column names 

# In[14]:


headers = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']


# Define a function "getcol" that takes a column name and returns the data in that column as a list of numbers.

# In[15]:


def getcol(name): # TODO write getcol
    return


# ## Basic Statistics

# What is the minimum, maximum, mean and standard deviation of the age of this set of subjects? Use the numpy package with contains the mean() and std() functions. 

# In[16]:


def stats(col):
    return


# In[18]:


stats('age')


# ## Histograms of Data Fields

# Implement the function ```plot_hist``` that:
# 
#     Takes two strings, col name and it's unit as arguments
#     Get the data of col using the function get_col implemented above
#     Create a figure with one subplot
#     Plot the histogram of the coloumn
#     Set the unit as ylabel
#     Set the col name as title
#     Show the figure

# In[19]:


# Write your code here
def plot_hist(col, unit):
    return


# In[20]:


# Write your code here
plot_hist('trestbps', 'unit')


# > TODO
# Describe the rough shape of the distribution of bps.
# Is it skewed? 

# In[21]:


# Write your answer here


# ## Scatter Plots

# Implement the function ```plot_scatter``` that:
# 
#     Takes four strings, col1 , col2 , unit1 and unit2 as arguments
#     Get the data of cols using the function get_col implemented above
#     Create a figure with one subplot
#     Plot the histogram of the coloumn
#     Set the appropriate units as x and y labels
#     Set the title by concatinating col names
#     Show the figure

# In[22]:


# Write your code here
def plot_scatter(col1, col2, unit1, unit2):
    return


# Make scatter plots of:
# *    age vs bp (resting blood pressure) 
# *    age vs thalach (max heart rate)
#     

# In[23]:


# Write your code here
plot_scatter('age','trestbps', 'uni1','unit2' )


# In[24]:


# Write your code here


# 

# ## Critical Thinking with Data

# Think about relationship between blood pressure and heart disease

# In[17]:


#plot the relationshiop


# > TODO: Based on this plot, do you think blood pressure influences heart disease?

# In[18]:


# write your answer here


# Now repeat this plot of age versus num:

# > TODO: Based on this plot of Age vs Num and the previous plot of Age vs BPS, what would you say now about the relation between BPS and Num?

# In[ ]:





# ## Dimension Reduction

# Recall that dimension reduction allows you to look at the dominant factors in high-dimensional data. Matplotlib includes the PCA function for this purpose. You use it like this:

# In[31]:


# you may need to sanitize the data here


# In[24]:


from matplotlib.mlab import PCA
cleveland_matrix = np.array(df, dtype=np.float64) # First put the data in a 2D array of double-precision floats
print cleveland_matrix[:,0:8].shape

results = PCA(cleveland_matrix[:,0:8])                      # leave out columns with None in them
yy = results.Y                                              # returns the projections of the data into the principal component directions


# In[25]:


plt.scatter(yy[:,0],yy[:,1])


# > TODO: Do you see a relationship between the two main variables (X and Y axes of this plot)?

# In[ ]:





# ### Get Creative
# Now it's your turn to use the tools you have learned and find out intresting insights from the data.

# In[ ]:




