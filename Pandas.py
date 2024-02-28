#!/usr/bin/env python
# coding: utf-8

# # Pandas
# Primarily a data analysis library which has matured quite a lot recently. Pandas is able to solve many data handling
# problems with high level data structures and manipulation tools built for and around Numpy centric activities.

# ## Introducing Pandas Objects
# 
# 1. At the very basic level, Pandas objects can be thought of as enhanced versions of NumPy structured arrays in which the rows and columns are identified with labels rather than simple integer indices.
# 
# 2. Pandas provides a host of useful tools, methods, and functionality on top of the basic data structures, but nearly everything that follows will require an understanding of what these structures are.
#     - Three fundamental Pandas data structures:
#         1. ``Series``
#         2. ``DataFrame``
#         3. ``Index``.

# ## Series
# Series is a one dimensional array like object containing an array of data and data associated labels called index. lets 
# see an example

# In[1]:


import numpy as np
import pandas as pd


# In[20]:


data=[4,8,9,-1]
dataSeries=pd.Series(data)

print (dataSeries)
print 
print ("Indexes of the series : ", dataSeries.index)
print
print ("Values : ", dataSeries.values)

pd.Series()


# lets Give the Series our custom labels

# In[6]:


data=[4,8,9,-1,'a']
labels=['a','b','c','d',2]
dataSeries=pd.Series(data=data,index=labels)

print (dataSeries)
print 
print ("Indexes of the series : ", dataSeries.index)
print
print ("Values : ", dataSeries.values)


# ### Accessing Values

# We can access the values using the indexes. In the example above we used```[a,b,c,d]``` as indexes so we can use them to access values : 

# In[3]:


print ("Value at index 'a' is : ", dataSeries['a'])
print 
idx=['a','c']
print ("Values as indexes", idx, " are : ")
print (dataSeries[['a','c']])


# Lets try some numpy operation and see there results

# In[5]:


print (dataSeries[dataSeries<5].index)


# In[6]:


print (dataSeries*2)


# In[7]:


print (np.exp(dataSeries))


# In[27]:


print ('a' in dataSeries)


# In[9]:


print ('e' in dataSeries)


# ### Series as specialized dictionary
# 
# In this way, you can think of a Pandas ``Series`` a bit like a specialization of a Python dictionary.
# A dictionary is a structure that maps arbitrary keys to a set of arbitrary values, and a ``Series`` is a structure which maps typed keys to a set of typed values.
# This typing is important: just as the type-specific compiled code behind a NumPy array makes it more efficient than a Python list for certain operations, the type information of a Pandas ``Series`` makes it much more efficient than Python dictionaries for certain operations.
# 
# The ``Series``-as-dictionary analogy can be made even more clear by constructing a ``Series`` object directly from a Python dictionary:

# In[14]:


sdata={'Ohio':3500 , 'Texas':7100 , 'Oregon': 1600 , 'Utah': 5000 }
ssdata=pd.Series(sdata)
print (sdata)
print(type(ssdata))
type(sdata)


# In[15]:


labels=['Utah','Ohio','Oregon','Texas']
adata=pd.Series(sdata,index=labels)
print (adata)
type(adata)


# pd.isnull and pd.notnull functions must be used to detect the missing value

# In[12]:


pd.isnull(adata)


# In[13]:


pd.notnull(adata)


# In[20]:


adata/ssdata


# ## Dataframe
# Dataframe is similar to a data table like a spreadsheet containing ordered collection of columns; each of which can be a 
# different type. Dataframe has both a row and column index. Dataframe can be though of a dictionary of Series. Data frame store data internally in two dimensional blocks.   

# In[15]:


data={
        'state': ['Ohio','Ohio','Ohio','Nevada','Nevada'],
        'year': [2000,2001,2002,2001,2002],
        'pop': [1.5,1.7,3.6,2.4,2.9]
     }
dataDF=pd.DataFrame(data)
print (dataDF)


# If you specify a sequence of columns, the dataframe will be exactly what you pass

# In[16]:


dataDF=pd.DataFrame(data,columns=['year','state','pop'])
print (dataDF)


# if you pass a column that does not exists in data it will appear with NA values.

# In[20]:


dataDF=pd.DataFrame(data,columns=['year','state','pop','debt'],index=['one','two','three','four','five'])
print (dataDF)


# ### Indexing

# You can access the coloumns in a dataframe using either as an attribute, or as a key value pair
# - As attribute

# In[18]:


dataDF.year


# - As Key-Value pair

# In[19]:


dataDF['year']


# * You can access the rows using the _ix_ utility

# In[27]:


dataDF.ix[2]


# In[28]:


# Row selecting using Boolean Indexing

dataDF[dataDF['year']==2002]


# - A new coloumn can be added by giving same value as all rows as follows

# In[61]:


dataDF['debt']=16.5
print dataDF


# - Or if you want different values for each row

# In[74]:


val=pd.Series(data=[-1.2,-1.5,-1.7],index=['two','five','three'])
dataDF['debt']=val
dataDF


# - Adding a new column on the fly by simpling assigning a column. The del keyword will delete the specified column.

# In[62]:


dataDF['eastern']=dataDF.state=='Ohio'
dataDF


# **P.S : ** In the example above _dataDF.state=='Ohio'_ is creating a boolean array and it is being assigned to a new coloumn in data frame using _dataDF['eastern']= dataDF.state=='Ohio'_ . It is to be noted that "Eastern" coloumn did not exist prior to this statement rather is created on the fly.  

# - We can also delete a coloumn using the _del_ keyword

# In[21]:


del dataDF['debt']
dataDF


# - A row can be deleted using _drop_ function

# In[24]:


dataDF.drop('two')
print (dataDF)


# - We can take the transpose of a dataframe as following 

# In[29]:


dataDF.T


# - The reindex function of pandas can re-arrange the rows and coloumns 

# In[30]:


dataDF.reindex(index=['five','three','two','four','one'])


# In[31]:


dataDF.reindex(index=['five','three','two','four','one'],columns=['debt','state','year','pop'])


# - Applying custom functions on pandas

# In[32]:


f=lambda x:x-1
dataDF['year'].apply(f)


# In[33]:


dataDF['pop'].apply(f)


# ### Sorting in pandas !

# - We can sort a dataframe on the basis of index or values

# In[34]:


dataDF.sort_index(axis=0)


# In[35]:


dataDF.sort_values('pop')


# Data frame provide a lot of stats about the values. Lets see the overall statistical map of our dataframe

# In[36]:


dataDF.describe()


# - Of if you just want to apply one of the aggregation function

# In[37]:


dataDF.sum()


# The following table summarizes some other built-in Pandas aggregations:
# 
# | Aggregation              | Description                     |
# |--------------------------|---------------------------------|
# | ``count()``              | Total number of items           |
# | ``mean()``, ``median()`` | Mean and median                 |
# | ``min()``, ``max()``     | Minimum and maximum             |
# | ``std()``, ``var()``     | Standard deviation and variance |
# | ``mad()``                | Mean absolute deviation         |
# | ``prod()``               | Product of all items            |
# | ``sum()``                | Sum of all items                |
# 
# These are all methods of ``DataFrame`` and ``Series`` objects.

# ### Handling Missing data

# In[ ]:


data = pd.Series([ "Umman", np.nan,"Samad", None])
print data 


# **P.S**: Here NAN and None represent missing values, remember that if a value is not provided, pandas fill it with NAN

# - One way is to just drop the missing values

# In[ ]:


data = data.dropna()
print data


# - You can also fill al the NAN and None values with a default value 

# In[17]:


data = pd.Series([ "Usman", np.nan,"Samad", None])

data = data.fillna("Default")
print (data)


# **Note : ** IN case of numerical data, these values can be filled with mean or mode etc.

# ## GroupBy: Split, Apply, Combine
# 
# Simple aggregations can give you a flavor of your dataset, but often we would prefer to aggregate conditionally on some label or index: this is implemented in the so-called ``groupby`` operation.
# The name "group by" comes from a command in the SQL database language, but it is perhaps more illuminative to think of it in the terms first coined by Hadley Wickham of Rstats fame: *split, apply, combine*.

# ![](figures/03.08-split-apply-combine.png)

# This makes clear what the ``groupby`` accomplishes:
# 
# - The *split* step involves breaking up and grouping a ``DataFrame`` depending on the value of the specified key.
# - The *apply* step involves computing some function, usually an aggregate, transformation, or filtering, within the individual groups.
# - The *combine* step merges the results of these operations into an output array.
# 
# While this could certainly be done manually using some combination of the masking, aggregation, and merging commands covered earlier, an important realization is that *the intermediate splits do not need to be explicitly instantiated*. Rather, the ``GroupBy`` can (often) do this in a single pass over the data, updating the sum, mean, count, min, or other aggregate for each group along the way.
# The power of the ``GroupBy`` is that it abstracts away these steps: the user need not think about *how* the computation is done under the hood, but rather thinks about the *operation as a whole*.
# 
# As a concrete example, let's take a look at using Pandas for the computation shown in this diagram.
# We'll start by creating the input ``DataFrame``:

# In[69]:


dataDF.groupby('year')


# Notice that what is returned is not a set of ``DataFrame``s, but a ``DataFrameGroupBy`` object.
# This object is where the magic is: you can think of it as a special view of the ``DataFrame``, which is poised to dig into the groups but does no actual computation until the aggregation is applied.
# This "lazy evaluation" approach means that common aggregates can be implemented very efficiently in a way that is almost transparent to the user.
# 
# To produce a result, we can apply an aggregate to this ``DataFrameGroupBy`` object, which will perform the appropriate apply/combine steps to produce the desired result:

# In[ ]:


dataDF


# In[70]:


dataDF.groupby('year').sum()


# In[ ]:


dataDF.groupby('year').mean()


# In[71]:


dataDF.groupby('year')['pop'].mean()


# In[ ]:


dataDF.groupby('year').aggregate(['min', np.median, max])


# ### Transformation
# 
# While aggregation must return a reduced version of the data, transformation can return some transformed version of the full data to recombine.
# For such a transformation, the output is the same shape as the input.
# A common example is to center the data by subtracting the group-wise mean:

# In[72]:


def subtract_group_by_mean(x):
#     print x
    m=np.mean(x)
#     print 'mean:',m
    return x-m


# In[73]:


dataDF.groupby('year')['pop'].transform(subtract_group_by_mean)


# In[ ]:




