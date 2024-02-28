#!/usr/bin/env python
# coding: utf-8

# ### Pandas for Exploratory Data Analysis
# 
# - MovieLens 100k movie rating data:
#     - Main page: http://grouplens.org/datasets/movielens/
#     - Data dictionary: http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
#     - Files: u.user, u.user_original (no header row)
# 
# - WHO alcohol consumption data:
#     - Article: http://fivethirtyeight.com/datalab/dear-mona-followup-where-do-people-drink-the-most-beer-wine-and-spirits/    
#     - Original data: https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption
#     - file: drinks.csv (with additional 'continent' column)
# 
# - National UFO Reporting Center data:
#     - Main page: http://www.nuforc.org/webreports.html
#     - File: ufo.csv

# In[1]:


import pandas as pd

'''
Reading Files, Selecting Columns, and Summarizing
'''

# can read a file from local computer 
pd.read_table('./Data/u.user')

# read 'u.user' into 'users'
users = pd.read_table('./Data/u.user', sep='|', index_col='user_id')


# ### Examine the users data
# Print the following values and understand the data
# - users                   
# - type(users)             
# - users.head()            
# - users.head(10)          
# - users.tail()            
# - users.index             
# - users.columns           
# - users.dtypes            
# - users.shape             
# - users.values            

# In[ ]:





# ### Select a column

# In[2]:


users['gender']         # select one column
type(users['gender'])   # Series
users.gender            # select one column using the DataFrame attribute


# ### Summarize (describe) the DataFrame
# Try the folloing functions to descrive the data
# 
# - users.describe()                    # describe all numeric columns
# - users.describe(include=['object'])  # describe all object columns
# - users.describe(include='all')       # describe all columns

# In[ ]:





# ### Summarize a Series
# Use  the following to summerise a series
# 
# - users.gender.describe()             # describe a single column
# - users.age.mean()                    # only calculate the mean

# In[ ]:





# ### Count the number of occurrences of each value
# - users.gender.value_counts()     # most useful for categorical variables
# - users.age.value_counts()        # can also be used with numeric variables

# In[ ]:





# ### Exercise 1

# In[2]:


# read drinks.csv into a DataFrame called 'drinks'
drinks = pd.read_table('./Data/drinks.csv', sep=',')
drinks = pd.read_csv('./Data/drinks.csv')              # assumes separator is comma


# In[14]:


# print the head and the tail


# examine the default index, data types, and shape


# print the 'beer_servings' Series


# calculate the mean 'beer_servings' for the entire dataset


# count the number of occurrences of each 'continent' value and see if it looks correct


# BONUS: display only the number of rows of the 'users' DataFrame


# BONUS: display the 3 most frequent occupations in 'users'


# BONUS: create the 'users' DataFrame from the u.user_original file (which lacks a header row)
# Hint: read the pandas.read_table documentation



# ### Filtering and Sorting

# ### Boolean filtering
# Use the following functions to show users with age < 20
# ``` python
# young_bool = users.age < 20         # create a Series of booleans...
# users[young_bool]                   # ...and use that Series to filter rows
# users[users.age < 20]               # or, combine into a single step
# users[users.age < 20].occupation    # select one column from the filtered results
# users[users.age < 20].occupation.value_counts()     # value_counts of resulting Series
# ```

# In[ ]:





# ### Boolean filtering with multiple conditions
# ```python
# users[(users.age < 20) & (users.gender=='M')]       # ampersand for AND condition
# users[(users.age < 20) | (users.age > 60)]          # pipe for OR condition
# ```

# In[ ]:





# ### Sorting
# ``` python
# users.age.order()                   # sort a column
# users.sort('age')                   # sort a DataFrame by a single column
# users.sort('age', ascending=False)  # use descending order instead
# ```

# In[ ]:





# ### Exercise 2

# In[4]:


# filter 'drinks' to only include European countries


# filter 'drinks' to only include European countries with wine_servings > 300

# calculate the mean 'beer_servings' for all of Europe

# determine which 10 countries have the highest total_litres_of_pure_alcohol


# BONUS: sort 'users' by 'occupation' and then by 'age' (in a single command)


# BONUS: filter 'users' to only include doctors and lawyers without using a |
# Hint: read the pandas.Series.isin documentation


# ### Renaming, Adding, and Removing Columns
# Try the floowing code
# ``` python
# # rename one or more columns
# drinks.rename(columns={'beer_servings':'beer', 'wine_servings':'wine'})
# drinks.rename(columns={'beer_servings':'beer', 'wine_servings':'wine'}, inplace=True)
# 
# # replace all column names
# drink_cols = ['country', 'beer', 'spirit', 'wine', 'liters', 'continent']
# drinks.columns = drink_cols
# 
# # replace all column names when reading the file
# drinks = pd.read_csv('./Data/drinks.csv', header=0, names=drink_cols)
# 
# # add a new column as a function of existing columns
# drinks['servings'] = drinks.beer + drinks.spirit + drinks.wine
# drinks['mL'] = drinks.liters * 1000
# 
# # removing columns
# drinks.drop('mL', axis=1)                               # axis=0 for rows, 1 for columns
# drinks.drop(['mL', 'servings'], axis=1, inplace=True)   # drop multiple columns
# ```

# In[ ]:





# ### Handling Missing Values
# Try the following code :
# ```python
# 
# # missing values are usually excluded by default
# drinks.continent.value_counts()              # excludes missing values
# drinks.continent.value_counts(dropna=False)  # includes missing values
# 
# # find missing values in a Series
# drinks.continent.isnull()           # True if missing
# drinks.continent.notnull()          # True if not missing
# 
# # use a boolean Series to filter DataFrame rows
# drinks[drinks.continent.isnull()]   # only show rows where continent is missing
# drinks[drinks.continent.notnull()]  # only show rows where continent is not missing
# 
# # side note: understanding axes
# drinks.sum()            # sums "down" the 0 axis (rows)
# drinks.sum(axis=0)      # equivalent (since axis=0 is the default)
# drinks.sum(axis=1)      # sums "across" the 1 axis (columns)
# 
# # side note: adding booleans
# pd.Series([True, False, True])          # create a boolean Series
# pd.Series([True, False, True]).sum()    # converts False to 0 and True to 1
# 
# # find missing values in a DataFrame
# drinks.isnull()             # DataFrame of booleans
# drinks.isnull().sum()       # count the missing values in each column
# 
# # drop missing values
# drinks.dropna()             # drop a row if ANY values are missing
# drinks.dropna(how='all')    # drop a row only if ALL values are missing
# 
# # fill in missing values
# drinks.continent.fillna(value='NA', inplace=True)   # fill in missing values with 'NA'
# 
# # turn off the missing value filter
# drinks = pd.read_csv('./Data/drinks.csv', header=0, names=drink_cols, na_filter=False)
# ```

# In[ ]:





# ### Exercise 3

# In[ ]:


# read ufo.csv into a DataFrame called 'ufo'

# check the shape of the DataFrame

# calculate the most frequent value for each of the columns (in a single command)

# what are the four most frequent colors reported?


# for reports in VA, what's the most frequent city?


# show only the UFO reports from Arlington, VA

# count the number of missing values in each column


# show only the UFO reports in which the City is missing


# how many rows remain if you drop all rows with any missing values?


# replace any spaces in the column names with an underscore


# BONUS: redo the task above, writing generic code to replace spaces with underscores
# In other words, your code should not reference the specific column names

# BONUS: create a new column called 'Location' that includes both City and State
# For example, the 'Location' for the first row would be 'Ithaca, NY'


# ### Split-Apply-Combine
# ![](http://i.imgur.com/yjNkiwL.png)

# Try the following 
# ```python
# # for each continent, calculate the mean beer servings
# drinks.groupby('continent').beer.mean()
# 
# # for each continent, count the number of occurrences
# drinks.continent.value_counts()
# 
# # for each continent, describe beer servings
# drinks.groupby('continent').beer.describe()
# 
# # similar, but outputs a DataFrame and can be customized
# drinks.groupby('continent').beer.agg(['count', 'mean', 'min', 'max'])
# drinks.groupby('continent').beer.agg(['count', 'mean', 'min', 'max']).sort('mean')
# 
# # if you don't specify a column to which the aggregation function should be applied,
# # it will be applied to all numeric columns
# drinks.groupby('continent').mean()
# drinks.groupby('continent').describe()
# ```

# In[ ]:





# ### Exercise 4

# In[3]:


# for each occupation in 'users', count the number of occurrences

# for each occupation, calculate the mean age

# BONUS: for each occupation, calculate the minimum and maximum ages

# BONUS: for each combination of occupation and gender, calculate the mean age


# ### Other Frequently Used Features
# ```python 
# 
# # map existing values to a different set of values
# users['is_male'] = users.gender.map({'F':0, 'M':1})
# 
# # encode strings as integer values (automatically starts at 0)
# users['occupation_num'] = users.occupation.factorize()[0]
# 
# # determine unique values in a column
# users.occupation.nunique()      # count the number of unique values
# users.occupation.unique()       # return the unique values
# 
# # replace all instances of a value in a column (must match entire value)
# ufo.State.replace('Fl', 'FL', inplace=True)
# 
# # string methods are accessed via 'str'
# ufo.State.str.upper()                               # converts to uppercase
# ufo.Colors_Reported.str.contains('RED', na='False') # checks for a substring
# 
# # convert a string to the datetime format
# ufo['Time'] = pd.to_datetime(ufo.Time)
# ufo.Time.dt.hour                        # datetime format exposes convenient attributes
# (ufo.Time.max() - ufo.Time.min()).days  # also allows you to do datetime "math"
# ufo[ufo.Time > pd.datetime(2014, 1, 1)] # boolean filtering with datetime format
# 
# # setting and then removing an index
# ufo.set_index('Time', inplace=True)
# ufo.reset_index(inplace=True)
# 
# # sort a column by its index
# ufo.State.value_counts().sort_index()
# 
# # change the data type of a column
# drinks['beer'] = drinks.beer.astype('float')
# 
# # change the data type of a column when reading in a file
# pd.read_csv('drinks.csv', dtype={'beer_servings':float})
# 
# # create dummy variables for 'continent' and exclude first dummy column
# continent_dummies = pd.get_dummies(drinks.continent, prefix='cont').iloc[:, 1:]
# 
# # concatenate two DataFrames (axis=0 for rows, axis=1 for columns)
# drinks = pd.concat([drinks, continent_dummies], axis=1)
# ```

# In[ ]:





# ### Less Frequently Used Features
# ``` python
# # create a DataFrame from a dictionary
# pd.DataFrame({'capital':['Montgomery', 'Juneau', 'Phoenix'], 'state':['AL', 'AK', 'AZ']})
# 
# # create a DataFrame from a list of lists
# pd.DataFrame([['Montgomery', 'AL'], ['Juneau', 'AK'], ['Phoenix', 'AZ']], columns=['capital', 'state'])
# 
# # detecting duplicate rows
# users.duplicated()          # True if a row is identical to a previous row
# users.duplicated().sum()    # count of duplicates
# users[users.duplicated()]   # only show duplicates
# users.drop_duplicates()     # drop duplicate rows
# users.age.duplicated()      # check a single column for duplicates
# users.duplicated(['age', 'gender', 'zip_code']).sum()   # specify columns for finding duplicates
# 
# # display a cross-tabulation of two Series
# pd.crosstab(users.occupation, users.gender)
# 
# # alternative syntax for boolean filtering (noted as "experimental" in the documentation)
# users.query('age < 20')                 # users[users.age < 20]
# users.query("age < 20 and gender=='M'") # users[(users.age < 20) & (users.gender=='M')]
# users.query('age < 20 or age > 60')     # users[(users.age < 20) | (users.age > 60)]
# 
# # display the memory usage of a DataFrame
# ufo.info()          # total usage
# ufo.memory_usage()  # usage by column
# 
# # change a Series to the 'category' data type (reduces memory usage and increases performance)
# ufo['State'] = ufo.State.astype('category')
# 
# # temporarily define a new column as a function of existing columns
# drinks.assign(servings = drinks.beer + drinks.spirit + drinks.wine)
# 
# # limit which rows are read when reading in a file
# pd.read_csv('drinks.csv', nrows=10)           # only read first 10 rows
# pd.read_csv('drinks.csv', skiprows=[1, 2])    # skip the first two rows of data
# 
# # write a DataFrame out to a CSV
# drinks.to_csv('drinks_updated.csv')                 # index is used as first column
# drinks.to_csv('drinks_updated.csv', index=False)    # ignore index
# 
# # save a DataFrame to disk (aka 'pickle') and read it from disk (aka 'unpickle')
# drinks.to_pickle('drinks_pickle')
# pd.read_pickle('drinks_pickle')
# 
# # randomly sample a DataFrame
# train = drinks.sample(frac=0.75, random_state=1)    # will contain 75% of the rows
# test = drinks[~drinks.index.isin(train.index)]      # will contain the other 25%
# 
# # change the maximum number of rows and columns printed ('None' means unlimited)
# pd.set_option('max_rows', None)     # default is 60 rows
# pd.set_option('max_columns', None)  # default is 20 columns
# print drinks
# 
# # reset options to defaults
# pd.reset_option('max_rows')
# pd.reset_option('max_columns')
# 
# # change the options temporarily (settings are restored when you exit the 'with' block)
# with pd.option_context('max_rows', None, 'max_columns', None):
#     print drinks
#     
# ```

# In[ ]:




