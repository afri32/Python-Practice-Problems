#!/usr/bin/env python
# coding: utf-8

# https://towardsdatascience.com/classification-basics-walk-through-with-the-iris-data-set-d46b0331bf82

# In[1]:


from sklearn.datasets import load_iris
#save data information as variable
iris = load_iris()
#view data description and information
print(iris.DESCR)


# Putting Data into a Data Frame
# Feature Data
# To view the data more easily we can put this information into a data frame by using the Pandas library. Let’s create a data frame to store the data information about the flowers’ features first.

# In[2]:


import pandas as pd
#make sure to save the data frame to a variable
data = pd.DataFrame(iris.data)
#By default print 5 rows of data
data.head()


# In[3]:


data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#note: it is common practice to use underscores between words, and avoid spaces
data.head()


# In[6]:


#put target data into data frame
target = pd.DataFrame(iris.target)
#Lets rename the column so that we know that these values refer to the target values
target = target.rename(columns = {0: 'target'})
target.head()


# In[7]:


df = pd.concat([data, target], axis = 1)
#note: it is common practice to name your data frame as "df", but you can name it anything as long as you are clear
#and consistent #in the code above, axis = 1 tells the data frame to add the target data frame as another column of
#the data data frame, axis = 0 would add the values as another row on the bottom
df.head()


# In[8]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.heatmap(df.corr(), annot = True);
#annot = True adds the numbers onto the squares


# In[13]:


# The indices of the features that we are plotting (class 0 & 1)
x_index = 0
y_index = 1
# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.tight_layout()
plt.show()


# In[14]:


x_index = 2
y_index = 3
# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.tight_layout()
plt.show()


# In[15]:


#divide our data into predictors (X) and target values (y)
X = df.copy()
y = X.pop('target')
y


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify = y)

#by stratifying on y we assure that the different classes are represented proportionally to the amount 
#in the total data (this makes sure that all of class 1 is not in the test group only)


# In[1]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
#The fit(data) method is used to compute the mean and std dev for a given feature to be used further for scaling.
#The transform(data) method is used to perform scaling using mean and std dev calculated using the .fit() method.
#The fit_transform() method does both fits and transform.


# In[15]:


df.target.value_counts(normalize= True)


# In[16]:


from sklearn.linear_model import LogisticRegression
#create the model instance
model = LogisticRegression()
#fit the model on the training data
model.fit(X_train, y_train)
#the score, or accuracy of the model
model.score(X_test, y_test)
# Output = 0.9666666666666667
#the test score is already very high, but we can use the cross validated score to ensure the model's strength 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=10)
print(np.mean(scores))
# Output = 0.9499999999999998


# In[17]:


df_coef = pd.DataFrame(model.coef_, columns=X_train.columns)
df_coef


# In[18]:


redictions = model.predict(X_test)
#compare predicted values with the actual scores
compare_df = pd.DataFrame({'actual': y_test, 'predicted': predictions})
compare_df = compare_df.reset_index(drop = True)
compare_df


# In[19]:


from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test, predictions, labels=[2, 1, 0]),index=[2, 1, 0], columns=[2, 1, 0])


# In[20]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[21]:


probs = model.predict_proba(X_test)
#put the probabilities into a dataframe for easier viewing
Y_pp = pd.DataFrame(model.predict_proba(X_test), 
             columns=['class_0_pp', 'class_1_pp', 'class_2_pp'])
Y_pp.head()


# In[22]:


# load the iris dataset as an example  
from sklearn.datasets import load_iris  
iris = load_iris()  
    
# store the feature matrix (X) and response vector (y)  
X = iris.data  
y = iris.target  
    
# store the feature and target names  
feature_names = iris.feature_names  
target_names = iris.target_names  
    
# printing features and target names of our dataset  
print("Feature names:", feature_names)  
print("Target names:", target_names)  
    
# X and y are numpy arrays  
print("\nType of X is:", type(X))  
    
# printing first 5 input rows  
print("\nFirst 5 rows of X:\n", X[:5])


# In[23]:


# load the iris dataset as an example 
from sklearn.datasets import load_iris 
iris = load_iris() 
  
# store the feature matrix (X) and response vector (y) 
X = iris.data 
y = iris.target 
  
# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 
  
# printing the shapes of the new X objects 
print(X_train.shape) 
print(X_test.shape) 
  
# printing the shapes of the new y objects 
print(y_train.shape) 
print(y_test.shape)


# In[24]:


# load the iris dataset as an example 
from sklearn.datasets import load_iris 
iris = load_iris() 
  
# store the feature matrix (X) and response vector (y) 
X = iris.data 
y = iris.target 
  
# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 
  
# training the model on training set 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train) 
  
# making predictions on the testing set 
y_pred = knn.predict(X_test) 
  
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("kNN model accuracy:", metrics.accuracy_score(y_test, y_pred)) 
  
# making prediction for out of sample data 
sample = [[3, 5, 4, 2], [2, 3, 5, 4]] 
preds = knn.predict(sample) 
pred_species = [iris.target_names[p] for p in preds] 
print("Predictions:", pred_species) 


# In[25]:


# import packages 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
  
# importing data 
df = pd.read_csv('headbrain1.csv') 
  
# head of the data 
print(df.head()) 
  
X= df['Head Size(cm^3)'] 
y=df['Brain Weight(grams)'] 
  
# using the train test split function 
X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=104, test_size=0.25, shuffle=True) 
# printing out train and test sets 
  
print('X_train : ') 
print(X_train.head()) 
print('') 
print('X_test : ') 
print(X_test.head()) 
print('') 
print('y_train : ') 
print(y_train.head()) 
print('') 
print('y_test : ') 
print(y_test.head())


# In[ ]:





# In[ ]:




