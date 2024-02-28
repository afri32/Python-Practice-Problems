#!/usr/bin/env python
# coding: utf-8

# # Lists

# In[3]:


myList=[11,42,15,64,80,32,12]   #creating a list of the same data type

print(myList[0:3])              #printing items from zero to three
print(myList[:3])               #printing items from zero to three
print(myList[3:])               #printing items from three to last
print(myList[2:5])              #printing items from two to five


# In[4]:


myList=[10,12.5,'H',"Hello"]  #Creating a list of different data types
print(myList)                 #printing the whole list
print(myList[0])              #printing the 0th indexed item of the list
print(myList[1])              #printing the 1st indexed item of the list


# In[6]:


myList = ['Maths', 'Science', 2020, 2021]
print ("Value available at index 2 : ", myList[2])

myList[2] = 2001   #Assigning new value to the index:2
print ("New value available at index 2 : ", myList[2])


# In[15]:


myList = ['Maths', 'Science', 2020, 2021]
print ("Initial list                     : ",myList)

del myList[2]
print ("After deleting value at index 2  : ", myList)


# In[17]:


myList=[11,42,15,64,80,32,12]
print ("Length of the list     : ", len(myList))
print ("Maximum value element  : ", max(myList))
print ("Minimum value element  : ", min(myList))
print ("Sum of the list values : ", sum(myList))


# In[32]:


myList=[11,42,15,64,80,32,12,42]
print ("Initial list  : ",myList)             #Original List

myList.append(50)                             #Appending the value 50 to the end of the list
print ("updated list  : ", myList)            #Printing the list after appending the value

print ("Count for 42  : ", myList.count(42))  #Printing the no of times 42 appears

print ('Index of 15   : ', myList.index(15))  #Printing the index of the value 15

list2 = [5,10,15]                             #Creating a second list
myList.extend(list2)                          #Appending the second list to the first list
print ('Extended List : ', myList)


# # Tuples

# In[36]:


tup1 = ('Maths', 'Science', 2020, 2021)       #Creating a python tuple with parentheses
tup2 = "a", "b", "c", "d"                     #Creating a python tuple without parentheses

print ("tup1[0]   : ", tup1[0])               #Printing the 0th indexed value
print ("tup2[1:5] : ", tup2[1:5])             #Printing the values from one to five


# In[46]:


myTuple = (11,42,15,64,80,32,12)

# Following action is not valid for tuples
myTuple[0] = 100


# In[47]:


myTuple = (11,42,15,64,80,32,12)
print(max(myTuple))               #printing the max value of the tuple
print(len(myTuple))               #printing the length of the tuple


# In[51]:


myTuple = (11,42,15,64,80,32,12)

myList=list(myTuple)             #Converting tuple into a list
print(myList)                    #Print the initial list

myList.append(100)               #Appending the value 100 to the list
myList[0]=20                     #Assign 20 as the 0th indexed value
print(myList)                     #Print the changed list


# # Dictionaries

# In[57]:


Student={'name':'John','maths':82,'phy':88,'chem':92,'english':80}

print(Student['name'])     #Printing the value of the key:name


# In[58]:


Student={'name':'John','maths':82,'phy':88,'chem':92,'english':80}
print('Original dictionary      : ',Student)

Student['name']='William'                       #Changing the value for the key:name
print('After changing the value : ',Student)


# In[61]:


myDict = {'Name': 'John', 'Age': 20, 'Name': 'William'}
print ("myDict['Name']: ", myDict['Name'])       


# In[62]:


myDict = {['Name']: 'John', 'Age': 20}
print ("dict['Name']: ", myDict['Name'])


# In[66]:


myDict={'name':'John','maths':82,'phy':88,'chem':92,'english':80}

print("Length of the dict  : ", len (myDict))

dict2 = myDict.copy()
print("Copied Dictionary   : ",dict2)

keys=myDict.keys()
print("Dictionary keys     : ",keys)

vals=myDict.values()
print("Dictionary values   : ",vals)


# In[ ]:




