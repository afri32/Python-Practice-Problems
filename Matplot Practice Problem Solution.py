#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,10)
y=np.square(x)
plt.plot(x, y,'g', linestyle='-')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.show()


# In[2]:


x=np.random.rand(10)
y=np.random.rand(10)
sizes = np.random.randint(1,10)
plt.scatter(x, y, s=sizes, alpha=0.7)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('scatter plot')
c = plt.colorbar()
c.set_label('sizes')
plt.show()


# In[3]:


cate = [' A', 'B', 'C', 'D']
v = np.array([6, 8, 4, 5])
e = np.array([1, 1.5, 0.5, 0.9])
plt.bar(categories, v, yerr=e, color='green')
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Bar Chart ')
plt.show()


# In[ ]:


data = np.random.normal(size=1000)
plt.hist(data, bins=20, color='g')
plt.hist(data, bins=30, color='r')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram ')
plt.show()


# In[ ]:


x1 = x**2
y1 = x1**3
x2 = x2**3
y2 = x2**3
x3 = x3+1
y3 = np.exp(x3)
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0,0].plot(x1, y1, color='b')
axs[0,0].set_title('Plot 1')
axs[0,0].set_xlabel('X-axis')
axs[0,0].set_ylabel('Y-axis')
axs[0, 1].plot(x2, y2, color='g')
axs[0, 1].set_title('Plot 2')
axs[0, 1].set_xlabel('X-axis')
axs[0, 1].set_ylabel('Y-axis')
axs[1, 0].scatter(x3, y3, color='r')
axs[1, 0].set_title('Plot 3')
axs[1, 0].set_xlabel('X-axis')
axs[1, 0].set_ylabel('Y-axis')
fig.suptitle('Complex')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
arr1=np.arange(1,10)
arr2=arr1**2
print(arr1)
print(arr2)
plt.plot(arr1,arr2,'g','--')
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("x-y graph")
plt.show()


# In[ ]:


import random
arr1=np.random.rand(1,10)
arr2=np.random.rand(1,10)
sizes=np.random.randint(1,10)
plt.scatter(arr1,arr2,s=sizes,alpha=0.7)
plt.xlabel("x axis")
plt.ylabel("y-axis")
plt.title("scatter plot")
c=plt.colorbar()
c.set_label("sizes")
plt.show()


# In[ ]:


cat=['A','B','C']
val=[1,2,3]
err=[1,1.2,1.5]
plt.bar(cat,val,yerr=err,color='yellow')
plt.show()


# In[ ]:


data=np.random.normal(size=1000)
plt.hist(data,bins=20,color="blue")
plt.hist(data,bins=30,color="red")
plt.show()


# In[ ]:


x1=arr1**2
y1=x1+1
x2=arr2**2
y2=x2+1
x3=arr2**3
y3=np.exp(x3)
fig,ax=plt.subplots(2,2,figsize=(8,4))
ax[0,0].plot(x1,y1,'g')
ax[0,1].plot(x2,y2,'b')
ax[1,0].plot(x3,y3,'r')
plt.show()


# In[ ]:


import numpy as np
arr=np.random.randint(1,100,(5,5))
print(arr)
for i in range(5):
    for j in range(5):
        if arr[i][j]%3==0 and arr[i][j]%5==0:
            arr[i][j]=arr[i][j]**2
print(arr)


# In[ ]:


a=np.random.randint(1,100,(4,4))
b=np.random.randint(1,100,(4,4))
c=a*b
print(c)


# In[ ]:


a=np.random.randint(1,50,(6,6))
print(a)
d=[]
for i in range(6):
    for j in range(6):
        if(i==j):
            d.append(a[i,j])
print(d)
sorti=np.sort(d)
print(sorti)


# In[ ]:


data={'Name':['afra','hafsa','barira','afri','aafi'],
     'Age':np.random.randint(1,100,(5)),
     'Score':np.random.randint(1,100,(5))}
df=pd.DataFrame(data)
filter=df[df['Score']>70]
print(filter)


# In[ ]:


data={'Product':['face-wash','powder','bleach','maskhara'],
     'Quantity':[1,2,3,4],
     'Price':[1200,1300,1400,6700]}
df=pd.DataFrame(data)
df['total']=df['Quantity']*df['Price']
df=df.sort_values(by='total',ascending=True)
df


# In[ ]:


data={'Name':['afra','hafsa','donkey1','shopper'],
     'Math':np.random.randint(1,100,(4)),
     'Science':np.random.randint(1,100,(4)),
     'Urdu':np.random.randint(1,100,(4))
     }
df=pd.DataFrame(data)
df['Avg']=df[['Math','Science','Urdu']].mean(axis=1)
Max=df['Avg'].idxmax()
print(Max)
print(df)


# In[ ]:


data={'Name':['afra','hafsa','bari'],
     'Age':[12,13,14],
     'Salary':[1200,1300,1400]}
df=pd.DataFrame(data)
df['Bonus']=np.random.randint(1,1000,(3))
df['Earn']=df['Salary']+df['Bonus']
df


# In[ ]:


arr=np.random.randint(1,50,(5,5))
df=pd.DataFrame(arr,columns=['A','B','C','D','E'])
df=df.applymap(lambda x:x**2 if x%3==0 else x)
df


# In[ ]:


class multi:
    def __init__(self,arr,n):
        self.arr=arr
        self.n=n
    def __mul__(self,other):
        m=self.arr*self.n
        return m
    def __repr__(self):
        return f"{self.arr}+{self.n}"
a=np.random.randint(1,10,(2,2))
print("array is:\n",a)
c1=multi(a,2)
c2=multi(a,6)
r=c1*c2
print(r)

