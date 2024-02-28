#!/usr/bin/env python
# coding: utf-8

# ##  OPEN CV 

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2


# In[15]:


image=cv2.imread("test.jpg")
image_in_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#convert in hsv
image_in_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#spit channels for rgb image
r_channel,g_channel,b_channel=cv2.split(image_in_rgb)
#split hsv image
h_channel,s_channel,v_channel=cv2.split(image_in_hsv)
plt.figure(figsize=(8,4))
plt.subplot(2,4,1)
plt.imshow(image_in_rgb)
plt.title('original image')


# In[17]:


plt.subplot(2,4,1)
plt.imshow(r_channel,cmap='gray')
plt.title('red channel')


# In[19]:


plt.subplot(2,4,1)
plt.imshow(b_channel)
plt.title('blue channel')


# In[20]:


plt.imshow(image_in_hsv)


# In[22]:


plt.subplot(2,4,2)
plt.imshow(image_in_rgb)


# In[44]:


import cv2
import matplotlib.pyplot as plt
image=cv2.imread("test.jpg")
image_in_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_in_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
suggested_size_rgb=(256,256)
suggested_size_hsv=(256,256)
resize_methods=[cv2.INTER_LINEAR,cv2.INTER_NEAREST,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4]
resized_image_rgb=[]
resized_image_hsv=[]
plt.subplot(2,4,1)
plt.imshow(image_in_rgb)
plt.title('original rgb image')



# In[43]:


plt.subplot(2,4,1)
plt.imshow(image_in_hsv)
plt.title('original image in hsv')
for i,j in enumerate(resize_methods,start=2):
    resized_rgb=cv2.resize(image_in_rgb,suggested_size_rgb,interpolation=j)
    resized_image_rgb.append((f'Resized RGB({j})',resized_rgb))
    plt.subplot(2,4,1)
    plt.imshow(resized_rgb)
    plt.title('f Resized RGB({j})')


# In[46]:


for i,j in enumerate(resize_methods,start=2):
    resized_hsv=cv2.resize(image_in_hsv,suggested_size_hsv,interpolation=j)
    resized_image_hsv.append((f'Resized HSV ({j})',resized_hsv))
    plt.subplot(2,4,1)
    plt.imshow(resized_hsv)
    plt.title('resized hsv')
plt.tight_layout()
plt.show()


# In[61]:


image=cv2.imread("test.jpg")
image_in_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_in_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
angle=45
scale_factor=1.5
translation_matrix=np.float32([[1,0,20],[0,2,80]])
shear_matrix=np.float32([[1,0.5,2],[0.5,1,2]])
image_rotated=cv2.warpAffine(image_in_rgb,cv2.getRotationMatrix2D((image_in_rgb.shape[0]//2,image_in_rgb.shape[1]//2),angle,1),
                             (image_in_rgb.shape[0],image_in_rgb.shape[1]))
image_scaled=cv2.resize(image_in_rgb,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_LINEAR)
image_translated=cv2.warpAffine(image_in_rgb,translation_matrix, (image_in_rgb.shape[0],image_in_rgb.shape[1]))
image_sheared=cv2.warpAffine(image_in_rgb,shear_matrix,(image_in_rgb.shape[1],image_in_rgb.shape[0]))
image_flip_hori=cv2.flip(image_in_rgb,1)
image_flip_ver=cv2.flip(image_in_rgb,0)
plt.figure(figsize=(12,5))
plt.subplot(2,4,1)
plt.imshow(image_rotated)
plt.title('rotated_image')
plt.subplot(2,4,1)
plt.imshow(image_flip_hori)
plt.title('horizentally flipped')
plt.imshow(image_flip_ver)
plt.title('vertically')
plt.imshow(image_sheared)
plt.title('sheared')


# In[63]:


a=cv2.getRotationMatrix2D((image_in_rgb.shape[0]//2,image_in_rgb.shape[1]//2),angle,1)
a


# In[75]:


import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
def loading(path):
    x=[]
    y=[]
    for i,j in enumerate(os.listdir(path)):
        class_path=os.path.join(path,j)
        if os.path.isdir(class_path): 
            for file_name in tqdm(os.listdir(class_path),desc=f"Loadin {j}"):
                file_path=os.path.join(class_path,file_name)
                image=cv2.imread(file_path)
                label=int(j)
                x.append(image)
                y.append(label)
    return np.array(x),np.array(y)
path=r"C:\Users\Super\Pictures"
x,y=loading(path)
                              
                
    


# In[81]:


import cv2 as cv
image=cv.imread("test.jpg",cv.IMREAD_GRAYSCALE)
rows,cols=image.shape
t=np.float32([[1,0,20],[0,1,20]])
d=cv.warpAffine(image,t,(rows,cols))
plt.imshow(d)
plt.title('translation')


# In[87]:


image=cv.imread("test.jpg",cv.IMREAD_GRAYSCALE)
rows,cols=image.shape
vm=np.float32([[1,0,0],[0,-1,rows],[0,0,1]])
flip=cv.warpPerspective(image,vm,(cols,rows))
plt.imshow(flip)


# In[89]:


image_shrinked=cv.resize(image,(100,100),interpolation=cv.INTER_AREA)
plt.imshow(image_shrinked)


# In[91]:


image_enlarge=cv.resize(image_shrinked,None,fx=2.5,fy=2.5,interpolation=cv.INTER_CUBIC)
plt.imshow(image_enlarge)


# In[92]:


image_crop=image[0:40,0:20]
plt.imshow(image_crop)


# In[93]:


r,c=image.shape
m=np.float32([[1,0.5,0],[0,1,0],[0,0,1]])
image_sheared=cv.warpPerspective(image,m,(int(r*1.5),int(c*1.5)))
plt.imshow(image_sheared)


# ## Scipy

# In[94]:


from scipy import constants
print(dir(constants))


# In[97]:


print(constants.milli)
print(constants.centi)
print(constants.yotta)


# In[101]:


from scipy.optimize import root
from math import cos
def eqn(x):
    return x+cos(x)
myroot=root(eqn,0)
print(myroot)


# In[102]:


import numpy as np
from scipy.sparse import csr_matrix
arr=np.array([1,0,0,0,1,0,1,1])
print(csr_matrix(arr).data)


# In[103]:


from scipy.sparse import csr_matrix
arr=np.array([[1,0,0],[0,0,1],[1,1,1]])
print(csr_matrix(arr))


# In[104]:


from scipy.sparse import csr_matrix
arr=np.array([[1,0,0],[1,0,0],[0,0,0]])
print(csr_matrix(arr).count_nonzero())


# In[105]:


mat=csr_matrix(arr)
mat.eliminate_zeros()
print(mat)


# In[106]:


mat=csr_matrix(arr)
mat.sum_duplicates()
print(mat)


# In[107]:


newarr=csr_matrix(arr).tocsc()
print(newarr)


# In[110]:


import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
arr=np.array([[0,1,2],[0,0,1],[1,0,2]])
new_arr=csr_matrix(arr)
print(new_arr)
print(connected_components(new_arr))


# In[114]:


import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
arr=np.array([[1,0,2],[0,0,1],[1,0,2]])
new_arr=csr_matrix(arr)
print(dijkstra(new_arr,return_predecessors=True,indices=0))


# ## Sklearn

# In[6]:


from sklearn.datasets import load_iris
iris=load_iris()
print(iris.DESCR)


# In[7]:


import pandas as pd
df=pd.DataFrame(iris.data)
df.head()


# In[8]:


df.columns=['sepal_length','sepal_width','petal_length','petal_width']
df.head()


# In[9]:


target=pd.DataFrame(iris.target)
target=target.rename(columns={0:'target'})
target.head()


# In[10]:


df=pd.concat([df,target],axis=1)
df.head()


# In[11]:


df.dtypes


# In[12]:


df.isnull().sum()


# In[13]:


df.describe()


# In[14]:


import seaborn as sns
import numpy as np
import pandas as pd
sns.heatmap(df.corr(),annot=True)


# In[19]:


import matplotlib.pyplot as plt
x_index=0
y_index=1
formatter=plt.FuncFormatter(lambda i,*args:iris.target_names[int(i)])
plt.figure(figsize=(8,4))
plt.scatter(iris.data[:,x_index],iris.data[:,y_index],c=iris.target)
plt.colorbar(ticks=[0,1,2],format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.title('scatter plot with iris')
plt.tight_layout()
plt.plot()


# In[20]:


x_index=2
y_index=3
formatter=plt.FuncFormatter(lambda i,*args:iris.target_names[int(i)])
plt.figure(figsize=(8,4))
plt.scatter(iris.data[:,x_index],iris.data[:,y_index],c=iris.target)
plt.colorbar(ticks=[0,1,2],format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.title('another graph')
plt.tight_layout()
plt.show()


# In[21]:


x=df.copy()
y=x.pop('target')
y


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1,stratify=y)


# In[25]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns)
x_test-pd.DataFrame(scaler.transform(x_train),columns=x_test.columns)


# In[26]:


df.target.value_counts(normalize=True)


# In[28]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,x_train,y_train,cv=10)
print(np.mean(scores))


# In[29]:


df=pd.DataFrame(model.coef_,columns=x_train.columns)
df


# In[32]:


p=model.predict(x_test)
c_df=pd.DataFrame({'actual':y_test,'predicted':p})
c_df=c_df.reset_index(drop=True)
c_df


# In[35]:


from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test,p,labels=[2,1,0]),index=[2,1,0],columns=[2,1,0])


# In[37]:


from sklearn.metrics import classification_report
print(classification_report(y_test,p))


# In[39]:


probs=model.predict_proba(x_test)
p=pd.DataFrame(model.predict_proba(x_test),columns=[2,1,0])
p.head()


# In[45]:


from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
x_f=iris.feature_names
y_f=iris.target_names
print(x_f)
print(y_f)
print(type(x))
print(x[:5])


# In[48]:


from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[50]:


# training the model on training set 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(x_train, y_train) 
  


# In[52]:


# making predictions on the testing set 
y_pred = knn.predict(x_test) 
  
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("kNN model accuracy:", metrics.accuracy_score(y_test, y_pred)) 
  
# making prediction for out of sample data 
sample = [[3, 5, 4, 2], [2, 3, 5, 4]] 
preds = knn.predict(sample) 
pred_species = [iris.target_names[p] for p in preds] 
print("Predictions:", pred_species) 


# ## Folium
# 

# In[53]:


import pandas as pd
import folium
m=folium.Map(location=[80,-45],zoom_start=4)
m.save('my_map.html')


# In[55]:


url="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
state_geo=f"{url}/us-states.json"
state_unemployment = f"{url}/US_Unemployment_Oct2012.csv"
state_data = pd.read_csv(state_unemployment)


# In[56]:


folium.Choropleth(
   
      # geographical locations
    geo_data = state_geo,                     
    name = "choropleth",
   
      # the data set we are using
    data = state_data,                        
    columns = ["State", "Unemployment"],     
   
      # YlGn refers to yellow and green
    fill_color = "YlGn",                      
    fill_opacity = 0.7,
    line_opacity = .1,
      key_on = "feature.id",
    legend_name = "Unemployment Rate (%)",
).add_to(m)                                 
 
m.save('final_map.html')


# In[70]:


import folium
from folium.plugins import MarkerCluster
import pandas as pd


# In[71]:


my_map=folium.Map(location=[40.015, -105.2705],zoom_start=13)
my_map


# In[72]:


folium.Marker(location=[40.007153, -105.266930],popup='CU Boulder').add_to(my_map)
my_map


# In[73]:


folium.Marker(location=[40.013501, -105.251889],popup='US satets').add_to(my_map)
my_map


# In[74]:


#with tiles
m=folium.Map(location=[40.007153, -105.266930],tiles='stamen Toner')
m


# In[80]:


import folium
poly_map=folium.Map(location=[40.007153, -105.266930],zoom_start=13)
folium.RegularPolygonMarker(location=[40.007153, -105.266930],popup='CU Boulder',fill_color='#00ff40'
                           ,number_of_sides=3,radius=10).add_to(poly_map)
folium.RegularPolygonMarker(location=[40.009837, -105.241905],popup='SEEC east',fill_color='#ff0000'
                           ,number_of_sides=5,radius=10).add_to(poly_map)
folium.RegularPolygonMarker(location=[40,-95],popup='start zoo',fill_color='#ff0000',
                           number_of_sides=8,radius=10).add_to(poly_map)
poly_map


# In[81]:


kol=folium.Map(location=[40,-95],tiles='openstreetmap',zoom_start=12)
kol


# In[83]:


tool_tip_1="this is memorial garden"
tool_tip_2="this is eden garden"
folium.Marker(location=[22.54472, 88.34273],popup='memorial garden',tooltip=tool_tip_1).add_to(kol)
folium.Marker(location=[22.56487826917627, 88.34336378854425],popup='eden garden',tooltip=tool_tip_2).add_to(kol)
kol


# In[84]:


folium.Marker(location=[22.55790780507432, 88.35087264462007],popup='indian Museum'
              ,icon=folium.Icon(color="red",icon="info sign")).add_to(kol)
kol


# In[85]:


kol2 = folium.Map(location=[22.55790780507432, 88.35087264462007], tiles="Stamen Toner", zoom_start=13)
kol2


# In[89]:


folium.Circle(location=[22.585728381244373, 88.41462932675563],popup='kolkata'
             ,radius=1500,color="blue",fill=True).add_to(kol2)
kol2


# In[88]:


folium.Circle(location=[22.56602918189088, 88.36508424354102],popup='street',
             radius=1000,color="red",fill=True).add_to(kol2)
kol2


# In[90]:


pak=folium.Map(location=[30,69],tiles='openstreetmap',zoom_start=13)
pak


# In[92]:


loc= [(33.738045, 73.084488),(32.082466, 72.669128),(30.18414, 67.00141) ,
      (24.860966, 66.990501)]
folium.PolyLine(locations=loc,line_opacity=0.5).add_to(pak)
pak


# In[94]:


df=pd.read_csv("state wise centroids_2011.csv")
df.head()


# In[95]:


india2=folium.Map(location=[20.180862078886562, 78.77642751195584],tiles='openstreetmap',zoom_start=14)
india2


# In[102]:


for i in range(0,35):
    state=df["State"][i]
    lat=df["Latitude"][i]
    long=df["Longitude"][i]
    folium.Marker(location=[lat,long],popup=state,tooltip=state).add_to(india2)
india2


# In[101]:


for i in range (0,35):
    state=df["State"][i]
    lat=df["Latitude"][i]
    long=df["Longitude"][i]
    folium.Marker(
    [lat, long], popup=state, tooltip=state).add_to(india2)
 
india2


# ## Beautiful soup
# 

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# In[3]:


url=requests.get("https://content.codecademy.com/courses/beautifulsoup/cacao/index.html")
if url.status_code==200:
    print(url.text)
else:
    print("failed to fetch")


# In[4]:


soup=BeautifulSoup(url.content,'html.parser')
print(soup)


# In[5]:


rating_c=soup.find_all(attrs={"class":"Rating"})
coca_percent=soup.find_all(attrs={"class":"CocaPercent"})
r=[]
c=[]


# In[9]:


for i in rating_c[1:]:
    r.append(i.get_text())
for j in coca_percent[1:]:
    p=float(j.get_text().strip('%'))
    c.append(p)
data={"Rating":r,"CocaPercent":c}
df=pd.DataFrame.from_dict(data)
df


# ## Lab 13

# In[10]:


url='https://www.coingecko.com/'
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
soup=requests.get(url,headers=headers)
response=BeautifulSoup(soup.text,'html.parser')
response


# In[11]:


title=response.find("title").get_text()
title


# In[12]:


nav=response.find("nav")
nav


# In[13]:


para=response.find("p")
para


# In[19]:


links=response.find_all("a")
for link in links:
    print(link.get("href"))
    


# In[20]:


data=response.find("ul")
for i in data:
    print(i.get_text())


# ## lab 12 Matplot
# 

# In[21]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[29]:


x=np.arange(0,10)
y=x**2
print(y)
print(x)
plt.plot(x,y,color="blue",linestyle='--')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('line plot')
plt.show()


# In[33]:


x=np.random.rand(1,10)
y=np.random.rand(1,10)
sizes=np.random.randint(1,10)
plt.scatter(x,y,s=sizes,alpha=0.7)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('scatter plot')
c=plt.colorbar()
c.set_label('sizes')
plt.show()


# In[40]:


catagory=['A','B','C','D','E']
arr=np.array([1,2,3,4,5])
e=np.array([1,1.5,0.5,0.9,0.3])
plt.bar(catagory,arr,yerr=e,color='green')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Bar Chart')
plt.show()


# In[43]:


data=np.random.normal(size=1000)
plt.hist(data,bins=20,color='blue')
plt.hist(data,bins=30,color='red')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('histogram')
plt.show()


# In[51]:


x1 = x**2
y1 = x1**3
x2 = x2**3
y2 = x2**3
x3 = x3+1
y3 = np.exp(x3)
fig,axs=plt.subplots(2,2,figsize=(12,4))
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


# ## lab 11 Numpy And Pandas
# 

# In[52]:


import pandas as pd
df=pd.read_csv('hypothetical_dataset - hypothetical_dataset.csv',delimiter=',')
df.head()


# In[53]:


df.describe()


# In[54]:


df.dtypes


# In[56]:


df.isnull().sum()


# In[57]:


df=df.fillna(df.mean())
df


# In[58]:


df.dropna()


# In[59]:


df['Education Level']=df['Education Level'].astype('category')
df


# In[61]:


corr=df.corr()
print(corr)


# In[62]:


df.corr()


# In[66]:


x='Age'
y='Education Level'
plt.scatter(df[x],df[y])
plt.show()


# In[70]:


plt.hist(df[x],bins=20,edgecolor='black')
plt.show()


# ## lab 10 pandas
# 

# In[2]:


import pandas as pd
df=pd.read_csv('Titanic-Dataset.csv',delimiter=',')
df


# In[5]:


x=df['Fare'].max()
maximum=df[(df['Fare']==x)]
maximum


# In[8]:


x=df['Survived']
survived=df[(df['Survived']==1)]
survived



# In[9]:


count=df['Pclass'].value_counts()
count


# In[16]:


import matplotlib.pyplot as plt
x=df['Sex']
y=df['Age']
plt.bar(x,y)
plt.show()


# In[17]:


df.isnull().sum()


# In[18]:


df=df.fillna(df['Age'].mean())
df


# In[19]:


df['Embarked']=df['Embarked'].astype('category')
df


# In[24]:


df=df.dropna(subset=['Age','Cabin'])
df


# In[25]:


df['Family_size']=df['Parch']+df['SibSp']
df


# In[29]:


mean_f=df['Fare'].mean()
df_filter=df[df['Fare']>mean_f]
df_filter


# In[30]:


x=df['Survived']
y=df['Pclass']
plt.bar(x,y)
plt.show()


# In[31]:


import seaborn as sns
numerical_features = df.select_dtypes(include=['float64', 'int64'])
corre=numerical_features.corr()
sns.heatmap(corre,annot=True)
plt.show()


# In[32]:


x=df['Age']
y=df['PassengerId']
plt.hist(x,bins=20)
plt.show()


# In[33]:


emb=df['Embarked'].value_counts()
emb


# In[35]:


avg=emb/len(df['Embarked'])*100
avg


# In[37]:


x=df['SibSp']
y=df['Parch']
plt.scatter(x,y)
plt.show()


# In[38]:


sorting=df['Age'].sort_values()
sorting


# In[39]:


group=df.groupby('PassengerId')['Fare'].mean()
group


# In[40]:


g=df[df['Age']>=18]
l=df[df['Age']<=18]
print(g)
print(l)


# In[45]:


passenger_counts = df.groupby(['Pclass', 'Survived']).size().reset_index(name='Passenger_Count')

# Display the result
print(passenger_counts)


# ## Lab 10 Numpy 

# In[47]:


import numpy as np
arr=np.arange(0,9)
arr


# In[53]:


arr1=np.random.randint(1,10,(3,3))
arr1


# In[54]:


arr2=np.random.randint(1,10,(3,3))
arr2


# In[55]:


add=arr1+arr2
add


# In[57]:


mul=arr1*arr2
mul


# In[58]:


sub=arr1-arr2
sub


# In[60]:


arr.mean()


# In[61]:


arr.std()


# In[63]:


arr1=np.median(arr)
arr1


# In[77]:


arr=np.arange(1,10)
mask1=(arr%2==0)
mask1


# In[78]:


arr[mask1]


# In[79]:


arr.reshape(3,3)


# In[80]:


h=np.hstack(arr)
h


# In[82]:


v=np.vstack(arr)
v


# In[84]:


arr=np.array([[n+m*2 for n in range(3)]for m in range(3)])
arr*arr


# In[85]:


mini=np.max(arr)
mini


# In[86]:


index=np.argmax(arr)
index


# In[87]:


arr=np.random.randint(1,10,(3,3))
arr


# In[88]:


r=np.sum(arr,axis=1)
r


# In[89]:


c=np.sum(arr,axis=0)
c


# In[97]:


df=np.genfromtxt('historical_stock_prices.csv',delimiter=',',dtype=float)
df


# In[99]:


Close=df[:, 4]
daily_returns = np.diff(Close) / Close[:1]
daily_returns


# In[100]:


maxi=np.max(df)
maxi


# In[101]:


import random

class NumberGuessingGame:
    def __init__(self, max_number, max_attempts):
        self.random_number = random.randint(1, max_number)
        self.max_attempts = max_attempts
        self.attempts_left = max_attempts
        self.is_won = False

    def guess_number(self, user_guess):
        if self.attempts_left <= 0:
            print("Out of attempts. Game over.")
            return

        self.attempts_left -= 1

        if user_guess == self.random_number:
            print("Congratulations! You guessed the correct number.")
            self.is_won = True
        elif user_guess < self.random_number:
            print("Too low. Try again.")
        else:
            print("Too high. Try again.")

        self.is_game_over()

    def is_game_over(self):
        if self.is_won or self.attempts_left <= 0:
            print("Game over. The correct number was", self.random_number)
            return True
        else:
            return False

    def play(self):
        print("Welcome to the Number Guessing Game!")
        print(f"Guess a number between 1 and {max_number}")

        while not self.is_game_over():
            try:
                user_guess = int(input("Enter your guess: "))
                self.guess_number(user_guess)
            except ValueError:
                print("Invalid input. Please enter a valid number.")

# Example Usage:
if __name__ == "__main__":
    max_number = 100  # You can change this value according to your preference
    max_attempts = 5  # You can change this value according to your preference

    game = NumberGuessingGame(max_number, max_attempts)
    game.play()


# In[102]:


class ListOperations:
    def __init__(self, initial_list=None):
        self.my_list = initial_list if initial_list else []

    def add_element(self, element):
        self.my_list.append(element)

    def remove_element(self, element):
        if element in self.my_list:
            self.my_list.remove(element)
        else:
            print(f"{element} not found in the list.")

    def find_maximum(self):
        if not self.my_list:
            print("List is empty. Cannot find maximum.")
            return None
        return max(self.my_list)

    def find_minimum(self):
        if not self.my_list:
            print("List is empty. Cannot find minimum.")
            return None
        return min(self.my_list)

# Example Usage:
if __name__ == "__main__":
    # Create an instance of ListOperations with an initial list
    my_list_operations = ListOperations(initial_list=[3, 7, 1, 9, 4, 5])

    # Add an element to the list
    my_list_operations.add_element(8)

    # Remove an element from the list
    my_list_operations.remove_element(4)

    # Find and print the maximum value in the list
    max_value = my_list_operations.find_maximum()
    print("Maximum Value:", max_value)

    # Find and print the minimum value in the list
    min_value = my_list_operations.find_minimum()
    print("Minimum Value:", min_value)

    # Print the current state of the list
    print("Current List:", my_list_operations.my_list)


# In[ ]:


import random

class HangmanGame:
    def __init__(self, word_list, max_attempts=6):
        self.word_to_guess = random.choice(word_list).upper()
        self.guessed_letters = set()
        self.max_attempts = max_attempts
        self.attempts_left = max_attempts
        self.is_won = False
        self.is_lost = False

    def display_word(self):
        displayed_word = ""
        for letter in self.word_to_guess:
            if letter in self.guessed_letters:
                displayed_word += letter + " "
            else:
                displayed_word += "_ "
        return displayed_word.strip()

    def make_guess(self, letter):
        letter = letter.upper()

        if letter in self.guessed_letters:
            print(f"You already guessed the letter '{letter}'. Try again.")
        else:
            self.guessed_letters.add(letter)

            if letter not in self.word_to_guess:
                self.attempts_left -= 1

            self.is_game_over()

    def is_game_over(self):
        if set(self.word_to_guess) <= self.guessed_letters:
            self.is_won = True
            print("Congratulations! You've guessed the word:", self.word_to_guess)
        elif self.attempts_left <= 0:
            self.is_lost = True
            print("Game over. The word was:", self.word_to_guess)

    def play(self):
        print("Welcome to Hangman! Try to guess the word.")
        print(self.display_word())

        while not self.is_game_over():
            letter_guess = input("Enter a letter: ")

            if len(letter_guess) == 1 and letter_guess.isalpha():
                self.make_guess(letter_guess)
            else:
                print("Invalid input. Please enter a single letter.")

            print("Word:", self.display_word())
            print(f"Attempts left: {self.attempts_left}")

# Example Usage:
if __name__ == "__main__":
    word_list = ["python", "hangman", "programming", "developer", "computer"]
    hangman_game = HangmanGame(word_list)

    hangman_game.play()


# ## Lists and Dictonaries

# In[2]:


list1 = ["apple","cherry", "orange", "kiwi","melon", "mango"]
print(list1)
list1.remove("cherry")
print(list1)


# In[3]:


list2=['banaaba']
list1.insert(-1,list2)
print(list1)


# In[5]:


my_list = [1, 4, 56, 2, 4, 12, 6, 89, 11, 0]
while my_list:
    my_list.pop()
print(my_list)


# In[6]:


import random

def find_min_max(numbers):
    if not numbers:
        return None, None

    # Initialize min_val and max_val with the first element of the list
    min_val = max_val = numbers[0]

    for num in numbers:
        if num < min_val:
            min_val = num
        elif num > max_val:
            max_val = num

    return min_val, max_val

# Get the size of the list from the user
try:
    size = int(input("Enter the size of the list: "))
except ValueError:
    print("Invalid input. Please enter a valid integer.")
    exit()

# Populate the list with random numbers
random_numbers = [random.randint(1, 100) for _ in range(size)]

# Display the generated list
print("Generated List:", random_numbers)

# Find the min and max values without using built-in functions
min_value, max_value = find_min_max(random_numbers)

# Display the results
print("Minimum Value:", min_value)
print("Maximum Value:", max_value)


# In[1]:


import pandas as pd
import numpy as np
data=[0,0,9,8]
datas=pd.Series(data)
print(datas.values)


# In[ ]:




