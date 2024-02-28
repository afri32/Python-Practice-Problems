#!/usr/bin/env python
# coding: utf-8

# # Numpy -  multidimensional data arrays

# Credits:J.R. Johansson (jrjohansson at gmail.com)
# 

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# ## Introduction

# The `numpy` package (module) is used in almost all numerical computation using Python. It is a package that provide high-performance vector, matrix and higher-dimensional data structures for Python. It is implemented in C and Fortran so when calculations are vectorized (formulated with vectors and matrices), performance is very good. 
# 
# To use `numpy` you need to import the module, using for example:

# In[2]:


from numpy import *


# In the `numpy` package the terminology used for vectors, matrices and higher-dimensional data sets is *array*. 
# 
# 

# ## Creating `numpy` arrays

# There are a number of ways to initialize new numpy arrays, for example from
# 
# * a Python list or tuples
# * using functions that are dedicated to generating numpy arrays, such as `arange`, `linspace`, etc.
# * reading data from files

# ### From lists

# For example, to create new vector and matrix arrays from Python lists we can use the `numpy.array` function.

# In[3]:


# a vector: the argument to the array function is a Python list
v = array([1,2,13,14,15])

v


# In[4]:


# a matrix: the argument to the array function is a nested Python list
M = array([[1, 2], [3, 4],[11, 12],[111, 112]])

M


# The `v` and `M` objects are both of the type `ndarray` that the `numpy` module provides.

# In[5]:


type(v), type(M)


# The difference between the `v` and `M` arrays is only their shapes. We can get information about the shape of an array by using the `ndarray.shape` property.

# In[6]:


v.shape


# In[7]:


M.shape


# The number of elements in the array is available through the `ndarray.size` property:

# In[8]:


M.size


# Equivalently, we could use the function `numpy.shape` and `numpy.size`

# In[9]:


shape(M)


# In[10]:


size(M)


# So far the `numpy.ndarray` looks awefully much like a Python list (or nested list). Why not simply use Python lists for computations instead of creating a new array type? 
# 
# There are several reasons:
# 
# * Python lists are very general. They can contain any kind of object. They are dynamically typed. They do not support mathematical functions such as matrix and dot multiplications, etc. Implementing such functions for Python lists would not be very efficient because of the dynamic typing.
# * Numpy arrays are **statically typed** and **homogeneous**. The type of the elements is determined when the array is created.
# * Numpy arrays are memory efficient.
# * Because of the static typing, fast implementation of mathematical functions such as multiplication and addition of `numpy` arrays can be implemented in a compiled language (C and Fortran is used).
# 
# Using the `dtype` (data type) property of an `ndarray`, we can see what type the data of an array has:

# In[11]:


M.dtype


# We get an error if we try to assign a value of the wrong type to an element in a numpy array:

# In[12]:


M[0,0] = "hello"


# If we want, we can explicitly define the type of the array data when we create it, using the `dtype` keyword argument: 

# In[13]:


M = array([[1, 2], [3, 4]], dtype=complex)

M


# Common data types that can be used with `dtype` are: `int`, `float`, `complex`, `bool`, `object`, etc.
# 
# We can also explicitly define the bit size of the data types, for example: `int64`, `int16`, `float128`, `complex128`.

# ### Using array-generating functions

# For larger arrays it is inpractical to initialize the data manually, using explicit python lists. Instead we can use one of the many functions in `numpy` that generate arrays of different forms. Some of the more common are:

# #### arange

# In[14]:


# create a range

x = arange(-10, -200, -20) # arguments: start, stop, step

x


# In[15]:


x = arange(-1, 2, 0.1)

x


# #### linspace and logspace

# In[16]:


# using linspace, both end points ARE included
linspace(5, 20, 25)
#linspace(0, 10, 25) will produce an array with 25 values that start from 0, increase evenly, and end at 10. 
#The resulting array will have values spaced by (10 - 0) / (25 - 1) = 0.41666667 approximately.


# In[17]:


logspace(0, 10, 10, base=e)
#logspace(0, 10, 10, base=e) generates an array with 10 values that are logarithmically spaced between 1 (e^0) and approximately 22026.47 (e^10).


# #### mgrid

# In[18]:


x, y = mgrid[0:5, 0:4] # similar to meshgrid in MATLAB


# In[19]:


x


# In[20]:


y


# In[21]:


# Sample code for generation of first example
import numpy as np
# from matplotlib import pyplot as plt
# pyplot imported for plotting graphs
 
x = np.linspace(-4, 4, 9)
 
# numpy.linspace creates an array of
# 9 linearly placed elements between
# -4 and 4, both inclusive 
y = np.linspace(-5, 5, 11)
 
# The meshgrid function returns
# two 2-dimensional arrays 
x_1, y_1 = np.meshgrid(x, y)
 
print("x_1 = ")
print(x_1)
print("y_1 = ")
print(y_1)


# #### random data

# In[22]:


from numpy import random


# In[23]:


# uniform random numbers in [0,1]
random.rand(5,3)


# In[24]:


# standard normal distributed random numbers
random.randn(2,2)


# In[25]:


#Uniformly distributed random numbers on an interval have equal probability of being selected or happening.
#Normally distributed random numbers on an interval have probabilities that follow the normal distribution 
#bell curve, so numbers closer to the mean are more likely to be selected or to happen.


# #### diag

# In[26]:


# a diagonal matrix
diag([1,2,13,4])


# In[27]:


# diagonal with offset from the main diagonal
diag([1,2,3,45], k=5) 


# In[28]:


# Python Programming illustrating 
# numpy.diag method 
  
import numpy as np
  
# matrix creation by array input 
a = np.matrix([[1, 21, 30],  
                 [63 ,434, 3],  
                 [54, 54, 56]]) 
  
print("Main Diagonal elements : \n", np.diag(a), "\n") 
  
print("Diagonal above main diagonal : \n", np.diag(a, 1), "\n") 
  
print("Diagonal below main diagonal : \n", np.diag(a, -1)) 


# #### zeros and ones

# In[29]:


zeros((3,3))


# In[30]:


ones((3,3))


# ## File I/O

# ### Comma-separated values (CSV)

# A very common file format for data files is comma-separated values (CSV), or related formats such as TSV (tab-separated values). To read data from such files into Numpy arrays we can use the `numpy.genfromtxt` function. For example, 

# In[31]:


get_ipython().system('head stockholm_td_adj.dat')


# In[32]:


data = genfromtxt('stockholm_td_adj.dat')


# In[33]:


data.shape


# In[42]:


fig, ax = plt.subplots(figsize=(14,4))
ax.plot(data[:,0]+data[:,1]/12.0+data[:,2]/365, data[:,5])
ax.axis('tight')
ax.set_title('tempeatures in Stockholm')
ax.set_xlabel('year')
ax.set_ylabel('temperature (C)');


# In[43]:


data[:,1]/12.0


# Using `numpy.savetxt` we can store a Numpy array to a file in CSV format:

# In[44]:


M = random.rand(3,3)

M


# In[45]:


savetxt("random-matrix.csv", M)


# In[46]:


get_ipython().system('cat random-matrix.csv')


# In[47]:


savetxt("random-matrix.csv", M, fmt='%.5f') # fmt specifies the format

get_ipython().system('cat random-matrix.csv')


# ### Numpy's native file format

# Useful when storing and reading back numpy array data. Use the functions `numpy.save` and `numpy.load`:

# In[48]:


save("random-matrix.npy", M)

get_ipython().system('file random-matrix.npy')


# In[49]:


load("random-matrix.npy")


# ## More properties of the numpy arrays

# In[50]:


M.itemsize # bytes per element


# In[51]:


M.nbytes # number of bytes


# In[52]:


M.ndim # number of dimensions


# ## Manipulating arrays

# ### Indexing

# We can index elements in an array using square brackets and indices:

# In[53]:


# v is a vector, and has only one dimension, taking one index
v[0]


# In[54]:


# M is a matrix, or a 2 dimensional array, taking two indices 
M[1,1]


# If we omit an index of a multidimensional array it returns the whole row (or, in general, a N-1 dimensional array) 

# In[55]:


M


# In[56]:


M[1]


# The same thing can be achieved with using `:` instead of an index: 

# In[57]:


M[1,:] # row 1


# In[58]:


M[:,1] # column 1


# We can assign new values to elements in an array using indexing:

# In[59]:


M[0,0] = 1


# In[60]:


M


# In[61]:


# also works for rows and columns
M[1,:] = 0
M[:,2] = -1


# In[62]:


M


# ### Index slicing

# Index slicing is the technical name for the syntax `M[lower:upper:step]` to extract part of an array:

# In[63]:


A = array([1,2,3,4,5])
A


# In[64]:


A[1:3]
X=A[:,4]


# Array slices are *mutable*: if they are assigned a new value the original array from which the slice was extracted is modified:

# In[65]:


A[1:3] = [-2,-3]

A


# We can omit any of the three parameters in `M[lower:upper:step]`:

# In[66]:


A[::] # lower, upper, step all take the default values


# In[67]:


A[::2] # step is 2, lower and upper defaults to the beginning and end of the array


# In[68]:


A[:3] # first three elements


# In[69]:


A[3:] # elements from index 3


# Negative indices counts from the end of the array (positive index from the begining):

# In[70]:


A = array([1,2,3,4,5])


# In[71]:


A[-1] # the last element in the array


# In[72]:


A[-3:] # the last three elements


# Index slicing works exactly the same way for multidimensional arrays:

# In[73]:


A = array([[n+m*10 for n in range(5)] for m in range(5)])

A


# In[74]:


# a block from the original array
A[1:4:1, 1:4:1]


# In[75]:


# strides
A[::2, ::2]


# ### Fancy indexing

# Fancy indexing is the name for when an array or list is used in-place of an index: 

# In[80]:


row_indices = [1, 2, 4]
A[row_indices]


# In[82]:


col_indices = [2, 2, 3] # remember, index -1 means the last element
A[row_indices, col_indices]


# ### Boolean Indexing

# We can also use index masks: If the index mask is an Numpy array of data type `bool`, then an element is selected (True) or not (False) depending on the value of the index mask at the position of each element: 

# In[88]:


B = array([n for n in range(5)])
B


# In[89]:


row_mask = array([True, False, True, False, False])
B[row_mask]


# In[90]:


# same thing
row_mask = array([1,0,1,0,0], dtype=bool)
B[row_mask]


# This feature is very useful to conditionally select elements from an array, using for example comparison operators:

# In[91]:


x = arange(0, 10, 0.5)
x


# In[95]:


mask = (5 < x) * (x < 7.5)

mask


# In[97]:


x[mask]


# ## Functions for extracting data from arrays and creating arrays

# ### where

# The index mask can be converted to position index using the `where` function

# In[98]:


indices = where(mask)

indices


# In[99]:


x[indices] # this indexing is equivalent to the fancy indexing x[mask]


# ### diag

# With the diag function we can also extract the diagonal and subdiagonals of an array:

# In[100]:


diag(A)


# In[101]:


diag(A, -1)


# ### take

# The `take` function is similar to fancy indexing described above:

# In[102]:


v2 = arange(-3,3)
v2


# In[103]:


row_indices = [1, 3, 5]
v2[row_indices] # fancy indexing


# In[104]:


v2.take(row_indices)


# But `take` also works on lists and other objects:

# In[105]:


take([-3, -2, -1,  0,  1,  2], row_indices)


# ### choose

# Constructs an array by picking elements from several arrays:

# In[108]:


which = [1, 0, 1, 0]
choices = [[-1,-2,-3,-4], [15,25,35,55]]

choose(which, choices)


# In[109]:


which = [1, 0, 1, 0]
choices = [[-2, -2, -2, -2], [5, 5, 5, 5]]

result = [choices[which[i]][i] for i in range(len(which))]

print(result)


# In[110]:


for i in range(len(which)):
    print(which[i],"=",i)


# In[111]:


len(which)


# ## Linear algebra

# Vectorizing code is the key to writing efficient numerical calculation with Python/Numpy. That means that as much as possible of a program should be formulated in terms of matrix and vector operations, like matrix-matrix multiplication.

# In[112]:


v1 = arange(0, 5000)


# In[113]:


get_ipython().run_cell_magic('time', '', 'for i in range(v1.shape[0]):\n    v1[i] *= 2 \n')


# In[114]:


v1 = arange(0, 5000)


# In[115]:


get_ipython().run_cell_magic('time', '', 'v1 *= 2\n')


# In[116]:


0.00192 / 2.13e-5  


# ### Scalar-array operations

# We can use the usual arithmetic operators to multiply, add, subtract, and divide arrays with scalar numbers.

# In[121]:


v1 = arange(0, 5)


# In[122]:


v1 * 2


# In[123]:


v1 + 2


# In[124]:


A * 2, A + 2


# ### Element-wise array-array operations

# When we add, subtract, multiply and divide arrays with each other, the default behaviour is **element-wise** operations:

# In[125]:


A * A # element-wise multiplication


# In[126]:


v1 * v1


# If we multiply arrays with compatible shapes, we get an element-wise multiplication of each row:

# In[127]:


A.shape, v1.shape


# In[128]:


v1[:, newaxis]
A.shape, v1.shape
#v1 is an existing array with a certain shape, let's say it's a 1D array.
#newaxis is used to insert a new axis into the array. 
#In NumPy, this new axis essentially converts the 1D array into a 2D array, where one of the dimensions has size
#This can be particularly useful when you want to transform a 1D array into a column or row vector.


# In[129]:


A * v1


# ### Matrix algebra

# What about matrix mutiplication? There are two ways. We can either use the `dot` function, which applies a matrix-matrix, matrix-vector, or inner vector multiplication to its two arguments: 

# In[133]:


dot(A, A)


# In[134]:


dot(A, v1)


# In[135]:


dot(v1, v1)


# In[137]:


v1*v1


# Alternatively, we can cast the array objects to the type `matrix`. This changes the behavior of the standard arithmetic operators `+, -, *` to use matrix algebra.

# In[138]:


M = matrix(A)
v = matrix(v1).T # make it a column vector


# In[139]:


v


# In[140]:


M * M


# In[141]:


M * v


# In[142]:


# inner product
v.T * v


# In[143]:


# with matrix objects, standard matrix algebra applies
v + M*v


# If we try to add, subtract or multiply objects with incomplatible shapes we get an error:

# In[144]:


v = matrix([1,2,3,4,5,6]).T


# In[145]:


shape(M), shape(v)


# In[146]:


M * v


# In[147]:


# import numpy 
import numpy as np 
  
# using np.kron() method 
NU = np.kron([1, 2, 3], [5, 10, 15]) 
  
print(NU) 


# See also the related functions: `inner`, `outer`, `cross`, `kron`, `tensordot`. Try for example `help(kron)`.

# In[148]:


# import numpy 
import numpy as np 
#Kronecker product of two lists 
# using np.kron() method 
FAST = np.kron([[1, 2, 3], [9, 8, 7]], [5, 10, 15]) 
  
print(FAST) 


# In[150]:


ar1 = np.array([[1, 3, 7], [2, 9, 4]])
ar2 = np.array([[5, 6], [8, 0], [1, 7]])
# [1,2,3][5,8,1] and [1,2,3][6,0,7]
#np.tensordot(ar1, ar2, axes=0)
np.tensordot(ar1, ar2, axes=1)


# In[151]:


import numpy as np

# Create two 2D arrays
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Perform a tensor contraction along the last dimension of A and the first dimension of B
result = np.tensordot(A, B, axes=0)
result = np.tensordot(A, B, axes=1)
print(result)


# ### Matrix computations

# #### Inverse

# In[152]:


linalg.inv(C) # equivalent to C.I 


# In[153]:


import numpy as np 
x = np.array([[1,2],[3,4]])
#AX = B => X = A-1 B
y = np.linalg.inv(x) 
print (x)
print (y)
print (np.dot(x,y))


# In[154]:


# Import required package
import numpy as py
   
# Taking a 3rd order matrix
A = py.array([[2, 3, 4],
              [-3, -3, -2],
              [-2, 1, -1]])
# Calculating the inverse of the given matrix
Ainv= py.linalg.inv(A)
print("the inverse of the matrix is= ", Ainv)


# In[155]:


C.I * C


# #### Determinant

# In[156]:


linalg.det(C)


# In[157]:


linalg.det(C.I)


# ### Data processing

# Often it is useful to store datasets in Numpy arrays. Numpy provides a number of functions to calculate statistics of datasets in arrays. 
# 
# For example, let's calculate some properties from the Stockholm temperature dataset used above.

# In[158]:


# reminder, the tempeature dataset is stored in the data variable:
shape(data)


# #### mean

# In[159]:


# the temperature data is in column 3
mean(data[:,3])


# The daily mean temperature in Stockholm over the last 200 years has been about 6.2 C.

# #### standard deviations and variance

# In[160]:


std(data[:,3]), var(data[:,3])


# #### min and max

# In[161]:


# lowest daily average temperature
data[:,3].min()


# In[162]:


# highest daily average temperature
data[:,3].max()


# #### sum, prod, and trace

# In[163]:


d = arange(0, 10)
d


# In[164]:


# sum up all elements
sum(d)


# In[165]:


# product of all elements
prod(d+1)


# In[166]:


# cummulative sum
cumsum(d)


# In[167]:


# cummulative product
cumprod(d+1)


# In[168]:


# same as: diag(A).sum()
trace(A)


# ### Computations on subsets of arrays

# We can compute with subsets of the data in an array using indexing, fancy indexing, and the other methods of extracting data from an array (described above).
# 
# For example, let's go back to the temperature dataset:

# In[170]:


get_ipython().system('head -n 3 stockholm_td_adj.dat')


# The dataformat is: year, month, day, daily average temperature, low, high, location.
# 
# If we are interested in the average temperature only in a particular month, say February, then we can create a index mask and use it to select only the data for that month using:

# In[171]:


unique(data[:,1]) # the month column takes values from 1 to 12


# In[176]:


mask_feb = data[:,1] == 2
mask_feb


# In[179]:


# the temperature data is in column 3
mean(data[mask_feb,3])


# In[180]:


data[mask_feb,3]


# In[181]:


data[mask_feb,3]


# With these tools we have very powerful data processing capabilities at our disposal. For example, to extract the average monthly average temperatures for each month of the year only takes a few lines of code: 

# In[182]:


months = arange(1,13)
monthly_mean = [mean(data[data[:,1] == month, 3]) for month in months]

fig, ax = plt.subplots()
ax.bar(months, monthly_mean)
ax.set_xlabel("Month")
ax.set_ylabel("Monthly avg. temp.");


# ### Calculations with higher-dimensional data

# When functions such as `min`, `max`, etc. are applied to a multidimensional arrays, it is sometimes useful to apply the calculation to the entire array, and sometimes only on a row or column basis. Using the `axis` argument we can specify how these functions should behave: 

# In[183]:


m = random.rand(3,3)
m


# In[184]:


# global max
m.max()


# In[185]:


# max in each column
m.max(axis=0)


# In[186]:


# max in each row
m.max(axis=1)


# Many other functions and methods in the `array` and `matrix` classes accept the same (optional) `axis` keyword argument.

# ## Reshaping, resizing and stacking arrays

# The shape of an Numpy array can be modified without copying the underlaying data, which makes it a fast operation even for large arrays.

# In[187]:


A


# In[188]:


n, m = A.shape


# In[189]:


B = A.reshape((1,n*m))
B


# In[192]:


B[0,0:5] = 5 # modify the array

B


# In[193]:


A # and the original variable is also changed. B is only a different view of the same data


# We can also use the function `flatten` to make a higher-dimensional array into a vector. But this function create a copy of the data.

# In[194]:


B = A.flatten()

B


# In[195]:


B[0:5] = 10

B


# In[196]:


A # now A has not changed, because B's data is a copy of A's, not refering to the same data


# ## Adding a new dimension: newaxis

# With `newaxis`, we can insert new dimensions in an array, for example converting a vector to a column or row matrix:

# In[197]:


v = array([1,2,3])


# In[198]:


shape(v)


# In[199]:


# make a column matrix of the vector v
v[:, newaxis]


# In[200]:


# column matrix
v[:,newaxis].shape


# In[203]:


# row matrix
v[newaxis,:].shape
shape(v)


# ## Stacking and repeating arrays

# Using function `repeat`, `tile`, `vstack`, `hstack`, and `concatenate` we can create larger vectors and matrices from smaller ones:

# ### tile and repeat

# In[204]:


a = array([[1, 2], [3, 4]])


# In[205]:


# repeat each element 3 times, by default inputs flattened array and returns flattened array
repeat(a,3)


# In[213]:


repeat(a, 3, axis=0)


# In[214]:


repeat(a, 3, axis=1)


# In[215]:


# tile the matrix 3 times 
tile(a, 3)


# ### concatenate

# In[221]:


b = array([[5, 6]])


# In[222]:


concatenate((a, b), axis=0)


# In[223]:


concatenate((a, b.T), axis=1)


# ### hstack and vstack

# In[224]:


vstack((a,b))


# In[225]:


hstack((a,b.T))


# ## Copy and "deep copy"

# To achieve high performance, assignments in Python usually do not copy the underlaying objects. This is important for example when objects are passed between functions, to avoid an excessive amount of memory copying when it is not necessary (technical term: pass by reference). 

# In[226]:


A = array([[1, 2], [3, 4]])

A


# In[227]:


# now B is referring to the same array data as A 
B = A 


# In[228]:


# changing B affects A
B[0,0] = 10

B


# In[229]:


A


# If we want to avoid this behavior, so that when we get a new completely independent object `B` copied from `A`, then we need to do a so-called "deep copy" using the function `copy`:

# In[230]:


B = copy(A)


# In[231]:


# now, if we modify B, A is not affected
B[0,0] = -5

B


# In[232]:


A


# ## Iterating over array elements

# Generally, we want to avoid iterating over the elements of arrays whenever we can (at all costs). The reason is that in a interpreted language like Python (or MATLAB), iterations are really slow compared to vectorized operations. 
# 
# However, sometimes iterations are unavoidable. For such cases, the Python `for` loop is the most convenient way to iterate over an array:

# In[233]:


v = array([1,2,3,4])

for element in v:
    print(element)


# In[234]:


M = array([[1,2], [3,4]])

for row in M:
    print("row", row)
    
    for element in row:
        print(element)


# When we need to iterate over each element of an array and modify its elements, it is convenient to use the `enumerate` function to obtain both the element and its index in the `for` loop: 

# In[235]:


for row_idx, row in enumerate(M):
    print("row_idx", row_idx, "row", row)
    
    for col_idx, element in enumerate(row):
        print("col_idx", col_idx, "element", element)
       
        # update the matrix M: square each element
        M[row_idx, col_idx] = element ** 2


# In[236]:


# each element in M is now squared
M


# ## Vectorizing functions

# As mentioned several times by now, to get good performance we should try to avoid looping over elements in our vectors and matrices, and instead use vectorized algorithms. The first step in converting a scalar algorithm to a vectorized algorithm is to make sure that the functions we write work with vector inputs.

# In[237]:


def Theta(x):
    """
    Scalar implemenation of the Heaviside step function.
    """
    if x >= 0:
        return 1
    else:
        return 0


# In[238]:


Theta(array([-3,-2,-1,0,1,2,3]))


# OK, that didn't work because we didn't write the `Theta` function so that it can handle a vector input... 
# 
# To get a vectorized version of Theta we can use the Numpy function `vectorize`. In many cases it can automatically vectorize a function:

# In[239]:


Theta_vec = vectorize(Theta)


# In[241]:


Theta_vec(array([-3,-2,-1,0,1,2,3]))


# We can also implement the function to accept a vector input from the beginning (requires more effort but might give better performance):

# In[242]:


def Theta(x):
    """
    Vector-aware implemenation of the Heaviside step function.
    """
    return 1 * (x >= 0)


# In[245]:


Theta(array([-3,-2,-1,0,1,2,3]))


# In[246]:


# still works for scalars as well
Theta(-1.2), Theta(2.6)


# ## Using arrays in conditions

# When using arrays in conditions,for example `if` statements and other boolean expressions, one needs to use `any` or `all`, which requires that any or all elements in the array evalutes to `True`:

# In[247]:


M


# In[250]:


if (M > 5).any():
    print("at least one element in M is larger than 5")
else:
    print("no element in M is larger than 5")


# In[251]:


if (M > 5).all():
    print("all elements in M are larger than 5")
else:
    print("all elements in M are not larger than 5")


# ## Type casting

# Since Numpy arrays are *statically typed*, the type of an array does not change once created. But we can explicitly cast an array of some type to another using the `astype` functions (see also the similar `asarray` function). This always create a new array of new type:

# In[252]:


M.dtype


# In[253]:


M2 = M.astype(float)

M2


# In[254]:


M2.dtype


# In[255]:


M3 = M.astype(bool)

M3


# ## Further reading

# * http://numpy.scipy.org
# * http://scipy.org/Tentative_NumPy_Tutorial
# * http://scipy.org/NumPy_for_Matlab_Users - A Numpy guide for MATLAB users.

# ## Versions

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'version_information')

get_ipython().run_line_magic('version_information', 'numpy')

