#!/usr/bin/env python
# coding: utf-8

# ## Lab Regular Expressions

# In[12]:


import re
patter=r"#$%^+*."
def password_strong(input):
    has_lower=False
    has_upper=False
    has_digit=False
    has_pattern=False
    if(len(input)>=8):
        for i in input:
            if i.islower():
                    has_lower=True
            elif i.isupper():
                has_upper=True
            elif i.isdigit():
                has_digit=True
            elif i in patter:
                has_pattern=True
    return has_lower and has_upper and has_digit and has_pattern
str=input("enter your string:")
r=password_strong(str)
print(r)
            


# In[36]:


import numpy as np
with open('file_path.txt','r') as file:
    df=file.read()
print(df)
pattern=r'[a-zA-Z_]\w*'
var=re.findall(pattern,df)
print(var)
for i in var:
    print(i)


# In[37]:


import numpy as np
with open('file_path.txt','r') as file:
    df=file.read()
pattern=r'[a-zA-Z_]\w*'
var=re.findall(pattern,df)
print(var)
for i in var:
    print(i)


# In[41]:


import numpy as np
with open('file_path.txt','r') as file:
    df=file.read()
print(df)
pattern=r'[0-9]'
roll=re.findall(pattern,df)
print(roll)


# In[12]:


import re
with open('file_path.txt','r') as file:
    df=file.read()
print(df)
count=0
pattern=r'[a-zA-Z]'
var=re.findall(pattern,df)
for i in var:
    count=count+1
print(count)


# In[14]:


# Input the starting and ending range
start_range = int(input("Input the starting range or number: "))
end_range = int(input("Input the ending range of number: "))

# Find and print the perfect numbers within the given range
perfect_numbers = []
for num in range(start_range, end_range + 1):
    divisors_sum = sum([divisor for divisor in range(1, num) if num % divisor == 0])
    if divisors_sum == num:
        perfect_numbers.append(num)

print(f"The Perfect numbers within the given range {start_range} to {end_range}: {perfect_numbers}")


# In[15]:


import string
import random

class PasswordGenerator:
    def __init__(self, length=12, use_uppercase=True, use_lowercase=True, use_numbers=True, use_special_chars=True):
        self.length = length
        self.use_uppercase = use_uppercase
        self.use_lowercase = use_lowercase
        self.use_numbers = use_numbers
        self.use_special_chars = use_special_chars

        self.all_chars = ''
        if use_uppercase:
            self.all_chars += string.ascii_uppercase
        if use_lowercase:
            self.all_chars += string.ascii_lowercase
        if use_numbers:
            self.all_chars += string.digits
        if use_special_chars:
            self.all_chars += string.punctuation

    def generate_password(self):
        if not any([self.use_uppercase, self.use_lowercase, self.use_numbers, self.use_special_chars]):
            raise ValueError("At least one character type should be selected.")

        password = ''.join(random.choice(self.all_chars) for _ in range(self.length))
        return password

# Example usage:
password_generator = PasswordGenerator(length=16, use_uppercase=True, use_lowercase=True, use_numbers=True, use_special_chars=True)

password = password_generator.generate_password()
print("Generated Password:", password)


# In[92]:


class complex_:
    def __init__(self,imag,real):
        self.real=real
        self.imag=imag
    def __add__(self,other):
        real_sum=self.real+other.real
        imag_sum=self.imag+other.imag
        return complex_(real_sum,imag_sum)
    def __repr__(self):
        return f"{self.real} + {self.imag}j"
c1=complex_(1,2)
c2=complex_(1,3)
r=c1+c2
print(r)


# In[94]:


class matrix:
    def __init__(self,arr1,arr2):
        self.arr1=arr1
        self.arr2=arr2
    def __add__(self,other):
        c=self.arr1+self.arr2
        return c
    def __repr__(self):
        return f"{self.arr1}+{self.arr2}"
a=np.random.randint(1,100,(2,2))
b=np.random.randint(1,100,(2,2))
c1=matrix(a,b)
c2=matrix(a,b)
r=c1+c2
print(r)


# ### 
