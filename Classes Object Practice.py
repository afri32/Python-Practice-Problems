#!/usr/bin/env python
# coding: utf-8

# In[2]:


# A Sample class with init method
class Person:
 
    # init method or constructor
    def __init__(self, name):
        self.name = name
 
    # Sample Method
    def say_hi(self):
        print('AoA, my name is', self.name)
 
 
p = Person('Ali')
p.say_hi()


# In[4]:


# A Sample class with init method
class Person:
 
    # init method or constructor
    def __init__(self, name):
        self.name = name
 
    # Sample Method
    def say_hi(self):
        print('AoA, my name is', self.name)
 
 
# Creating different objects
p1 = Person('Ali')
p2 = Person('Fatima')
p3 = Person('Hamza')
 
p1.say_hi()
p2.say_hi()
p3.say_hi()


# In[5]:


class mynumber:
    def __init__(self, value):
        self.value = value
     
    def print_value(self):
        print(self.value)
 
obj1 = mynumber(17)
obj1.print_value()


# In[6]:


class Subject:
 
    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2
 
 
obj = Subject('Maths', 'Science')
print(obj.attr1) 
print(obj.attr2)


# In[2]:


class check:
    def __init__(self):
        print("Address of self = ",id(self))
 
obj = check()
print("Address of class object = ",id(obj))


# In[8]:


class car():
     
    # init method or constructor
    def __init__(self, model, color):
        self.model = model
        self.color = color
         
    def show(self):
        print("Model is", self.model )
        print("color is", self.color )
         
# both objects have different self which contain their attributes
audi = car("audi a4", "blue")
ferrari = car("ferrari 488", "green")
 
audi.show()     # same output as car.show(audi)
ferrari.show()  # same output as car.show(ferrari)
 
print("Model for audi is ",audi.model)
print("Colour for ferrari is ",ferrari.color)


# In[12]:


set1 = {1, 2, 3, {4, 5}}
set2 = {3, 4, {5, 6, {7, 8}}}
set3 = {{9, 10}, {11, {12, 13}}, 14}
set4 = set([1,2,3,4,5,1,2])
set5 = set([6,7,8,3,4,5,6,2])
element1 = 4 in set1  
element2 = 8 in set2  
element3 = {12, 13} in set3
set3.remove((11))
print (set3)
print (element1)
print (element2)
print (element3)
print (set5 ^ set4)
print (set4 | set5)
print (set4 & set5)
print (set5  set4)


# In[30]:


company_structure = {
    "CEO": {
        "Name": "John Doe",
        "Departments": {
            "Engineering": {
                "Managers": ["Alice", "Bob"],
                "Teams": {
                    "Frontend": ["Eva", "Frank"],
                    "Backend": ["Grace", "Harry"]
                }
            },
            "Sales": {
                "Managers": ["Cathy", "David"],
                "Teams": {
                    "Domestic": ["Ivy", "Jack"],
                    "International": ["Kevin", "Linda"]
                }
            }
        }
    }
}

print (company_structure.keys())
print (company_structure[‘Teams’])
print (company_structure[‘Sales’])
print (company_structure[‘Departments’].values())
print (company_structure[‘CEO’].keys())
print (company_structure[‘Managers].values())
print (company_structure[‘Departments’][‘ Engineering’].values())
print (company_structure[‘Departments’][‘ Engineering’][‘Teams’][1])
print (company_structure[‘Departments’][‘ Engineering’][‘Teams’]
[‘Frontend’][1])
print (company_structure[‘Sales][‘Teams’][‘Domestic’].keys())


# In[31]:


company_structure = {
    "CEO": {
        "Name": "John Doe",
        "Departments": {
            "Engineering": {
                "Managers": ["Alice", "Bob"],
                "Teams": {
                    "Frontend": ["Eva", "Frank"],
                    "Backend": ["Grace", "Harry"]
                }
            },
            "Sales": {
                "Managers": ["Cathy", "David"],
                "Teams": {
                    "Domestic": ["Ivy", "Jack"],
                    "International": ["Kevin", "Linda"]
                }
            }
        }
    }
}

# 1. Corrected and simplified prints:
print("Keys in company_structure:", company_structure.keys())
print("Keys under CEO:", company_structure["CEO"].keys())
print("Values under Departments:", company_structure["CEO"]["Departments"].values())

# 2. Corrected:
print("Managers under Engineering:", company_structure["CEO"]["Departments"]["Engineering"]["Managers"].values())

# 3. Corrected and removed the space:
print("Values under Engineering:", company_structure["CEO"]["Departments"]["Engineering"].values())

# 4. Accessing elements within lists:
print("Second item in Teams under Engineering:", company_structure["CEO"]["Departments"]["Engineering"]["Teams"]["Backend"][1])


# In[33]:


set1 = {1, 2, 3, {4, 5}}
set2 = {3, 4, {5, 6, {7, 8}}}
set3 = {{9, 10}, {11, {12, 13}}, 14}
set4 = set([1,2,3,4,5,1,2])
set5 = set([6,7,8,3,4,5,6,2])
element1 = 4 in set1  
element2 = 8 in set2  
element3 = {12, 13} in set3
set3.remove((11))
print (set3)
print (element1)
print (element2)
print (element3)
print (set5 ^ set4)
print (set4 | set5)
print (set4 & set5)
print (set5 - set4)


# In[ ]:




