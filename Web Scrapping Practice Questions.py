#!/usr/bin/env python
# coding: utf-8

# In[35]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[55]:


import requests
from bs4 import BeautifulSoup
import csv

url = 'https://www.coingecko.com/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')
print(soup)


# In[56]:


t=soup.find("title").get_text()
t


# In[57]:


t=soup.find("nav")
t


# In[58]:


para=soup.find("p")
if para:
    print(para)
else:
    print("error ,not found:")


# In[60]:


list_tags=soup.find_all("a")
if list_tags:
    print("list_tags")
    for i in list_tags:
        print(i.get("href"))
else:
    print("no links found:")


# In[64]:


l=soup.find("ul")
if l:
        print("items:")
        for item in l:
            print(item.get_text())
        else:
            print("not found:")
else:
    print("no ul found:")


# In[ ]:




