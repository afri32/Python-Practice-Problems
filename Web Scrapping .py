#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
url='https://www.zyte.com/'
data=requests.get(url)
soup = BeautifulSoup(data.text, "html.parser")
print(soup.prettify())


# In[2]:


titles=soup.find_all("title")
for t in titles:
    print(t.get_text())


# In[3]:


url='https://www.zyte.com/data-types/news-articles-scraper/?_gl=1*11yg62c*_up*MQ..*_ga*MjM1MjYyOTgyLjE3MDUzMzg5MDU.*_ga_PC3KF6FQ4T*MTcwNTMzODkwNC4xLjAuMTcwNTMzODkwNC4wLjAuMA..'
data=requests.get(url)
print(data.text)


# In[4]:


soup=BeautifulSoup(data.text,'html.parser')
print(soup.prettify())


# In[5]:


titles=soup.find_all('title')
for t in titles:
    print(t.get_text())


# In[6]:


heading=soup.find_all(['h1','h2','h3','h4','h5','h6'])
for i in heading:
    print(i.get_text())


# In[7]:


from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
url='https://www.coingecko.com/'
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
response=requests.get(url,headers=headers)
soup=BeautifulSoup(response.text,'html.parser')
print(soup)


# In[8]:


title_web=soup.find("title").get_text()
title_web


# In[9]:


navi=soup.find("nav")
navi


# In[10]:


para=soup.find("p").get_text()
para


# In[ ]:


days=0
days=int(input("enter your days for ethereum bitcoin ripple"))
list=[]
for line in ['bitcoin','ripple','ethereum']:
    url = f'https://api.coingecko.com/api/v3/coins/{line}/market_chart'
    comparison_currency='usd'
    webpage=requests.get(url,params={'vs_currency':comparison_currency,'days':days})
    if(webpage.status_code==200):
        list=list.append(webpage.json())
    else:
        print("no")
print(list)


# In[ ]:


days = 0
days = int(input("ENTER THE DESIRED NUMBER OF DAYS TO PERFORM DATA ANALYSIS FOR THE FOLLOWING COINS (BITCOIN, ETHEREUM, RIPPLE):"))
coin_data_list = []
for line in ['bitcoin', 'ethereum', 'ripple']:
    url = f'https://api.coingecko.com/api/v3/coins/{line}/market_chart'
    comparison_currency = 'usd'
    webpage = requests.get(url, params={'vs_currency': comparison_currency, 'days': days})
    coin_name = line.upper()

    if webpage.status_code == 200:
        print(f"{coin_name}'S DATA SUCCESSFULLY SCRAPPED!")
        coin_data_list.append(webpage.json())
    else:
        print(f"DATA COULD NOT BE SCRAPPED DUE TO {webpage.status_code}")
        coin_data_list.append(None)
coin_data_list


# In[ ]:




