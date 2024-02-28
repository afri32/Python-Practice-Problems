#!/usr/bin/env python
# coding: utf-8

# In[55]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[56]:


#Fetch a website URL and store it to webpage
webpage = requests.get("https://content.codecademy.com/courses/beautifulsoup/cacao/index.html")
# Check if the request was successful (status code 200)
if webpage.status_code == 200:
    # Print the content of the webpage
    print(webpage.text)
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")


# In[33]:


#Take the webpage variable and fetch the content using BeautifulSoup
soup = BeautifulSoup(webpage.content, "html.parser")
print(soup)


# In[34]:


#Store the whole text in HTML that have the class "Rating" and "CocoaPercent"
rating_column = soup.find_all(attrs={"class": "Rating"})
cocoa_percent_tags = soup.find_all(attrs={"class": "CocoaPercent"})
#Make a empty lists for Rating and CocoaPercent
ratings = []
cocoa_percents = []


# In[1]:


#Loop for inserting each table data to list
for x in rating_column[1:] :
    ratings.append(float(x.get_text()))

for td in cocoa_percent_tags[1:] :
    percent = float(td.get_text().strip('%'))
    cocoa_percents.append(percent)

#Combining both ratings and cocoa_percents list to a dictionary
data = {"Rating": ratings, "CocoaPercentage": cocoa_percents}

#Make a new Data Frame from data dictionary
df = pd.DataFrame.from_dict(data)


# In[36]:


df


# In[37]:


#Find the fits using polyfit
z = np.polyfit(df.CocoaPercentage, df.Rating, 1)

#Make the line polynomial function using poly1d
line_function = np.poly1d(z)

#Plotting the data
plt.scatter(df.CocoaPercentage, df.Rating)
plt.title('Cocoa Percentage & Ratings Correlation')
plt.xlabel('Cocoa Percentage (%)')
plt.ylabel('Ratings')
plt.plot(df.CocoaPercentage, line_function(df.CocoaPercentage), "r--")
plt.show()


# In[38]:


import requests 
URL = "https://www.geeksforgeeks.org/data-structures/"
r = requests.get(URL) 
print(r.content) 


# In[39]:


#This will not run on online IDE 
import requests 
from bs4 import BeautifulSoup 
  
URL = "http://www.values.com/inspirational-quotes"
r = requests.get(URL) 
  
soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib 
print(soup.prettify()) 


# In[42]:


#Python program to scrape website  
#and save quotes from website 
import requests 
from bs4 import BeautifulSoup 
import csv 
   
URL = "http://www.values.com/inspirational-quotes"
r = requests.get(URL) 
   
soup = BeautifulSoup(r.content, 'html5lib') 
   
quotes=[]  # a list to store quotes 
   
table = soup.find('div', attrs = {'id':'all_quotes'})  
   
for row in table.findAll('div', attrs = {'class':'col-6 col-lg-3 text-center margin-30px-bottom sm-margin-30px-top'}): 
    quote = {} 
    quote['theme'] = row.h5.text 
    quote['url'] = row.a['href'] 
    quote['img'] = row.img['src'] 
    quote['lines'] = row.img['alt'].split(" #")[0] 
    quote['author'] = row.img['alt'].split(" #")[1] 
    quotes.append(quote) 
   
filename = 'inspirational_quotes.csv'
with open(filename, 'w', newline='') as f: 
    w = csv.DictWriter(f,['theme','url','img','lines','author']) 
    w.writeheader() 
    for quote in quotes: 
        w.writerow(quote)


# In[43]:


quotes


# In[44]:


import requests
from bs4 import BeautifulSoup

URL = "https://realpython.github.io/fake-jobs/"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")


# In[45]:


results = soup.find(id="ResultsContainer")
print(results.prettify())


# In[46]:


job_elements = results.find_all("div", class_="card-content")
for job_element in job_elements:
    print(job_element, end="\n"*2)


# In[47]:


for job_element in job_elements:
    title_element = job_element.find("h2", class_="title")
    company_element = job_element.find("h3", class_="company")
    location_element = job_element.find("p", class_="location")
    print(title_element)
    print(company_element)
    print(location_element)
    print()


# In[48]:


for job_element in job_elements:
    title_element = job_element.find("h2", class_="title")
    company_element = job_element.find("h3", class_="company")
    location_element = job_element.find("p", class_="location")
    print(title_element.text)
    print(company_element.text)
    print(location_element.text)
    print()


# In[49]:


for job_element in job_elements:
    title_element = job_element.find("h2", class_="title")
    company_element = job_element.find("h3", class_="company")
    location_element = job_element.find("p", class_="location")
    print(title_element.text.strip())
    print(company_element.text.strip())
    print(location_element.text.strip())
    print()


# In[50]:


python_jobs = results.find_all("h2", string="Python")
print(python_jobs)


# In[51]:


python_jobs = results.find_all("h2", string=lambda text: "python" in text.lower())
print(len(python_jobs))


# In[52]:


python_jobs = results.find_all("h2", string=lambda text: "python" in text.lower())

python_job_elements = [h2_element.parent.parent.parent for h2_element in python_jobs]
python_job_elements


# In[53]:


for job_element in python_job_elements:
    # -- snip --
    links = job_element.find_all("a")
    for link in links:
        print(link.text.strip())


# In[54]:


for job_element in python_job_elements:
    # -- snip --
    links = job_element.find_all("a")
    for link in links:
        link_url = link["href"]
        print(f"Apply here: {link_url}\n")


# In[ ]:





# In[ ]:




