#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
One of the most important tasks for someone working on datasets with countries, cities, etc. is to understand the 
relationships between their data’s physical location and their geographical context.  And one such way to visualize the 
data is using Folium.

Folium is a powerful data visualization library in Python that was built primarily to help people visualize 
geospatial data. With Folium, one can create a map of any location in the world. Folium is actually a python 
wrapper for leaflet.js which is a javascript library for plotting interactive maps.

We shall now see a simple way to plot and visualize geospatial data. We will use a dataset consisting of 
unemployment rates in the US'''
# import the folium, pandas libraries
import folium
import pandas as pd
 
# initialize the map and store it in a m object
m = folium.Map(location = [40, -95],
               zoom_start = 4)
 n
# show the map
m.save('my_map.html')


# In[20]:


# getting the data
url = (
    "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
)
state_geo = f"{url}/us-states.json"
state_unemployment = f"{url}/US_Unemployment_Oct2012.csv"
state_data = pd.read_csv(state_unemployment)


# In[3]:


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


# In[4]:


import folium
from folium.plugins import MarkerCluster
import pandas as pd


# In[5]:


#Define coordinates of where we want to center our map
boulder_coords = [40.015, -105.2705]

#Create the map
my_map = folium.Map(location = boulder_coords, zoom_start = 13)

#Display the map
my_map


# In[6]:


#Define the coordinates we want our markers to be at
CU_Boulder_coords = [40.007153, -105.266930]
East_Campus_coords = [40.013501, -105.251889]
SEEC_coords = [40.009837, -105.241905]

#Add markers to the map
folium.Marker(CU_Boulder_coords, popup = 'CU Boulder').add_to(my_map)
folium.Marker(East_Campus_coords, popup = 'East Campus').add_to(my_map)
folium.Marker(SEEC_coords, popup = 'SEEC Building').add_to(my_map)

#Display the map
my_map


# In[7]:


#Adding a tileset to our map
map_with_tiles = folium.Map(location = boulder_coords, tiles = 'Stamen Toner')
map_with_tiles


# In[8]:


#Using polygon markers with colors instead of default markers
polygon_map = folium.Map(location = boulder_coords, zoom_start = 13)

CU_Boulder_coords = [40.007153, -105.266930]
East_Campus_coords = [40.013501, -105.251889]
SEEC_coords = [40.009837, -105.241905]

#Add markers to the map
folium.RegularPolygonMarker(CU_Boulder_coords, popup = 'CU Boulder', fill_color = '#00ff40',
                            number_of_sides = 3, radius = 10).add_to(polygon_map)
folium.RegularPolygonMarker(East_Campus_coords, popup = 'EastCampus', fill_color = '#bf00ff',
                            number_of_sides = 5, radius = 10).add_to(polygon_map)
folium.RegularPolygonMarker(SEEC_coords, popup = 'SEEC Building', fill_color = '#ff0000',
                           number_of_sides = 8, radius = 10).add_to(polygon_map)

#Display the map
polygon_map


# In[9]:


kol = folium.Map(location=[22.57, 88.36], tiles='openstreetmap', zoom_start=12)
kol


# In[10]:


#add marker for a place

#victoria memorial
tooltip_1 = "This is Victoria Memorial"
tooltip_2 ="This is Eden Gardens"

folium.Marker(
    [22.54472, 88.34273], popup="Victoria Memorial", tooltip=tooltip_1).add_to(kol)

folium.Marker(
    [22.56487826917627, 88.34336378854425], popup="Eden Gardens", tooltip=tooltip_2).add_to(kol)

kol


# In[11]:


folium.Marker(
    location=[22.55790780507432, 88.35087264462007],
    popup="Indian Museum",
    icon=folium.Icon(color="red", icon="info-sign"),
).add_to(kol)

kol


# In[12]:


kol2 = folium.Map(location=[22.55790780507432, 88.35087264462007], tiles="Stamen Toner", zoom_start=13)
kol2


# In[13]:


#adding circle

folium.Circle(
    location=[22.585728381244373, 88.41462932675563],
    radius=1500,
    popup="Salt Lake",
    color="blue",
    fill=True,
).add_to(kol2)

folium.Circle(
    location=[22.56602918189088, 88.36508424354102],
    radius=2000,
    popup="Old Kolkata",
    color="red",
    fill=True,
).add_to(kol2)


kol2


# In[14]:


# Create a map
pak = folium.Map(location=[30.375321,69.345116], tiles='openstreetmap', zoom_start=5)
pak


# In[15]:


#adding 3 locations, Islmabad, Rawalpindi and Chakwal
loc= [(33.738045, 73.084488),(32.082466, 72.669128),(30.18414, 67.00141) ,
      (24.860966, 66.990501)]


folium.PolyLine(locations = loc,
                line_opacity = 0.5).add_to(pak)
pak


# In[16]:


df_state=pd.read_csv("state wise centroids_2011.csv")


# In[17]:


df_state.head()


# In[18]:


#creating a new map for India, for all states population centres to be plotted
# Create a map
india2 = folium.Map(location=[20.180862078886562, 78.77642751195584], tiles='openstreetmap', zoom_start=4.5)


# In[19]:


#adding the markers

for i in range (0,35):
    state=df_state["State"][i]
    lat=df_state["Latitude"][i]
    long=df_state["Longitude"][i]
    folium.Marker(
    [lat, long], popup=state, tooltip=state).add_to(india2)
 
india2


# In[ ]:





# In[ ]:




