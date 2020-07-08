
# Segmenting and Clustering Neighborhoods in Toronto
In the following program there is a geographic analysis of Toronto city. It is divided in three parts:
1. Web scraping: obtain the database of the neighborhoods needed for the analysis
2. Coordinates: obtain the coordinates of each neighborhood and venue. Create a dataframe with the characteristics of each neighborhood according to the frequency of the different categories of venues. 
3. K-means to cluster neighborhoods: use the k-means algorithm to cluster the neighborhoods into 5 groups as of their characterization in the previous step. Display the results.

## 1. Web Scraping


```python
import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library
```

From the wikipedia page extract all the dataframes:


```python
url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
df = pd.read_html(url, header=0)
len(df)
```




    3



Select the dataframe of interest and display the first 5 values:


```python
print(len(df[0]["Neighborhood"].unique()))
df_t1 = df[0]
#type(df_t1)
print(df_t1.shape)
df_t1.head()
```

    100
    (180, 3)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
  </tbody>
</table>
</div>



Delete the rows with "Not assigned" for the boroughs. Combine the neighborhood names according to its postal code. One thing to highlight is that there are less unique neighborhood values (i.e. less neighborhood names) than unique postal code values. This means that: i) one neighborhood has two postal codes; or ii) there are different neighborhoods with the same name. _**For this reason, great part of the data analysis is done with the postal code identifier instead of the neighborhood name**_:


```python
df_t2 = df_t1[df_t1["Borough"] != "Not assigned"]
print("The number of unique postal codes:",len(df_t2["Postal Code"].unique())) 
print("The number of unique neighborhoods:",len(df_t2["Neighborhood"].unique()))
df_t2.sort_values("Postal Code", ascending=True)
df_t2.reset_index(inplace=True, drop=True)
df_t2.head()
```

    The number of unique postal codes: 103
    The number of unique neighborhoods: 99
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
    </tr>
  </tbody>
</table>
</div>



Here I verify if there is any "Not assigned" value left in Borough or Neighborhood:


```python
if "Not assigned" in df_t2["Borough"].unique():
    print("clean")
else: print("ok")

if "Not assigned" in df_t2["Neighborhood"].unique():
    print("clean")
else: print("ok")
```

    ok
    ok
    


```python
print("The number of rows of the dataset is:",df_t2.shape[0],"and of columns:",df_t2.shape[1])
```

    The number of rows of the dataset is: 103 and of columns: 3
    

## 2. Coordinates
In this section I add the coordinates information to each neighborhood, according to its postal code.

In this first try is the code necessary to obtain the latitude and longitude from the `geocoder` library.


```python
#! pip install geocoder
#import geocoder
#column_names = ["Postal Code","Borough","Neighborhood","Latitude","Longitude"]
#nh = pd.DataFrame(columns=column_names)
#lat_lng_coords = None
#i = 0
#h= 0
#for post, bor, nei in zip(df_t2["Postal Code"],df_t2["Borough"],df_t2["Neighborhood"]):
#    lat_lng_coords = None
#    h=0
#    while(lat_lng_coords is None):
#        g = geocoder.google('{}, Toronto, Ontario'.format(post))
#        lat_lng_coords = g.latlng
#        h+=1
#        print(h)
#        if h==50: 
#            lat_lng_coords = ["None","None"]
            
#    latitude = lat_lng_coords[0]
#     longitude = lat_lng_coords[1]
#    nh = nh.append({'Postal Code': post,
#                   'Borough': bor,
#                   'Neighborhood': nei,
#                   'Latitude':latitude,
#                   'Longitude':longitude}, ignore_index = True)
#    print(i)
#    i+=1
```

After many iterations, this api wasn't able to produce the information needed, so I used an external database:


```python
df_lat_lng = pd.read_csv("Geospatial_Coordinates.csv")
print(df_lat_lng.shape)
df_lat_lng.head()
```

    (103, 3)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



Here I merge dataframes by "Postal Code":


```python
df_t3 = pd.merge(left=df_t2, right=df_lat_lng, how="left", left_on="Postal Code", right_on = "Postal Code")
print(df_t3.shape)
#print(df_t3["Neighborhood"].unique())
df_t3.head()
```

    (103, 5)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
      <td>43.753259</td>
      <td>-79.329656</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
      <td>43.725882</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
      <td>43.718518</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>43.662301</td>
      <td>-79.389494</td>
    </tr>
  </tbody>
</table>
</div>



Here is use the `geocode` library to get the coordinates of Toronto City:


```python
address = "Toronto, ON"

geolocator = Nominatim(user_agent="t_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude

print("The geographical coordinate of Toronto asre {}, {}.".format(latitude,longitude))
```

    The geographical coordinate of Toronto asre 43.6534817, -79.3839347.
    

The folowing script yields the map of Toronto with the location of each neighborhood (signaled by the markers):


```python
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

for lat, lng, borough, neighborhood in zip(df_t3['Latitude'], df_t3['Longitude'], df_t3['Borough'], df_t3['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='midnightblue',
        fill=True,
        fill_color='maroon',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgCiAgICAgICAgPHNjcmlwdD4KICAgICAgICAgICAgTF9OT19UT1VDSCA9IGZhbHNlOwogICAgICAgICAgICBMX0RJU0FCTEVfM0QgPSBmYWxzZTsKICAgICAgICA8L3NjcmlwdD4KICAgIAogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjYuMC9kaXN0L2xlYWZsZXQuanMiPjwvc2NyaXB0PgogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY29kZS5qcXVlcnkuY29tL2pxdWVyeS0xLjEyLjQubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9qcy9ib290c3RyYXAubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5qcyI+PC9zY3JpcHQ+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjYuMC9kaXN0L2xlYWZsZXQuY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vYm9vdHN0cmFwLzMuMi4wL2Nzcy9ib290c3RyYXAubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLXRoZW1lLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9mb250LWF3ZXNvbWUvNC42LjMvY3NzL2ZvbnQtYXdlc29tZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuanMuY2xvdWRmbGFyZS5jb20vYWpheC9saWJzL0xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLzIuMC4yL2xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL3Jhd2Nkbi5naXRoYWNrLmNvbS9weXRob24tdmlzdWFsaXphdGlvbi9mb2xpdW0vbWFzdGVyL2ZvbGl1bS90ZW1wbGF0ZXMvbGVhZmxldC5hd2Vzb21lLnJvdGF0ZS5jc3MiLz4KICAgIDxzdHlsZT5odG1sLCBib2R5IHt3aWR0aDogMTAwJTtoZWlnaHQ6IDEwMCU7bWFyZ2luOiAwO3BhZGRpbmc6IDA7fTwvc3R5bGU+CiAgICA8c3R5bGU+I21hcCB7cG9zaXRpb246YWJzb2x1dGU7dG9wOjA7Ym90dG9tOjA7cmlnaHQ6MDtsZWZ0OjA7fTwvc3R5bGU+CiAgICAKICAgICAgICAgICAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwKICAgICAgICAgICAgICAgIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxlPTEuMCwgdXNlci1zY2FsYWJsZT1ubyIgLz4KICAgICAgICAgICAgPHN0eWxlPgogICAgICAgICAgICAgICAgI21hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSB7CiAgICAgICAgICAgICAgICAgICAgcG9zaXRpb246IHJlbGF0aXZlOwogICAgICAgICAgICAgICAgICAgIHdpZHRoOiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgbGVmdDogMC4wJTsKICAgICAgICAgICAgICAgICAgICB0b3A6IDAuMCU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIDwvc3R5bGU+CiAgICAgICAgCjwvaGVhZD4KPGJvZHk+ICAgIAogICAgCiAgICAgICAgICAgIDxkaXYgY2xhc3M9ImZvbGl1bS1tYXAiIGlkPSJtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UiID48L2Rpdj4KICAgICAgICAKPC9ib2R5Pgo8c2NyaXB0PiAgICAKICAgIAogICAgICAgICAgICB2YXIgbWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlID0gTC5tYXAoCiAgICAgICAgICAgICAgICAibWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlIiwKICAgICAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgICAgICBjZW50ZXI6IFs0My42NTM0ODE3LCAtNzkuMzgzOTM0N10sCiAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NywKICAgICAgICAgICAgICAgICAgICB6b29tOiAxMCwKICAgICAgICAgICAgICAgICAgICB6b29tQ29udHJvbDogdHJ1ZSwKICAgICAgICAgICAgICAgICAgICBwcmVmZXJDYW52YXM6IGZhbHNlLAogICAgICAgICAgICAgICAgfQogICAgICAgICAgICApOwoKICAgICAgICAgICAgCgogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzhjMTgyNDZhMDJlNjQ1NDQ4YzZmNmZhM2UwYjAxOTYzID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAiaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmciLAogICAgICAgICAgICAgICAgeyJhdHRyaWJ1dGlvbiI6ICJEYXRhIGJ5IFx1MDAyNmNvcHk7IFx1MDAzY2EgaHJlZj1cImh0dHA6Ly9vcGVuc3RyZWV0bWFwLm9yZ1wiXHUwMDNlT3BlblN0cmVldE1hcFx1MDAzYy9hXHUwMDNlLCB1bmRlciBcdTAwM2NhIGhyZWY9XCJodHRwOi8vd3d3Lm9wZW5zdHJlZXRtYXAub3JnL2NvcHlyaWdodFwiXHUwMDNlT0RiTFx1MDAzYy9hXHUwMDNlLiIsICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwgIm1heE5hdGl2ZVpvb20iOiAxOCwgIm1heFpvb20iOiAxOCwgIm1pblpvb20iOiAwLCAibm9XcmFwIjogZmFsc2UsICJvcGFjaXR5IjogMSwgInN1YmRvbWFpbnMiOiAiYWJjIiwgInRtcyI6IGZhbHNlfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDVjOWUwOTgzODRjNDU3MWFlYTA2YzNjNDVjYjU3NmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTMyNTg2LCAtNzkuMzI5NjU2NV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTdiOWM1ZjQ3MWU3NGI3NDlkODE5ZTIwODU0Y2NjMDQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2FhZjNjZWZmNmVkNDQ2ZWViZGVmMThiZmFhZjMwYTU5ID0gJChgPGRpdiBpZD0iaHRtbF9hYWYzY2VmZjZlZDQ0NmVlYmRlZjE4YmZhYWYzMGE1OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya3dvb2RzLCBOb3J0aCBZb3JrPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2U3YjljNWY0NzFlNzRiNzQ5ZDgxOWUyMDg1NGNjYzA0LnNldENvbnRlbnQoaHRtbF9hYWYzY2VmZjZlZDQ0NmVlYmRlZjE4YmZhYWYzMGE1OSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfZDVjOWUwOTgzODRjNDU3MWFlYTA2YzNjNDVjYjU3NmMuYmluZFBvcHVwKHBvcHVwX2U3YjljNWY0NzFlNzRiNzQ5ZDgxOWUyMDg1NGNjYzA0KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zODVlYzJiMjljMjM0YTVjYWFjODhlZTY2YmNhMGZhZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNTg4MjI5OTk5OTk5NSwgLTc5LjMxNTU3MTU5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9jMTM1MTA5MGQxODc0MTk0YTkyNzZlNmI3NWI3ZmNjZCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNDQxYmM3YjYzOTI5NGEwNmFmMTVkNzk4N2ZjOTMxYjcgPSAkKGA8ZGl2IGlkPSJodG1sXzQ0MWJjN2I2MzkyOTRhMDZhZjE1ZDc5ODdmYzkzMWI3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5WaWN0b3JpYSBWaWxsYWdlLCBOb3J0aCBZb3JrPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2MxMzUxMDkwZDE4NzQxOTRhOTI3NmU2Yjc1YjdmY2NkLnNldENvbnRlbnQoaHRtbF80NDFiYzdiNjM5Mjk0YTA2YWYxNWQ3OTg3ZmM5MzFiNyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMzg1ZWMyYjI5YzIzNGE1Y2FhYzg4ZWU2NmJjYTBmYWUuYmluZFBvcHVwKHBvcHVwX2MxMzUxMDkwZDE4NzQxOTRhOTI3NmU2Yjc1YjdmY2NkKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82NTBhNTg5NzMzMTI0MTZkODFiMDc5Mjc3OWIxN2YxMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NDI1OTksIC03OS4zNjA2MzU5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8wMTU2OTEyODk3MDk0NWNhOGRlOGMwZTU4NGZlZjdmNCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNjk1N2ZjMjE0OGNlNDllMmE0ODJiNjY3Y2NkYmMxZGYgPSAkKGA8ZGl2IGlkPSJodG1sXzY5NTdmYzIxNDhjZTQ5ZTJhNDgyYjY2N2NjZGJjMWRmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SZWdlbnQgUGFyaywgSGFyYm91cmZyb250LCBEb3dudG93biBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzAxNTY5MTI4OTcwOTQ1Y2E4ZGU4YzBlNTg0ZmVmN2Y0LnNldENvbnRlbnQoaHRtbF82OTU3ZmMyMTQ4Y2U0OWUyYTQ4MmI2NjdjY2RiYzFkZik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNjUwYTU4OTczMzEyNDE2ZDgxYjA3OTI3NzliMTdmMTEuYmluZFBvcHVwKHBvcHVwXzAxNTY5MTI4OTcwOTQ1Y2E4ZGU4YzBlNTg0ZmVmN2Y0KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83M2QwMTU3Mjg0YjY0OGY4OTY3NzIxYzBjYTNkOTI3NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxODUxNzk5OTk5OTk5NiwgLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8wZjE4NzYyOTNlMjM0MTdlYjQ5ODUyMWFmNDcyODAxZiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNmMxMmJlNmYwMjU5NDQ0NGJiNzY0MTA5MWQzNjhhYWEgPSAkKGA8ZGl2IGlkPSJodG1sXzZjMTJiZTZmMDI1OTQ0NDRiYjc2NDEwOTFkMzY4YWFhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MYXdyZW5jZSBNYW5vciwgTGF3cmVuY2UgSGVpZ2h0cywgTm9ydGggWW9yazwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8wZjE4NzYyOTNlMjM0MTdlYjQ5ODUyMWFmNDcyODAxZi5zZXRDb250ZW50KGh0bWxfNmMxMmJlNmYwMjU5NDQ0NGJiNzY0MTA5MWQzNjhhYWEpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzczZDAxNTcyODRiNjQ4Zjg5Njc3MjFjMGNhM2Q5Mjc0LmJpbmRQb3B1cChwb3B1cF8wZjE4NzYyOTNlMjM0MTdlYjQ5ODUyMWFmNDcyODAxZikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjY4YjI4NGE3NTNhNDBhMzlkYzI0MDY5YmEyNTg0MjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjIzMDE1LCAtNzkuMzg5NDkzOF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTljOWI5NDljNWI2NDU5OWFmYjVkNDk1MGQyNDM2YmQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzFiZTZiNTZhNTBhZDQwZDg4NmJlMjEzZWNmYTM1YmQ1ID0gJChgPGRpdiBpZD0iaHRtbF8xYmU2YjU2YTUwYWQ0MGQ4ODZiZTIxM2VjZmEzNWJkNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UXVlZW4mIzM5O3MgUGFyaywgT250YXJpbyBQcm92aW5jaWFsIEdvdmVybm1lbnQsIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZTljOWI5NDljNWI2NDU5OWFmYjVkNDk1MGQyNDM2YmQuc2V0Q29udGVudChodG1sXzFiZTZiNTZhNTBhZDQwZDg4NmJlMjEzZWNmYTM1YmQ1KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9mNjhiMjg0YTc1M2E0MGEzOWRjMjQwNjliYTI1ODQyNy5iaW5kUG9wdXAocG9wdXBfZTljOWI5NDljNWI2NDU5OWFmYjVkNDk1MGQyNDM2YmQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkyYWU5ODQ1NzZiYzQ3YzdhMTgyM2NlNjNkNjMxMzhlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY3ODU1NiwgLTc5LjUzMjI0MjQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hYTRhNjhlNGNjMTg0MGI0OWViMzliMGRkMDJmNTUwMyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMWU3MjI2M2I1NDFmNGRiNGJmMWE3NDZiNWIzNGZhZmMgPSAkKGA8ZGl2IGlkPSJodG1sXzFlNzIyNjNiNTQxZjRkYjRiZjFhNzQ2YjViMzRmYWZjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Jc2xpbmd0b24gQXZlbnVlLCBIdW1iZXIgVmFsbGV5IFZpbGxhZ2UsIEV0b2JpY29rZTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9hYTRhNjhlNGNjMTg0MGI0OWViMzliMGRkMDJmNTUwMy5zZXRDb250ZW50KGh0bWxfMWU3MjI2M2I1NDFmNGRiNGJmMWE3NDZiNWIzNGZhZmMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzkyYWU5ODQ1NzZiYzQ3YzdhMTgyM2NlNjNkNjMxMzhlLmJpbmRQb3B1cChwb3B1cF9hYTRhNjhlNGNjMTg0MGI0OWViMzliMGRkMDJmNTUwMykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2Q4ZDQ3OTQzY2E4NGI1MTk1Y2MyZTEzZTBjMWFhN2IgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My44MDY2ODYyOTk5OTk5OTYsIC03OS4xOTQzNTM0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMTk0MTBjMmE1YzAyNGY4OGJiMWEzNjE5MzZhYjcwZjAgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzIyOWI4M2YwYWI0NzRjYzQ5ZWRkZDNlYjhhZjgyODEyID0gJChgPGRpdiBpZD0iaHRtbF8yMjliODNmMGFiNDc0Y2M0OWVkZGQzZWI4YWY4MjgxMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWFsdmVybiwgUm91Z2UsIFNjYXJib3JvdWdoPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzE5NDEwYzJhNWMwMjRmODhiYjFhMzYxOTM2YWI3MGYwLnNldENvbnRlbnQoaHRtbF8yMjliODNmMGFiNDc0Y2M0OWVkZGQzZWI4YWY4MjgxMik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfY2Q4ZDQ3OTQzY2E4NGI1MTk1Y2MyZTEzZTBjMWFhN2IuYmluZFBvcHVwKHBvcHVwXzE5NDEwYzJhNWMwMjRmODhiYjFhMzYxOTM2YWI3MGYwKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wZDEyZWE5NDM0MzM0YjhlYmNiM2UwZTcyMTdlMDZmNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc0NTkwNTc5OTk5OTk5NiwgLTc5LjM1MjE4OF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfN2NkZTNhNmY3NzJhNDU4M2E0NjUzZWVhOTg2MDg4OGIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2YxOWIzMGFmMzg5MjQwNzI4OTBkYzgxYmNlZTg4YzZlID0gJChgPGRpdiBpZD0iaHRtbF9mMTliMzBhZjM4OTI0MDcyODkwZGM4MWJjZWU4OGM2ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9uIE1pbGxzLCBOb3J0aCBZb3JrPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzdjZGUzYTZmNzcyYTQ1ODNhNDY1M2VlYTk4NjA4ODhiLnNldENvbnRlbnQoaHRtbF9mMTliMzBhZjM4OTI0MDcyODkwZGM4MWJjZWU4OGM2ZSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMGQxMmVhOTQzNDMzNGI4ZWJjYjNlMGU3MjE3ZTA2ZjUuYmluZFBvcHVwKHBvcHVwXzdjZGUzYTZmNzcyYTQ1ODNhNDY1M2VlYTk4NjA4ODhiKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mOGFjZDhkNDczMzk0MzY4YjM4NTJmNjM4YzQ5YzVlMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNjM5NzIsIC03OS4zMDk5MzddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2M1ZWQxNTJlYzIxMjRlZGI4NTZiOThkYzY2ODEyNGI5ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hZWNmNzgyOTA3YzQ0YWEyYWRkYjI5YTU4M2Q3MzI4MCA9ICQoYDxkaXYgaWQ9Imh0bWxfYWVjZjc4MjkwN2M0NGFhMmFkZGIyOWE1ODNkNzMyODAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmt2aWV3IEhpbGwsIFdvb2RiaW5lIEdhcmRlbnMsIEVhc3QgWW9yazwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9jNWVkMTUyZWMyMTI0ZWRiODU2Yjk4ZGM2NjgxMjRiOS5zZXRDb250ZW50KGh0bWxfYWVjZjc4MjkwN2M0NGFhMmFkZGIyOWE1ODNkNzMyODApOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2Y4YWNkOGQ0NzMzOTQzNjhiMzg1MmY2MzhjNDljNWUxLmJpbmRQb3B1cChwb3B1cF9jNWVkMTUyZWMyMTI0ZWRiODU2Yjk4ZGM2NjgxMjRiOSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmM3ZjAyMmFjN2IxNDMzY2FjM2I5ODUzOTg0OThhYzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTcxNjE4LCAtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzQ4OTEzNDRkNWNmZDQwMDRiNmMyNGM0NWQzZTY5MjViID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8wM2NiNTkwMWQwMzc0MDQxYjJjYzEyMWZjNThmNzRlYiA9ICQoYDxkaXYgaWQ9Imh0bWxfMDNjYjU5MDFkMDM3NDA0MWIyY2MxMjFmYzU4Zjc0ZWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdhcmRlbiBEaXN0cmljdCwgUnllcnNvbiwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF80ODkxMzQ0ZDVjZmQ0MDA0YjZjMjRjNDVkM2U2OTI1Yi5zZXRDb250ZW50KGh0bWxfMDNjYjU5MDFkMDM3NDA0MWIyY2MxMjFmYzU4Zjc0ZWIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzJjN2YwMjJhYzdiMTQzM2NhYzNiOTg1Mzk4NDk4YWMzLmJpbmRQb3B1cChwb3B1cF80ODkxMzQ0ZDVjZmQ0MDA0YjZjMjRjNDVkM2U2OTI1YikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWEwZjM1NTAwZWI5NDI2MWIyMjE5Njc5MDNkNWE5MjUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDk1NzcsIC03OS40NDUwNzI1OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfM2YwZDRhZDIxYjU2NDExOGJjY2Q4NmFlYWM2MzhiMDYgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzQ4M2RlODQxMzU0MjRjYmU4ZDljN2VhNmRhYWE4YWI0ID0gJChgPGRpdiBpZD0iaHRtbF80ODNkZTg0MTM1NDI0Y2JlOGQ5YzdlYTZkYWFhOGFiNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R2xlbmNhaXJuLCBOb3J0aCBZb3JrPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzNmMGQ0YWQyMWI1NjQxMThiY2NkODZhZWFjNjM4YjA2LnNldENvbnRlbnQoaHRtbF80ODNkZTg0MTM1NDI0Y2JlOGQ5YzdlYTZkYWFhOGFiNCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMWEwZjM1NTAwZWI5NDI2MWIyMjE5Njc5MDNkNWE5MjUuYmluZFBvcHVwKHBvcHVwXzNmMGQ0YWQyMWI1NjQxMThiY2NkODZhZWFjNjM4YjA2KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZTZjZDdjMWRlYjk0ZTM0YTA1N2M5MDBjZGFiOGQ3ZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDk0MzIsIC03OS41NTQ3MjQ0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMDI5ZTdjNjVjNjVjNDEzNjlhMDNjOGJjYjM5OWQzNzEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2Q5NWZmZWE2OTExMDQwNGFhY2IwMmM1MTA1NDQ0MDExID0gJChgPGRpdiBpZD0iaHRtbF9kOTVmZmVhNjkxMTA0MDRhYWNiMDJjNTEwNTQ0NDAxMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBEZWFuZSBQYXJrLCBQcmluY2VzcyBHYXJkZW5zLCBNYXJ0aW4gR3JvdmUsIElzbGluZ3RvbiwgQ2xvdmVyZGFsZSwgRXRvYmljb2tlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzAyOWU3YzY1YzY1YzQxMzY5YTAzYzhiY2IzOTlkMzcxLnNldENvbnRlbnQoaHRtbF9kOTVmZmVhNjkxMTA0MDRhYWNiMDJjNTEwNTQ0NDAxMSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYWU2Y2Q3YzFkZWI5NGUzNGEwNTdjOTAwY2RhYjhkN2YuYmluZFBvcHVwKHBvcHVwXzAyOWU3YzY1YzY1YzQxMzY5YTAzYzhiY2IzOTlkMzcxKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iZWU4MTEyNzNmYTA0ZjY2YTQxMTBhYzQ5NTBjOTc2YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4NDUzNTEsIC03OS4xNjA0OTcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzhmZmQ3OWNkN2E3NDE0NjgwMTRkZDYyMGZlZWZhYjUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzdmM2JjOGU0NjM1ODRhZjBiOWM4MmIyZjI0ZWMyNzU2ID0gJChgPGRpdiBpZD0iaHRtbF83ZjNiYzhlNDYzNTg0YWYwYjljODJiMmYyNGVjMjc1NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um91Z2UgSGlsbCwgUG9ydCBVbmlvbiwgSGlnaGxhbmQgQ3JlZWssIFNjYXJib3JvdWdoPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2M4ZmZkNzljZDdhNzQxNDY4MDE0ZGQ2MjBmZWVmYWI1LnNldENvbnRlbnQoaHRtbF83ZjNiYzhlNDYzNTg0YWYwYjljODJiMmYyNGVjMjc1Nik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYmVlODExMjczZmEwNGY2NmE0MTEwYWM0OTUwYzk3NmMuYmluZFBvcHVwKHBvcHVwX2M4ZmZkNzljZDdhNzQxNDY4MDE0ZGQ2MjBmZWVmYWI1KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZTM5Y2Q1NDYwMWM0MDgxOGQwMTJhMWE4MTQwYmUzYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNTg5OTcwMDAwMDAxLCAtNzkuMzQwOTIzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9lOGNlOWEzMDNmNjE0ZGUxOTQwOTg4OGUxZDBlOWI3OCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNjAwZTVmZTJlNzVhNDZjNzhmMzc1OGQzYzExMjEwZjIgPSAkKGA8ZGl2IGlkPSJodG1sXzYwMGU1ZmUyZTc1YTQ2Yzc4ZjM3NThkM2MxMTIxMGYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb24gTWlsbHMsIE5vcnRoIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZThjZTlhMzAzZjYxNGRlMTk0MDk4ODhlMWQwZTliNzguc2V0Q29udGVudChodG1sXzYwMGU1ZmUyZTc1YTQ2Yzc4ZjM3NThkM2MxMTIxMGYyKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9jZTM5Y2Q1NDYwMWM0MDgxOGQwMTJhMWE4MTQwYmUzYi5iaW5kUG9wdXAocG9wdXBfZThjZTlhMzAzZjYxNGRlMTk0MDk4ODhlMWQwZTliNzgpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U3ODZmNDhjOGFkYTQ4YTU4MTVmM2Q2MjA1MTFiMDNhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk1MzQzOTAwMDAwMDA1LCAtNzkuMzE4Mzg4N10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNThjZjM5ZWRkNWMwNDI1NmI5NDljOTNjNmU2MGE3N2QgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzk0OTk2NWQyMDQ2OTRkY2FiYWE3ZDczYjk3ODQ4NmVhID0gJChgPGRpdiBpZD0iaHRtbF85NDk5NjVkMjA0Njk0ZGNhYmFhN2Q3M2I5Nzg0ODZlYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V29vZGJpbmUgSGVpZ2h0cywgRWFzdCBZb3JrPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzU4Y2YzOWVkZDVjMDQyNTZiOTQ5YzkzYzZlNjBhNzdkLnNldENvbnRlbnQoaHRtbF85NDk5NjVkMjA0Njk0ZGNhYmFhN2Q3M2I5Nzg0ODZlYSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfZTc4NmY0OGM4YWRhNDhhNTgxNWYzZDYyMDUxMWIwM2EuYmluZFBvcHVwKHBvcHVwXzU4Y2YzOWVkZDVjMDQyNTZiOTQ5YzkzYzZlNjBhNzdkKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMmM3MDBmYTRjYWI0NThkYWFlNWViYzRiY2NmN2VlZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTQ5MzksIC03OS4zNzU0MTc5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iZjkwMmVhZjc2ZTE0NTIyYWQwMjFjYTBjOTY4NDM5OCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfOWZlOTQwMWMwYjMwNGM1NjkzYzE2MGRjNmNlZTMxZjcgPSAkKGA8ZGl2IGlkPSJodG1sXzlmZTk0MDFjMGIzMDRjNTY5M2MxNjBkYzZjZWUzMWY3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdC4gSmFtZXMgVG93biwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9iZjkwMmVhZjc2ZTE0NTIyYWQwMjFjYTBjOTY4NDM5OC5zZXRDb250ZW50KGh0bWxfOWZlOTQwMWMwYjMwNGM1NjkzYzE2MGRjNmNlZTMxZjcpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2MyYzcwMGZhNGNhYjQ1OGRhYWU1ZWJjNGJjY2Y3ZWVmLmJpbmRQb3B1cChwb3B1cF9iZjkwMmVhZjc2ZTE0NTIyYWQwMjFjYTBjOTY4NDM5OCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTY5YmZmM2NkZWNhNDYwYmJhYjZmMGY0NWYzMmIzM2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTM3ODEzLCAtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzlhMThjYmFhZDJlMTRjNjNiMzE3ZjAyYjM2ZTI4YWRjID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hZDJmNDgyZjA3NTU0Njk5ODYzMWEwZDhlNzliMzhhMiA9ICQoYDxkaXYgaWQ9Imh0bWxfYWQyZjQ4MmYwNzU1NDY5OTg2MzFhMGQ4ZTc5YjM4YTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWV3b29kLUNlZGFydmFsZSwgWW9yazwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF85YTE4Y2JhYWQyZTE0YzYzYjMxN2YwMmIzNmUyOGFkYy5zZXRDb250ZW50KGh0bWxfYWQyZjQ4MmYwNzU1NDY5OTg2MzFhMGQ4ZTc5YjM4YTIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2E2OWJmZjNjZGVjYTQ2MGJiYWI2ZjBmNDVmMzJiMzNhLmJpbmRQb3B1cChwb3B1cF85YTE4Y2JhYWQyZTE0YzYzYjMxN2YwMmIzNmUyOGFkYykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWViNjIxOTA4OTA0NGY2NGJkNzFmMGMxOGRiNWMxMjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDM1MTUyLCAtNzkuNTc3MjAwNzk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzdlOWExZWNkYTUxOTQ5MmU4N2FiYzNiNTEwNzVhZDA3ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9lNjY1YTNmM2ZkN2M0N2JiYTM2MmI2ZjcwYmU3N2Y0YiA9ICQoYDxkaXYgaWQ9Imh0bWxfZTY2NWEzZjNmZDdjNDdiYmEzNjJiNmY3MGJlNzdmNGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkVyaW5nYXRlLCBCbG9vcmRhbGUgR2FyZGVucywgT2xkIEJ1cm5oYW10aG9ycGUsIE1hcmtsYW5kIFdvb2QsIEV0b2JpY29rZTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF83ZTlhMWVjZGE1MTk0OTJlODdhYmMzYjUxMDc1YWQwNy5zZXRDb250ZW50KGh0bWxfZTY2NWEzZjNmZDdjNDdiYmEzNjJiNmY3MGJlNzdmNGIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzllYjYyMTkwODkwNDRmNjRiZDcxZjBjMThkYjVjMTI4LmJpbmRQb3B1cChwb3B1cF83ZTlhMWVjZGE1MTk0OTJlODdhYmMzYjUxMDc1YWQwNykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWExODZkZDI1MjA4NDJmMGEyOGJjMDZlNzM4MTc4MzEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NjM1NzI2LCAtNzkuMTg4NzExNV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfOWU1Mzg4ZmNiYTUzNGU1Zjk2NmU5OGZkZWRhNDIwZmQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2NjMDM4NzQyYzA4MzQzNjZhNTM5ZjFkNmNhNWJjNWIzID0gJChgPGRpdiBpZD0iaHRtbF9jYzAzODc0MmMwODM0MzY2YTUzOWYxZDZjYTViYzViMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R3VpbGR3b29kLCBNb3JuaW5nc2lkZSwgV2VzdCBIaWxsLCBTY2FyYm9yb3VnaDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF85ZTUzODhmY2JhNTM0ZTVmOTY2ZTk4ZmRlZGE0MjBmZC5zZXRDb250ZW50KGh0bWxfY2MwMzg3NDJjMDgzNDM2NmE1MzlmMWQ2Y2E1YmM1YjMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzVhMTg2ZGQyNTIwODQyZjBhMjhiYzA2ZTczODE3ODMxLmJpbmRQb3B1cChwb3B1cF85ZTUzODhmY2JhNTM0ZTVmOTY2ZTk4ZmRlZGE0MjBmZCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGRlMjgyODZmYTE3NDYwZmJiZGRmZGI1YTgzNjQ1NGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzYzNTczOTk5OTk5OSwgLTc5LjI5MzAzMTJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2ZkMGVhOGI2M2IxZDQxOGE4NGRlMTkyMTM4ZWI1NTczID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hNzM4ZmExNjA1NTc0YWI5YThlMmEwZTIxZDU4YTlkMSA9ICQoYDxkaXYgaWQ9Imh0bWxfYTczOGZhMTYwNTU3NGFiOWE4ZTJhMGUyMWQ1OGE5ZDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzLCBFYXN0IFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZmQwZWE4YjYzYjFkNDE4YTg0ZGUxOTIxMzhlYjU1NzMuc2V0Q29udGVudChodG1sX2E3MzhmYTE2MDU1NzRhYjlhOGUyYTBlMjFkNThhOWQxKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8wZGUyODI4NmZhMTc0NjBmYmJkZGZkYjVhODM2NDU0ZC5iaW5kUG9wdXAocG9wdXBfZmQwZWE4YjYzYjFkNDE4YTg0ZGUxOTIxMzhlYjU1NzMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M5YjIwYTc1N2U0YTQ5ZWJiMDkwNjg1YmZmNmQ4ZWMzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ0NzcwNzk5OTk5OTk2LCAtNzkuMzczMzA2NF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzUzZWM2ZjI2MTkyNDIwOWFmNWI0ZTM1ODU4N2IwMjcgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzRiODliNDYzZDViNzRjYzdhYWMzZTUzYmUxOGExZDRkID0gJChgPGRpdiBpZD0iaHRtbF80Yjg5YjQ2M2Q1Yjc0Y2M3YWFjM2U1M2JlMThhMWQ0ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVyY3p5IFBhcmssIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYzUzZWM2ZjI2MTkyNDIwOWFmNWI0ZTM1ODU4N2IwMjcuc2V0Q29udGVudChodG1sXzRiODliNDYzZDViNzRjYzdhYWMzZTUzYmUxOGExZDRkKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9jOWIyMGE3NTdlNGE0OWViYjA5MDY4NWJmZjZkOGVjMy5iaW5kUG9wdXAocG9wdXBfYzUzZWM2ZjI2MTkyNDIwOWFmNWI0ZTM1ODU4N2IwMjcpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc2OGM5MTA1MDcyMDQzODFhNDUzNjgzODg5YThhMjVmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5MDI1NiwgLTc5LjQ1MzUxMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMjA2YzVmZjQyYThiNGEwOGE0YzNiNGY4Nzc1ZWQ5ZjAgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2Y2ZTE1YTAwM2QxODRjYmJhMGQwYTE1YWMzZTVlMGIwID0gJChgPGRpdiBpZD0iaHRtbF9mNmUxNWEwMDNkMTg0Y2JiYTBkMGExNWFjM2U1ZTBiMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2FsZWRvbmlhLUZhaXJiYW5rcywgWW9yazwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8yMDZjNWZmNDJhOGI0YTA4YTRjM2I0Zjg3NzVlZDlmMC5zZXRDb250ZW50KGh0bWxfZjZlMTVhMDAzZDE4NGNiYmEwZDBhMTVhYzNlNWUwYjApOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzc2OGM5MTA1MDcyMDQzODFhNDUzNjgzODg5YThhMjVmLmJpbmRQb3B1cChwb3B1cF8yMDZjNWZmNDJhOGI0YTA4YTRjM2I0Zjg3NzVlZDlmMCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmJlOGY0MDliNWI4NGFhYWJjYTA4N2JhYmU0OTU0MWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzA5OTIxLCAtNzkuMjE2OTE3NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2UzNDk1Y2EwMjhlNTRjNzJiZTczNGVlMGM0OTAxMzlkID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9lZDFkMDZiODM5NmM0YTBkODRiNjk4Yzk5Mzk1MTJiOSA9ICQoYDxkaXYgaWQ9Imh0bWxfZWQxZDA2YjgzOTZjNGEwZDg0YjY5OGM5OTM5NTEyYjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldvYnVybiwgU2NhcmJvcm91Z2g8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZTM0OTVjYTAyOGU1NGM3MmJlNzM0ZWUwYzQ5MDEzOWQuc2V0Q29udGVudChodG1sX2VkMWQwNmI4Mzk2YzRhMGQ4NGI2OThjOTkzOTUxMmI5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8yYmU4ZjQwOWI1Yjg0YWFhYmNhMDg3YmFiZTQ5NTQxYS5iaW5kUG9wdXAocG9wdXBfZTM0OTVjYTAyOGU1NGM3MmJlNzM0ZWUwYzQ5MDEzOWQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzViZGY5Y2Y1YjRkODRjNjY4Y2EwN2E1ZWNkZDZhZDU4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA5MDYwNCwgLTc5LjM2MzQ1MTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzAxMTczMDkzNzc4ODRhODliZTg1YTJhN2RlOTMwZTAzID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF80YTM2OTQ1MDg3YjY0MWQwYmQ1YmNmMGMyMTI0ZDk2MiA9ICQoYDxkaXYgaWQ9Imh0bWxfNGEzNjk0NTA4N2I2NDFkMGJkNWJjZjBjMjEyNGQ5NjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxlYXNpZGUsIEVhc3QgWW9yazwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8wMTE3MzA5Mzc3ODg0YTg5YmU4NWEyYTdkZTkzMGUwMy5zZXRDb250ZW50KGh0bWxfNGEzNjk0NTA4N2I2NDFkMGJkNWJjZjBjMjEyNGQ5NjIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzViZGY5Y2Y1YjRkODRjNjY4Y2EwN2E1ZWNkZDZhZDU4LmJpbmRQb3B1cChwb3B1cF8wMTE3MzA5Mzc3ODg0YTg5YmU4NWEyYTdkZTkzMGUwMykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODJiYzZiNDZlNGY0NGRlMGJiMGM4Mzc4MzJjMzAxODMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTc5NTI0LCAtNzkuMzg3MzgyNl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNTE4MmY3Nzk0YjcyNDE1MWJmNDM4MDU5ODAyOGI0NWIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzA4NWNlNzBhZWNjOTQyZTg5N2ZlNDM3YzI3MTM4MGUyID0gJChgPGRpdiBpZD0iaHRtbF8wODVjZTcwYWVjYzk0MmU4OTdmZTQzN2MyNzEzODBlMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBCYXkgU3RyZWV0LCBEb3dudG93biBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzUxODJmNzc5NGI3MjQxNTFiZjQzODA1OTgwMjhiNDViLnNldENvbnRlbnQoaHRtbF8wODVjZTcwYWVjYzk0MmU4OTdmZTQzN2MyNzEzODBlMik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfODJiYzZiNDZlNGY0NGRlMGJiMGM4Mzc4MzJjMzAxODMuYmluZFBvcHVwKHBvcHVwXzUxODJmNzc5NGI3MjQxNTFiZjQzODA1OTgwMjhiNDViKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mMDc5M2I0M2QxMDg0OTU3OTc4YTU0MGVmZWZiMjg4YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwgLTc5LjQyMjU2MzddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzU3N2Q2NzY0MTJhYjQ2OGNhMzkyMzM0ZTg2NzJiMjRlID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kYTA2Mjg1ZTdmNzA0YjQwODFhOGYwODQ5MDc4MGUyZCA9ICQoYDxkaXYgaWQ9Imh0bWxfZGEwNjI4NWU3ZjcwNGI0MDgxYThmMDg0OTA3ODBlMmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNocmlzdGllLCBEb3dudG93biBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzU3N2Q2NzY0MTJhYjQ2OGNhMzkyMzM0ZTg2NzJiMjRlLnNldENvbnRlbnQoaHRtbF9kYTA2Mjg1ZTdmNzA0YjQwODFhOGYwODQ5MDc4MGUyZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfZjA3OTNiNDNkMTA4NDk1Nzk3OGE1NDBlZmVmYjI4OGIuYmluZFBvcHVwKHBvcHVwXzU3N2Q2NzY0MTJhYjQ2OGNhMzkyMzM0ZTg2NzJiMjRlKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ZDNjYjEyN2U0N2Q0YzE1YTkyMGJjMjNhZmNkZWU3YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc3MzEzNiwgLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8yY2RjNzBmYTBjMDQ0MDU4ODM0NGIzNjhmMmNiZjQ5NiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMjM3NmM3MDI4Yjc4NGVhNjhiOGUwOWNhMTBkNDc4MmUgPSAkKGA8ZGl2IGlkPSJodG1sXzIzNzZjNzAyOGI3ODRlYTY4YjhlMDljYTEwZDQ3ODJlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZWRhcmJyYWUsIFNjYXJib3JvdWdoPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzJjZGM3MGZhMGMwNDQwNTg4MzQ0YjM2OGYyY2JmNDk2LnNldENvbnRlbnQoaHRtbF8yMzc2YzcwMjhiNzg0ZWE2OGI4ZTA5Y2ExMGQ0NzgyZSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNGQzY2IxMjdlNDdkNGMxNWE5MjBiYzIzYWZjZGVlN2EuYmluZFBvcHVwKHBvcHVwXzJjZGM3MGZhMGMwNDQwNTg4MzQ0YjM2OGYyY2JmNDk2KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZDU2NjBlNmIyMWY0ZTBkOTBkZGJjZTIwMjZmNDNkZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjgwMzc2MjIsIC03OS4zNjM0NTE3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9lMmQ4MTdhOGU0YTA0MzEwYWE1MjRkMWY0MTFmYjljYiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYzE3ZTdlMGUwZmJhNDM4NzlkOGI4NjgwZjFkMTU3YWYgPSAkKGA8ZGl2IGlkPSJodG1sX2MxN2U3ZTBlMGZiYTQzODc5ZDhiODY4MGYxZDE1N2FmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IaWxsY3Jlc3QgVmlsbGFnZSwgTm9ydGggWW9yazwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9lMmQ4MTdhOGU0YTA0MzEwYWE1MjRkMWY0MTFmYjljYi5zZXRDb250ZW50KGh0bWxfYzE3ZTdlMGUwZmJhNDM4NzlkOGI4NjgwZjFkMTU3YWYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2NkNTY2MGU2YjIxZjRlMGQ5MGRkYmNlMjAyNmY0M2RkLmJpbmRQb3B1cChwb3B1cF9lMmQ4MTdhOGU0YTA0MzEwYWE1MjRkMWY0MTFmYjljYikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmU0YmQ4NGFiYWI1NDk3Y2JjN2VjZmUxNDYyZmZmZDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTQzMjgzLCAtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMTliYTNjMDFiOWU0NDc2M2IyZGRjMGVkZGIyMjVjZjEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2YwNTdhZWYyZGIxZDQzYjVhZjYxZWE3NWUxZGU5ZDdjID0gJChgPGRpdiBpZD0iaHRtbF9mMDU3YWVmMmRiMWQ0M2I1YWY2MWVhNzVlMWRlOWQ3YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF0aHVyc3QgTWFub3IsIFdpbHNvbiBIZWlnaHRzLCBEb3duc3ZpZXcgTm9ydGgsIE5vcnRoIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMTliYTNjMDFiOWU0NDc2M2IyZGRjMGVkZGIyMjVjZjEuc2V0Q29udGVudChodG1sX2YwNTdhZWYyZGIxZDQzYjVhZjYxZWE3NWUxZGU5ZDdjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9iZTRiZDg0YWJhYjU0OTdjYmM3ZWNmZTE0NjJmZmZkMy5iaW5kUG9wdXAocG9wdXBfMTliYTNjMDFiOWU0NDc2M2IyZGRjMGVkZGIyMjVjZjEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VlMjI1N2Y5MjdmMjRjYTFhY2M0OWFjYjZjYzNlNGFkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA1MzY4OSwgLTc5LjM0OTM3MTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8zZmVmNDBhZGYxYWY0OTllYTQzYzk3ODFhYjBmNGRjYSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMzM3OTY2NmIxOGQ4NDNhZThkM2IyYmEzOTE2ZDA0MTQgPSAkKGA8ZGl2IGlkPSJodG1sXzMzNzk2NjZiMThkODQzYWU4ZDNiMmJhMzkxNmQwNDE0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaG9ybmNsaWZmZSBQYXJrLCBFYXN0IFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfM2ZlZjQwYWRmMWFmNDk5ZWE0M2M5NzgxYWIwZjRkY2Euc2V0Q29udGVudChodG1sXzMzNzk2NjZiMThkODQzYWU4ZDNiMmJhMzkxNmQwNDE0KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9lZTIyNTdmOTI3ZjI0Y2ExYWNjNDlhY2I2Y2MzZTRhZC5iaW5kUG9wdXAocG9wdXBfM2ZlZjQwYWRmMWFmNDk5ZWE0M2M5NzgxYWIwZjRkY2EpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NhNjg5Zjc4OWE1MTRjZDM5ZmI2MmVlMjY1OWNiNjFhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsIC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8zMjY5MTVmN2FjNmI0MzI5YWU0N2QxMDM5MzhkZTYxNCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNzk1ZjFjMTRjNDk4NGMzYWFhMGYwYTg2NWNhOGFiMzkgPSAkKGA8ZGl2IGlkPSJodG1sXzc5NWYxYzE0YzQ5ODRjM2FhYTBmMGE4NjVjYThhYjM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SaWNobW9uZCwgQWRlbGFpZGUsIEtpbmcsIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMzI2OTE1ZjdhYzZiNDMyOWFlNDdkMTAzOTM4ZGU2MTQuc2V0Q29udGVudChodG1sXzc5NWYxYzE0YzQ5ODRjM2FhYTBmMGE4NjVjYThhYjM5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9jYTY4OWY3ODlhNTE0Y2QzOWZiNjJlZTI2NTljYjYxYS5iaW5kUG9wdXAocG9wdXBfMzI2OTE1ZjdhYzZiNDMyOWFlNDdkMTAzOTM4ZGU2MTQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M0MzA4MzA2YWRjZjRkZTU4ZGYzMDllMjBhNjE0ZjkwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5MDA1MTAwMDAwMDEsIC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8xNWU1MzlhZjIzODc0ZmE0OGMyYTdkNzMxMjMzYjM2ZiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMmQ0MThiOTM1Yzk2NDcyY2FiZWIzNTVkOGY5N2E1YzcgPSAkKGA8ZGl2IGlkPSJodG1sXzJkNDE4YjkzNWM5NjQ3MmNhYmViMzU1ZDhmOTdhNWM3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EdWZmZXJpbiwgRG92ZXJjb3VydCBWaWxsYWdlLCBXZXN0IFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMTVlNTM5YWYyMzg3NGZhNDhjMmE3ZDczMTIzM2IzNmYuc2V0Q29udGVudChodG1sXzJkNDE4YjkzNWM5NjQ3MmNhYmViMzU1ZDhmOTdhNWM3KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9jNDMwODMwNmFkY2Y0ZGU1OGRmMzA5ZTIwYTYxNGY5MC5iaW5kUG9wdXAocG9wdXBfMTVlNTM5YWYyMzg3NGZhNDhjMmE3ZDczMTIzM2IzNmYpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI2OTYwZTBiNzA5ZjQyZWY5OTM1OWQwM2E3YjVjYjg1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzQ0NzM0MiwgLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82MWMzYjRiYzRlNGY0NmUyYjc3YzgwYzg4ZjNiMGZiOSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZDNhMGJiMzFkYzc0NDFlZWFjYjI5MGViZjk2NjcxN2MgPSAkKGA8ZGl2IGlkPSJodG1sX2QzYTBiYjMxZGM3NDQxZWVhY2IyOTBlYmY5NjY3MTdjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TY2FyYm9yb3VnaCBWaWxsYWdlLCBTY2FyYm9yb3VnaDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF82MWMzYjRiYzRlNGY0NmUyYjc3YzgwYzg4ZjNiMGZiOS5zZXRDb250ZW50KGh0bWxfZDNhMGJiMzFkYzc0NDFlZWFjYjI5MGViZjk2NjcxN2MpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzI2OTYwZTBiNzA5ZjQyZWY5OTM1OWQwM2E3YjVjYjg1LmJpbmRQb3B1cChwb3B1cF82MWMzYjRiYzRlNGY0NmUyYjc3YzgwYzg4ZjNiMGZiOSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGI1YTZjMTc1YTZhNDY2ZGE1MTlkMDQ5ODlkN2U3ODYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Nzg1MTc1LCAtNzkuMzQ2NTU1N10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfOTE3ODljNzRlN2QxNGUzYTg1MGQxYTYyOGMyMjgyZGEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2Y3YTA2ZjhmNzUxMDRlYjliOTgyNmQ1OTMzZjkzMzgzID0gJChgPGRpdiBpZD0iaHRtbF9mN2EwNmY4Zjc1MTA0ZWI5Yjk4MjZkNTkzM2Y5MzM4MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RmFpcnZpZXcsIEhlbnJ5IEZhcm0sIE9yaW9sZSwgTm9ydGggWW9yazwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF85MTc4OWM3NGU3ZDE0ZTNhODUwZDFhNjI4YzIyODJkYS5zZXRDb250ZW50KGh0bWxfZjdhMDZmOGY3NTEwNGViOWI5ODI2ZDU5MzNmOTMzODMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzBiNWE2YzE3NWE2YTQ2NmRhNTE5ZDA0OTg5ZDdlNzg2LmJpbmRQb3B1cChwb3B1cF85MTc4OWM3NGU3ZDE0ZTNhODUwZDFhNjI4YzIyODJkYSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzk0ZTI1NTY5ODRlNDQ3NDkyMzU1ZDAxOGY3Mjc0OGMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Njc5ODAzLCAtNzkuNDg3MjYxOTAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzFkNjBlNzczNzg5MTQ4ZWRhN2U0NmY1MjE0ZDY1N2I1ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hZDU3MGQ4NjQzZjc0NTdiYjgyODQyMTljZjA2M2I0OSA9ICQoYDxkaXYgaWQ9Imh0bWxfYWQ1NzBkODY0M2Y3NDU3YmI4Mjg0MjE5Y2YwNjNiNDkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRod29vZCBQYXJrLCBZb3JrIFVuaXZlcnNpdHksIE5vcnRoIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMWQ2MGU3NzM3ODkxNDhlZGE3ZTQ2ZjUyMTRkNjU3YjUuc2V0Q29udGVudChodG1sX2FkNTcwZDg2NDNmNzQ1N2JiODI4NDIxOWNmMDYzYjQ5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl83OTRlMjU1Njk4NGU0NDc0OTIzNTVkMDE4ZjcyNzQ4Yy5iaW5kUG9wdXAocG9wdXBfMWQ2MGU3NzM3ODkxNDhlZGE3ZTQ2ZjUyMTRkNjU3YjUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZkODAzMDk1ZmI0MTQ5OTU4Y2E4YTI0NjA1ZTk4NGMxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg1MzQ3LCAtNzkuMzM4MTA2NV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNWM3YjQyY2RhMTgyNDlhYjk5ZGQwODg4ZTZmMTc4ZjQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2IwZWU3MGZmYWU3MTQ3Y2NhZjBkMTU4ZTllNmVhNzlkID0gJChgPGRpdiBpZD0iaHRtbF9iMGVlNzBmZmFlNzE0N2NjYWYwZDE1OGU5ZTZlYTc5ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvLCBCcm9hZHZpZXcgTm9ydGggKE9sZCBFYXN0IFlvcmspLCBFYXN0IFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNWM3YjQyY2RhMTgyNDlhYjk5ZGQwODg4ZTZmMTc4ZjQuc2V0Q29udGVudChodG1sX2IwZWU3MGZmYWU3MTQ3Y2NhZjBkMTU4ZTllNmVhNzlkKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl82ZDgwMzA5NWZiNDE0OTk1OGNhOGEyNDYwNWU5ODRjMS5iaW5kUG9wdXAocG9wdXBfNWM3YjQyY2RhMTgyNDlhYjk5ZGQwODg4ZTZmMTc4ZjQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJiYTc5NzgxN2Q3NDRiYzZhMGRmYjJlYTdmMmJiYTlhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywgLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF85NmYzZDJkZDI2Mzg0ODZlODQ3YzJmOWY1MzRkZjJjOSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfODg4OTMwZWUxODZlNDg4ZDkzZWI2OTQwODA5ZjEwMTMgPSAkKGA8ZGl2IGlkPSJodG1sXzg4ODkzMGVlMTg2ZTQ4OGQ5M2ViNjk0MDgwOWYxMDEzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQgRWFzdCwgVW5pb24gU3RhdGlvbiwgVG9yb250byBJc2xhbmRzLCBEb3dudG93biBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzk2ZjNkMmRkMjYzODQ4NmU4NDdjMmY5ZjUzNGRmMmM5LnNldENvbnRlbnQoaHRtbF84ODg5MzBlZTE4NmU0ODhkOTNlYjY5NDA4MDlmMTAxMyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMmJhNzk3ODE3ZDc0NGJjNmEwZGZiMmVhN2YyYmJhOWEuYmluZFBvcHVwKHBvcHVwXzk2ZjNkMmRkMjYzODQ4NmU4NDdjMmY5ZjUzNGRmMmM5KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84NGI3NjY1NmYxOTA0ZTU5OGNkMGI3MDllZGU0MjE1ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwgLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzZmMTkwN2Q1YjNjYTQ5NzU4ZGZiMDJjOWU5YThhZDBhID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF85YzFmOGU1ZDgxZGU0MGRiOTE5YWI1MDM4NzAwMmZkZiA9ICQoYDxkaXYgaWQ9Imh0bWxfOWMxZjhlNWQ4MWRlNDBkYjkxOWFiNTAzODcwMDJmZGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxpdHRsZSBQb3J0dWdhbCwgVHJpbml0eSwgV2VzdCBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzZmMTkwN2Q1YjNjYTQ5NzU4ZGZiMDJjOWU5YThhZDBhLnNldENvbnRlbnQoaHRtbF85YzFmOGU1ZDgxZGU0MGRiOTE5YWI1MDM4NzAwMmZkZik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfODRiNzY2NTZmMTkwNGU1OThjZDBiNzA5ZWRlNDIxNWUuYmluZFBvcHVwKHBvcHVwXzZmMTkwN2Q1YjNjYTQ5NzU4ZGZiMDJjOWU5YThhZDBhKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wMzRkMDBiNmFmODM0Mzk4YmVlNmM5ZjdhNjY1YzdjZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNzkyOTIsIC03OS4yNjIwMjk0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYmU3OWU2YWZlNzZiNGE0NzhhMWZlNzkwYjkwNjFlOWQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzAzZTczNWY4ZWM2MjRmM2NhOWYwYjk5Y2MzM2RlZWI5ID0gJChgPGRpdiBpZD0iaHRtbF8wM2U3MzVmOGVjNjI0ZjNjYTlmMGI5OWNjMzNkZWViOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2VubmVkeSBQYXJrLCBJb252aWV3LCBFYXN0IEJpcmNobW91bnQgUGFyaywgU2NhcmJvcm91Z2g8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYmU3OWU2YWZlNzZiNGE0NzhhMWZlNzkwYjkwNjFlOWQuc2V0Q29udGVudChodG1sXzAzZTczNWY4ZWM2MjRmM2NhOWYwYjk5Y2MzM2RlZWI5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8wMzRkMDBiNmFmODM0Mzk4YmVlNmM5ZjdhNjY1YzdjZi5iaW5kUG9wdXAocG9wdXBfYmU3OWU2YWZlNzZiNGE0NzhhMWZlNzkwYjkwNjFlOWQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdlNWYwZDMxY2E0YjQyMWM4Mzk1NzVkMjAzM2I3NjE4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzg2OTQ3MywgLTc5LjM4NTk3NV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNWM1YThiYTUyNDg3NGUwNmFlYmI3MzY2YzRiOWQzYzkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2Y5MGMzOWY4YjgxMDRmMTZiNmE4OWQ3MWZlZGJmNmNjID0gJChgPGRpdiBpZD0iaHRtbF9mOTBjMzlmOGI4MTA0ZjE2YjZhODlkNzFmZWRiZjZjYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF5dmlldyBWaWxsYWdlLCBOb3J0aCBZb3JrPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzVjNWE4YmE1MjQ4NzRlMDZhZWJiNzM2NmM0YjlkM2M5LnNldENvbnRlbnQoaHRtbF9mOTBjMzlmOGI4MTA0ZjE2YjZhODlkNzFmZWRiZjZjYyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfN2U1ZjBkMzFjYTRiNDIxYzgzOTU3NWQyMDMzYjc2MTguYmluZFBvcHVwKHBvcHVwXzVjNWE4YmE1MjQ4NzRlMDZhZWJiNzM2NmM0YjlkM2M5KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zZDhhZjM3OWMzNWM0Y2Q1OGYyODUwODFjMDYzODkxNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczNzQ3MzIwMDAwMDAwNCwgLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF80ODExMjZiZWUzNmU0NjhiYTQ5N2JkOGUxZDhmOWQ0MSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfODNhMDM4ZTkyMDI3NDNjMDliMjAxYzczNjlhZDJhYTYgPSAkKGA8ZGl2IGlkPSJodG1sXzgzYTAzOGU5MjAyNzQzYzA5YjIwMWM3MzY5YWQyYWE2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcsIE5vcnRoIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNDgxMTI2YmVlMzZlNDY4YmE0OTdiZDhlMWQ4ZjlkNDEuc2V0Q29udGVudChodG1sXzgzYTAzOGU5MjAyNzQzYzA5YjIwMWM3MzY5YWQyYWE2KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8zZDhhZjM3OWMzNWM0Y2Q1OGYyODUwODFjMDYzODkxNC5iaW5kUG9wdXAocG9wdXBfNDgxMTI2YmVlMzZlNDY4YmE0OTdiZDhlMWQ4ZjlkNDEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzcyZTU2MTBjYWZmNDRmZmViYzZjMzlkM2MwYTg2MThkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTU3MSwgLTc5LjM1MjE4OF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMDRmOGY3YTdiNmE2NDMzYWFiOWMyZTNkZTkzMmJkZTUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzg2YmJmZDQyNmFhZjQ4N2ZhZjAzZDRiNzc4MjY1ODA0ID0gJChgPGRpdiBpZD0iaHRtbF84NmJiZmQ0MjZhYWY0ODdmYWYwM2Q0Yjc3ODI2NTgwNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIERhbmZvcnRoIFdlc3QsIFJpdmVyZGFsZSwgRWFzdCBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzA0ZjhmN2E3YjZhNjQzM2FhYjljMmUzZGU5MzJiZGU1LnNldENvbnRlbnQoaHRtbF84NmJiZmQ0MjZhYWY0ODdmYWYwM2Q0Yjc3ODI2NTgwNCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNzJlNTYxMGNhZmY0NGZmZWJjNmMzOWQzYzBhODYxOGQuYmluZFBvcHVwKHBvcHVwXzA0ZjhmN2E3YjZhNjQzM2FhYjljMmUzZGU5MzJiZGU1KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNDE0NmUyZmVmNzM0YjRmYTRlNjBhMWI2OGYyYWY0NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsIC03OS4zODE1NzY0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMDYyODBhNGE3MjM5NDUwYTk0NDM5YTFkMjJlNTYwYmEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzEzOTk5ZDFlMTM4MTQxZGFhMjhiZjM1ODJmZGEyMjVmID0gJChgPGRpdiBpZD0iaHRtbF8xMzk5OWQxZTEzODE0MWRhYTI4YmYzNTgyZmRhMjI1ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9yb250byBEb21pbmlvbiBDZW50cmUsIERlc2lnbiBFeGNoYW5nZSwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8wNjI4MGE0YTcyMzk0NTBhOTQ0MzlhMWQyMmU1NjBiYS5zZXRDb250ZW50KGh0bWxfMTM5OTlkMWUxMzgxNDFkYWEyOGJmMzU4MmZkYTIyNWYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzE0MTQ2ZTJmZWY3MzRiNGZhNGU2MGExYjY4ZjJhZjQ0LmJpbmRQb3B1cChwb3B1cF8wNjI4MGE0YTcyMzk0NTBhOTQ0MzlhMWQyMmU1NjBiYSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzQ4MjFhNTZhMjVjNDg5YmFmZGM2ZDU2ZDk2NDIyZmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY4NDcyLCAtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2YwMmViMGY4OTc2YTQxMWY4MWM1OWFkZjcwNmRmN2U1ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9lMmY2NmJkMGM5OGU0ZjAzOTU0YWUwYTM3NzUyNGFmZiA9ICQoYDxkaXYgaWQ9Imh0bWxfZTJmNjZiZDBjOThlNGYwMzk1NGFlMGEzNzc1MjRhZmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrdG9uLCBQYXJrZGFsZSBWaWxsYWdlLCBFeGhpYml0aW9uIFBsYWNlLCBXZXN0IFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZjAyZWIwZjg5NzZhNDExZjgxYzU5YWRmNzA2ZGY3ZTUuc2V0Q29udGVudChodG1sX2UyZjY2YmQwYzk4ZTRmMDM5NTRhZTBhMzc3NTI0YWZmKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8zNDgyMWE1NmEyNWM0ODliYWZkYzZkNTZkOTY0MjJmYS5iaW5kUG9wdXAocG9wdXBfZjAyZWIwZjg5NzZhNDExZjgxYzU5YWRmNzA2ZGY3ZTUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZiMDhlNjdlNzI3OTQ2NzliYjg0N2FhYjdlODY4ZjVhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzExMTExNzAwMDAwMDA0LCAtNzkuMjg0NTc3Ml0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNzNjZDE0NDFmMzFjNDRlNWJmYjNlNmRmZWJmY2MzNjcgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzVhNDE2NTI0NDIyNjQ3ZDZiNzg2MDUzMWRiNTIzMmM5ID0gJChgPGRpdiBpZD0iaHRtbF81YTQxNjUyNDQyMjY0N2Q2Yjc4NjA1MzFkYjUyMzJjOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R29sZGVuIE1pbGUsIENsYWlybGVhLCBPYWtyaWRnZSwgU2NhcmJvcm91Z2g8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNzNjZDE0NDFmMzFjNDRlNWJmYjNlNmRmZWJmY2MzNjcuc2V0Q29udGVudChodG1sXzVhNDE2NTI0NDIyNjQ3ZDZiNzg2MDUzMWRiNTIzMmM5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9mYjA4ZTY3ZTcyNzk0Njc5YmI4NDdhYWI3ZTg2OGY1YS5iaW5kUG9wdXAocG9wdXBfNzNjZDE0NDFmMzFjNDRlNWJmYjNlNmRmZWJmY2MzNjcpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FiNDFlMjA2OGJjODRiMzdiODUyOWRhNjBiNDNlY2IyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU3NDkwMiwgLTc5LjM3NDcxNDA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82M2VjYmNlMGY2ZTY0NDU0YjU2OTY3NmYxNTBlMzU1YiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNTUwZTQzOGIwMWViNGE5NDhkZGFhNDMxZGI0MDc2NjkgPSAkKGA8ZGl2IGlkPSJodG1sXzU1MGU0MzhiMDFlYjRhOTQ4ZGRhYTQzMWRiNDA3NjY5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Zb3JrIE1pbGxzLCBTaWx2ZXIgSGlsbHMsIE5vcnRoIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNjNlY2JjZTBmNmU2NDQ1NGI1Njk2NzZmMTUwZTM1NWIuc2V0Q29udGVudChodG1sXzU1MGU0MzhiMDFlYjRhOTQ4ZGRhYTQzMWRiNDA3NjY5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9hYjQxZTIwNjhiYzg0YjM3Yjg1MjlkYTYwYjQzZWNiMi5iaW5kUG9wdXAocG9wdXBfNjNlY2JjZTBmNmU2NDQ1NGI1Njk2NzZmMTUwZTM1NWIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FiZjMzYmVjYTUzZjRkNThhMzQ4MDIzODM3ZDg0YTdmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM5MDE0NiwgLTc5LjUwNjk0MzZdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzlmNGFiNWYyNTA5ZDQ5N2FhZjMyZWM1MTBhOWQzNWViID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF84NGMxZDBlYmUxODg0MTYxODgxNjAyNzlmYzg1OTIyMiA9ICQoYDxkaXYgaWQ9Imh0bWxfODRjMWQwZWJlMTg4NDE2MTg4MTYwMjc5ZmM4NTkyMjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldywgTm9ydGggWW9yazwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF85ZjRhYjVmMjUwOWQ0OTdhYWYzMmVjNTEwYTlkMzVlYi5zZXRDb250ZW50KGh0bWxfODRjMWQwZWJlMTg4NDE2MTg4MTYwMjc5ZmM4NTkyMjIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2FiZjMzYmVjYTUzZjRkNThhMzQ4MDIzODM3ZDg0YTdmLmJpbmRQb3B1cChwb3B1cF85ZjRhYjVmMjUwOWQ0OTdhYWYzMmVjNTEwYTlkMzVlYikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzJjZTczZDYyODhiNDY0YThlZTllNzk4ZDYzY2IxMDYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njg5OTg1LCAtNzkuMzE1NTcxNTk5OTk5OThdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzNkMDMyZDEwYzg0MjQ5OGM5ZGZkNjM4ZDMxOTkyMTE0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9mNWYxODUxYjA0ZDM0ZjcyYjc0YzJmMDIzYjJmNWMzOCA9ICQoYDxkaXYgaWQ9Imh0bWxfZjVmMTg1MWIwNGQzNGY3MmI3NGMyZjAyM2IyZjVjMzgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkluZGlhIEJhemFhciwgVGhlIEJlYWNoZXMgV2VzdCwgRWFzdCBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzNkMDMyZDEwYzg0MjQ5OGM5ZGZkNjM4ZDMxOTkyMTE0LnNldENvbnRlbnQoaHRtbF9mNWYxODUxYjA0ZDM0ZjcyYjc0YzJmMDIzYjJmNWMzOCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNzJjZTczZDYyODhiNDY0YThlZTllNzk4ZDYzY2IxMDYuYmluZFBvcHVwKHBvcHVwXzNkMDMyZDEwYzg0MjQ5OGM5ZGZkNjM4ZDMxOTkyMTE0KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83MDQzZDAwNDRkYjA0MTBiODU3YzdmNmVjN2FhYTEwYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODE5ODUsIC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfY2MzZWQ1ZTc2MjYzNDEwMmFmNmExZjBkZTQ0OGE1MDkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzIwMmI0ZGM5MGI2MjRkM2Q5MTgzNzM2NmFkYTVmMmRmID0gJChgPGRpdiBpZD0iaHRtbF8yMDJiNGRjOTBiNjI0ZDNkOTE4MzczNjZhZGE1ZjJkZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q29tbWVyY2UgQ291cnQsIFZpY3RvcmlhIEhvdGVsLCBEb3dudG93biBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2NjM2VkNWU3NjI2MzQxMDJhZjZhMWYwZGU0NDhhNTA5LnNldENvbnRlbnQoaHRtbF8yMDJiNGRjOTBiNjI0ZDNkOTE4MzczNjZhZGE1ZjJkZik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNzA0M2QwMDQ0ZGIwNDEwYjg1N2M3ZjZlYzdhYWExMGMuYmluZFBvcHVwKHBvcHVwX2NjM2VkNWU3NjI2MzQxMDJhZjZhMWYwZGU0NDhhNTA5KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xN2I3ODQ0NTA1Nzg0ODU2YmE2Y2JjOGZhMGIzNDQxYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMzc1NjIwMDAwMDAwNiwgLTc5LjQ5MDA3MzhdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2YwMzU2NmMwNWVkNjQ4MjFhNTYyODJmZDQ5NDRhMGU5ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9jYTRlZjVlNmRmOGY0YzljYjlhMDc3YTAzNDczM2UzYiA9ICQoYDxkaXYgaWQ9Imh0bWxfY2E0ZWY1ZTZkZjhmNGM5Y2I5YTA3N2EwMzQ3MzNlM2IiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRoIFBhcmssIE1hcGxlIExlYWYgUGFyaywgVXB3b29kIFBhcmssIE5vcnRoIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZjAzNTY2YzA1ZWQ2NDgyMWE1NjI4MmZkNDk0NGEwZTkuc2V0Q29udGVudChodG1sX2NhNGVmNWU2ZGY4ZjRjOWNiOWEwNzdhMDM0NzMzZTNiKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8xN2I3ODQ0NTA1Nzg0ODU2YmE2Y2JjOGZhMGIzNDQxYy5iaW5kUG9wdXAocG9wdXBfZjAzNTY2YzA1ZWQ2NDgyMWE1NjI4MmZkNDk0NGEwZTkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNjYTMzZjQxODEyZTRjYmU5ZjE2YTU3YzBkNzkyYzY3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU2MzAzMywgLTc5LjU2NTk2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8zNjE0ZmY3Mzc2NTU0NDg5YTdjYjYxOTY4ZWY0MWE2NyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfM2Q4NjMxZWRlYmQ2NDJkMDk3Yjk0Yjk0NWY0MTliZmQgPSAkKGA8ZGl2IGlkPSJodG1sXzNkODYzMWVkZWJkNjQyZDA5N2I5NGI5NDVmNDE5YmZkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IdW1iZXIgU3VtbWl0LCBOb3J0aCBZb3JrPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzM2MTRmZjczNzY1NTQ0ODlhN2NiNjE5NjhlZjQxYTY3LnNldENvbnRlbnQoaHRtbF8zZDg2MzFlZGViZDY0MmQwOTdiOTRiOTQ1ZjQxOWJmZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfM2NhMzNmNDE4MTJlNGNiZTlmMTZhNTdjMGQ3OTJjNjcuYmluZFBvcHVwKHBvcHVwXzM2MTRmZjczNzY1NTQ0ODlhN2NiNjE5NjhlZjQxYTY3KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wMzdlNDFmMTU3ZDY0MzI0YmMzZjY1OWZlNGYxZGFjYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxNjMxNiwgLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82NjE0MjliYTU1NmI0MzE3YmUyNTcxYzk4ZDY5MmZjOSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNGZmYzk2YTJmNzA4NDk1Njk1ZTk0MjBiZWRhODRiZGEgPSAkKGA8ZGl2IGlkPSJodG1sXzRmZmM5NmEyZjcwODQ5NTY5NWU5NDIwYmVkYTg0YmRhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DbGlmZnNpZGUsIENsaWZmY3Jlc3QsIFNjYXJib3JvdWdoIFZpbGxhZ2UgV2VzdCwgU2NhcmJvcm91Z2g8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNjYxNDI5YmE1NTZiNDMxN2JlMjU3MWM5OGQ2OTJmYzkuc2V0Q29udGVudChodG1sXzRmZmM5NmEyZjcwODQ5NTY5NWU5NDIwYmVkYTg0YmRhKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8wMzdlNDFmMTU3ZDY0MzI0YmMzZjY1OWZlNGYxZGFjYi5iaW5kUG9wdXAocG9wdXBfNjYxNDI5YmE1NTZiNDMxN2JlMjU3MWM5OGQ2OTJmYzkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NjMzAwMTQ1YmQ4ODQyMjlhZGY3YzU1M2I2ZjkxNmUzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzg5MDUzLCAtNzkuNDA4NDkyNzk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzM4NzM0ZTlhMTIzMTQ1NmNhZmUxMTA4Yjk3NGFkMzg4ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kNTg4ZmI2NDMzMmI0OTU5YWQzYmYyMmU2YzUwMjFmOSA9ICQoYDxkaXYgaWQ9Imh0bWxfZDU4OGZiNjQzMzJiNDk1OWFkM2JmMjJlNmM1MDIxZjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldpbGxvd2RhbGUsIE5ld3RvbmJyb29rLCBOb3J0aCBZb3JrPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzM4NzM0ZTlhMTIzMTQ1NmNhZmUxMTA4Yjk3NGFkMzg4LnNldENvbnRlbnQoaHRtbF9kNTg4ZmI2NDMzMmI0OTU5YWQzYmYyMmU2YzUwMjFmOSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfY2MzMDAxNDViZDg4NDIyOWFkZjdjNTUzYjZmOTE2ZTMuYmluZFBvcHVwKHBvcHVwXzM4NzM0ZTlhMTIzMTQ1NmNhZmUxMTA4Yjk3NGFkMzg4KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNmMzOGNiZTQ5MDY0YTAyOGUzYTE2ZGI1MDJhNjJmMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODQ5NjQsIC03OS40OTU2OTc0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTE1MWIyOTgxYmZlNDQ1NThjZGIyMjRkN2VkODI2OGMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2I3MGI0OTNkNjI5NjRiODdiM2RjZmJkZWM2MmVlMjQyID0gJChgPGRpdiBpZD0iaHRtbF9iNzBiNDkzZDYyOTY0Yjg3YjNkY2ZiZGVjNjJlZTI0MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnN2aWV3LCBOb3J0aCBZb3JrPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2UxNTFiMjk4MWJmZTQ0NTU4Y2RiMjI0ZDdlZDgyNjhjLnNldENvbnRlbnQoaHRtbF9iNzBiNDkzZDYyOTY0Yjg3YjNkY2ZiZGVjNjJlZTI0Mik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfZjZjMzhjYmU0OTA2NGEwMjhlM2ExNmRiNTAyYTYyZjAuYmluZFBvcHVwKHBvcHVwX2UxNTFiMjk4MWJmZTQ0NTU4Y2RiMjI0ZDdlZDgyNjhjKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MmJlNzU1N2ViZjg0MzJlYjUyZDZjNWQ2YTlhMmE4ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1OTUyNTUsIC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2FlOGE2NGQ5M2I1ZjRlNzA4YmEzMzFkZDllZGNiNGRlID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF81MTFhYzZiZDYzYTM0Y2U5YTZmNWRiYTE5ODVmYzZiYyA9ICQoYDxkaXYgaWQ9Imh0bWxfNTExYWM2YmQ2M2EzNGNlOWE2ZjVkYmExOTg1ZmM2YmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0dWRpbyBEaXN0cmljdCwgRWFzdCBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2FlOGE2NGQ5M2I1ZjRlNzA4YmEzMzFkZDllZGNiNGRlLnNldENvbnRlbnQoaHRtbF81MTFhYzZiZDYzYTM0Y2U5YTZmNWRiYTE5ODVmYzZiYyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNTJiZTc1NTdlYmY4NDMyZWI1MmQ2YzVkNmE5YTJhOGQuYmluZFBvcHVwKHBvcHVwX2FlOGE2NGQ5M2I1ZjRlNzA4YmEzMzFkZDllZGNiNGRlKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zMTViN2Y3Nzg4MTg0YzNiODE3ZWU1MzE2Mjc3OWNiNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczMzI4MjUsIC03OS40MTk3NDk3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF80MjFiNTQ0YzIwZTQ0N2M1YWU2OTFiMThiZDhmMDhmNCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYmI0ODRhNTVmNDhkNDVkMmI4MDc5N2Q4N2ZiNThhYWMgPSAkKGA8ZGl2IGlkPSJodG1sX2JiNDg0YTU1ZjQ4ZDQ1ZDJiODA3OTdkODdmYjU4YWFjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZWRmb3JkIFBhcmssIExhd3JlbmNlIE1hbm9yIEVhc3QsIE5vcnRoIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNDIxYjU0NGMyMGU0NDdjNWFlNjkxYjE4YmQ4ZjA4ZjQuc2V0Q29udGVudChodG1sX2JiNDg0YTU1ZjQ4ZDQ1ZDJiODA3OTdkODdmYjU4YWFjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8zMTViN2Y3Nzg4MTg0YzNiODE3ZWU1MzE2Mjc3OWNiNi5iaW5kUG9wdXAocG9wdXBfNDIxYjU0NGMyMGU0NDdjNWFlNjkxYjE4YmQ4ZjA4ZjQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRhM2I3YjExNzUyYjQwYzRiYzgzY2EyMTkyYjRlZDIyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkxMTE1OCwgLTc5LjQ3NjAxMzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF80OTU0NmI0MjU1Zjc0NjMwYjg0YzAxNDkwZjY4NTMzNCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZTU1NThkMDcwMDVkNDUwOWJiNWZlMmRiYmY3ZTAyMDQgPSAkKGA8ZGl2IGlkPSJodG1sX2U1NTU4ZDA3MDA1ZDQ1MDliYjVmZTJkYmJmN2UwMjA0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZWwgUmF5LCBNb3VudCBEZW5uaXMsIEtlZWxzZGFsZSBhbmQgU2lsdmVydGhvcm4sIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNDk1NDZiNDI1NWY3NDYzMGI4NGMwMTQ5MGY2ODUzMzQuc2V0Q29udGVudChodG1sX2U1NTU4ZDA3MDA1ZDQ1MDliYjVmZTJkYmJmN2UwMjA0KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl80YTNiN2IxMTc1MmI0MGM0YmM4M2NhMjE5MmI0ZWQyMi5iaW5kUG9wdXAocG9wdXBfNDk1NDZiNDI1NWY3NDYzMGI4NGMwMTQ5MGY2ODUzMzQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhhM2IyYTg5NDlmNzQ2MmVhYjRiNjQwYThiNDhlMjUyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI0NzY1OSwgLTc5LjUzMjI0MjQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF81YzliNmRiMGY3NTY0ZDUwODViYzllNDU4OWEzZTQwMiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfOGQ3MWNjMjllZmJmNGI4OWFhOGZkODAwYWM1NzMwODMgPSAkKGA8ZGl2IGlkPSJodG1sXzhkNzFjYzI5ZWZiZjRiODlhYThmZDgwMGFjNTczMDgzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IdW1iZXJsZWEsIEVtZXJ5LCBOb3J0aCBZb3JrPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzVjOWI2ZGIwZjc1NjRkNTA4NWJjOWU0NTg5YTNlNDAyLnNldENvbnRlbnQoaHRtbF84ZDcxY2MyOWVmYmY0Yjg5YWE4ZmQ4MDBhYzU3MzA4Myk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfOGEzYjJhODk0OWY3NDYyZWFiNGI2NDBhOGI0OGUyNTIuYmluZFBvcHVwKHBvcHVwXzVjOWI2ZGIwZjc1NjRkNTA4NWJjOWU0NTg5YTNlNDAyKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81OGJmYzBlOTA1MjY0MGFmODJhMGFmODdhZWY4Y2ZhMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5MjY1NzAwMDAwMDAwNCwgLTc5LjI2NDg0ODFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2ZiYmM0NmQwODUyYzQwMjlhNjRiYjAxOTUzZDM4N2Q5ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9lOGNmNjE3MjI2MTM0YTM2ODY1YTllODJiNzdjNWIxNiA9ICQoYDxkaXYgaWQ9Imh0bWxfZThjZjYxNzIyNjEzNGEzNjg2NWE5ZTgyYjc3YzViMTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJpcmNoIENsaWZmLCBDbGlmZnNpZGUgV2VzdCwgU2NhcmJvcm91Z2g8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZmJiYzQ2ZDA4NTJjNDAyOWE2NGJiMDE5NTNkMzg3ZDkuc2V0Q29udGVudChodG1sX2U4Y2Y2MTcyMjYxMzRhMzY4NjVhOWU4MmI3N2M1YjE2KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl81OGJmYzBlOTA1MjY0MGFmODJhMGFmODdhZWY4Y2ZhMi5iaW5kUG9wdXAocG9wdXBfZmJiYzQ2ZDA4NTJjNDAyOWE2NGJiMDE5NTNkMzg3ZDkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFmOWFlM2JjMmRhMzQwMjU5YWMyYzczNWM4MWU5OTc2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzcwMTE5OSwgLTc5LjQwODQ5Mjc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8zMTVhOGZmNTEwYTE0MTFlOTAwNzEwZDE3YTEyNmY2YiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMjIxNTI3ZjI1NjNhNDcxMDg5YTIxM2ZkZjc1YTM0ZmMgPSAkKGA8ZGl2IGlkPSJodG1sXzIyMTUyN2YyNTYzYTQ3MTA4OWEyMTNmZGY3NWEzNGZjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XaWxsb3dkYWxlLCBXaWxsb3dkYWxlIEVhc3QsIE5vcnRoIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMzE1YThmZjUxMGExNDExZTkwMDcxMGQxN2ExMjZmNmIuc2V0Q29udGVudChodG1sXzIyMTUyN2YyNTYzYTQ3MTA4OWEyMTNmZGY3NWEzNGZjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8xZjlhZTNiYzJkYTM0MDI1OWFjMmM3MzVjODFlOTk3Ni5iaW5kUG9wdXAocG9wdXBfMzE1YThmZjUxMGExNDExZTkwMDcxMGQxN2ExMjZmNmIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUxYzdiYzNhM2NlZDQwMDhhMDE5YmVhNjRhMzBhYWU5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzYxNjMxMywgLTc5LjUyMDk5OTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9jMDc5MjM2ZTJkMmU0NTQ4OTAwMmJlN2Q3MDcyYWQ2OSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfN2ZiODRlZTQ1ZmE3NDRjMzhjOGQzOTE4OTQ0MzlmYTkgPSAkKGA8ZGl2IGlkPSJodG1sXzdmYjg0ZWU0NWZhNzQ0YzM4YzhkMzkxODk0NDM5ZmE5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcsIE5vcnRoIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYzA3OTIzNmUyZDJlNDU0ODkwMDJiZTdkNzA3MmFkNjkuc2V0Q29udGVudChodG1sXzdmYjg0ZWU0NWZhNzQ0YzM4YzhkMzkxODk0NDM5ZmE5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl81MWM3YmMzYTNjZWQ0MDA4YTAxOWJlYTY0YTMwYWFlOS5iaW5kUG9wdXAocG9wdXBfYzA3OTIzNmUyZDJlNDU0ODkwMDJiZTdkNzA3MmFkNjkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzliNDc1YjhiZmJlNjRiZjc4NjVmZTIxY2ZiNjhmZGFhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4MDIwNSwgLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzY5OTdkNzk0MDUzNjQ4Y2M4NWViODVhZWJhZmY0MTAyID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8yMWEzYjBmZTEwZDA0YTk5OTc3OGNkMzcyOTA3NWZkZSA9ICQoYDxkaXYgaWQ9Imh0bWxfMjFhM2IwZmUxMGQwNGE5OTk3NzhjZDM3MjkwNzVmZGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxhd3JlbmNlIFBhcmssIENlbnRyYWwgVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF82OTk3ZDc5NDA1MzY0OGNjODVlYjg1YWViYWZmNDEwMi5zZXRDb250ZW50KGh0bWxfMjFhM2IwZmUxMGQwNGE5OTk3NzhjZDM3MjkwNzVmZGUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzliNDc1YjhiZmJlNjRiZjc4NjVmZTIxY2ZiNjhmZGFhLmJpbmRQb3B1cChwb3B1cF82OTk3ZDc5NDA1MzY0OGNjODVlYjg1YWViYWZmNDEwMikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTZiNzNiMTQ3M2UyNDJjNTk1ZTA2ZWIzMDU2YTcyMzIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTE2OTQ4LCAtNzkuNDE2OTM1NTk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2RhMGNkNjUyYmIwMjQ4ODdhOTA3ZWYxY2I3NzcwMDQyID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9mNjYxM2NiMDlhOTM0Y2VlODcyZWVmYjBlMDJjNTQwZiA9ICQoYDxkaXYgaWQ9Imh0bWxfZjY2MTNjYjA5YTkzNGNlZTg3MmVlZmIwZTAyYzU0MGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VsYXduLCBDZW50cmFsIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZGEwY2Q2NTJiYjAyNDg4N2E5MDdlZjFjYjc3NzAwNDIuc2V0Q29udGVudChodG1sX2Y2NjEzY2IwOWE5MzRjZWU4NzJlZWZiMGUwMmM1NDBmKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9lNmI3M2IxNDczZTI0MmM1OTVlMDZlYjMwNTZhNzIzMi5iaW5kUG9wdXAocG9wdXBfZGEwY2Q2NTJiYjAyNDg4N2E5MDdlZjFjYjc3NzAwNDIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYwM2EwNTM3NGNkNTQ5YzdiM2IxNmYxOWIyYjEyM2UyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjczMTg1Mjk5OTk5OTksIC03OS40ODcyNjE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMjhjNTIzZjFlYTY4NDZmMGI2YWMzOTBkZWZlYWUxMDUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzU0N2Y3NWI1ODE4ZDQyOWRiOWFlNDAzYjdmNDY1MWMyID0gJChgPGRpdiBpZD0iaHRtbF81NDdmNzViNTgxOGQ0MjlkYjlhZTQwM2I3ZjQ2NTFjMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBUaGUgSnVuY3Rpb24gTm9ydGgsIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMjhjNTIzZjFlYTY4NDZmMGI2YWMzOTBkZWZlYWUxMDUuc2V0Q29udGVudChodG1sXzU0N2Y3NWI1ODE4ZDQyOWRiOWFlNDAzYjdmNDY1MWMyKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl82MDNhMDUzNzRjZDU0OWM3YjNiMTZmMTliMmIxMjNlMi5iaW5kUG9wdXAocG9wdXBfMjhjNTIzZjFlYTY4NDZmMGI2YWMzOTBkZWZlYWUxMDUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Q4N2Q1MTY1ZmEwMTQ3M2VhOWM3ZGRiM2QxZjEzMDhlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2ODc2LCAtNzkuNTE4MTg4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2YyYjRhZTcwY2MxNDRlOWJhYWRkYmNmMWM4Nzg2NGY0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF84ZGZjZGRjNjE3OTc0NjYwYjBiZTNlY2UxZDcyNmJhZiA9ICQoYDxkaXYgaWQ9Imh0bWxfOGRmY2RkYzYxNzk3NDY2MGIwYmUzZWNlMWQ3MjZiYWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3RvbiwgWW9yazwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9mMmI0YWU3MGNjMTQ0ZTliYWFkZGJjZjFjODc4NjRmNC5zZXRDb250ZW50KGh0bWxfOGRmY2RkYzYxNzk3NDY2MGIwYmUzZWNlMWQ3MjZiYWYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2Q4N2Q1MTY1ZmEwMTQ3M2VhOWM3ZGRiM2QxZjEzMDhlLmJpbmRQb3B1cChwb3B1cF9mMmI0YWU3MGNjMTQ0ZTliYWFkZGJjZjFjODc4NjRmNCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTQ5YzYwNmNlODAyNDJiNmIxNjFjYzY0YTY0NzJlOGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTc0MDk2LCAtNzkuMjczMzA0MDAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzQzODg4NjczMGE4NDQ5YmU5MDRkODY0MTc1MjNiYjJiID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF82ZjM2M2RiYWZlOTk0MTEwODFkMGJhOTk5MTZmMjhkNyA9ICQoYDxkaXYgaWQ9Imh0bWxfNmYzNjNkYmFmZTk5NDExMDgxZDBiYTk5OTE2ZjI4ZDciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvcnNldCBQYXJrLCBXZXhmb3JkIEhlaWdodHMsIFNjYXJib3JvdWdoIFRvd24gQ2VudHJlLCBTY2FyYm9yb3VnaDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF80Mzg4ODY3MzBhODQ0OWJlOTA0ZDg2NDE3NTIzYmIyYi5zZXRDb250ZW50KGh0bWxfNmYzNjNkYmFmZTk5NDExMDgxZDBiYTk5OTE2ZjI4ZDcpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzE0OWM2MDZjZTgwMjQyYjZiMTYxY2M2NGE2NDcyZThmLmJpbmRQb3B1cChwb3B1cF80Mzg4ODY3MzBhODQ0OWJlOTA0ZDg2NDE3NTIzYmIyYikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGE1NzEwZDY1OGY2NGZlNGJiZDk5YjczMGJhNTEyYzEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTI3NTgyOTk5OTk5OTYsIC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iZTMxY2E1OThkZWQ0Yzk0YmFmYzMyNDNhNzBjYWU4YyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfODcxY2Y1MDE1Y2EyNDNlNWI4Mjk5MmRlZjNmNjIwOTcgPSAkKGA8ZGl2IGlkPSJodG1sXzg3MWNmNTAxNWNhMjQzZTViODI5OTJkZWYzZjYyMDk3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Zb3JrIE1pbGxzIFdlc3QsIE5vcnRoIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYmUzMWNhNTk4ZGVkNGM5NGJhZmMzMjQzYTcwY2FlOGMuc2V0Q29udGVudChodG1sXzg3MWNmNTAxNWNhMjQzZTViODI5OTJkZWYzZjYyMDk3KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9kYTU3MTBkNjU4ZjY0ZmU0YmJkOTliNzMwYmE1MTJjMS5iaW5kUG9wdXAocG9wdXBfYmUzMWNhNTk4ZGVkNGM5NGJhZmMzMjQzYTcwY2FlOGMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FlN2MyNjFiNWQ0MjQ3M2M4MWU5NWI2ZTYwMTA3YjkwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEyNzUxMSwgLTc5LjM5MDE5NzVdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2Q0MTNmOTEzZDMwNTRlZmY4MGFmZjFjODE2ZWNiMzIzID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kZmYxMGUxYzY5ZTU0MmUwYWVhNWJkM2Q0M2RlMWNhZiA9ICQoYDxkaXYgaWQ9Imh0bWxfZGZmMTBlMWM2OWU1NDJlMGFlYTViZDNkNDNkZTFjYWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgTm9ydGgsIENlbnRyYWwgVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9kNDEzZjkxM2QzMDU0ZWZmODBhZmYxYzgxNmVjYjMyMy5zZXRDb250ZW50KGh0bWxfZGZmMTBlMWM2OWU1NDJlMGFlYTViZDNkNDNkZTFjYWYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2FlN2MyNjFiNWQ0MjQ3M2M4MWU5NWI2ZTYwMTA3YjkwLmJpbmRQb3B1cChwb3B1cF9kNDEzZjkxM2QzMDU0ZWZmODBhZmYxYzgxNmVjYjMyMykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTk2MjlhZGYwOTM2NDc5Mzk3NDQzYmZmZDNmYzc2ZDggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTY5NDc2LCAtNzkuNDExMzA3MjAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2EwYzJhY2Y3ZjJmMTRkNDI5YTFmNGY3ODgwMGY5M2JmID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF82MzY1ZjBhNWM4OGY0ZjRlOTQ4YmViZjg4OTFiNzhmNyA9ICQoYDxkaXYgaWQ9Imh0bWxfNjM2NWYwYTVjODhmNGY0ZTk0OGJlYmY4ODkxYjc4ZjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZvcmVzdCBIaWxsIE5vcnRoICZhbXA7IFdlc3QsIEZvcmVzdCBIaWxsIFJvYWQgUGFyaywgQ2VudHJhbCBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2EwYzJhY2Y3ZjJmMTRkNDI5YTFmNGY3ODgwMGY5M2JmLnNldENvbnRlbnQoaHRtbF82MzY1ZjBhNWM4OGY0ZjRlOTQ4YmViZjg4OTFiNzhmNyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNTk2MjlhZGYwOTM2NDc5Mzk3NDQzYmZmZDNmYzc2ZDguYmluZFBvcHVwKHBvcHVwX2EwYzJhY2Y3ZjJmMTRkNDI5YTFmNGY3ODgwMGY5M2JmKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80YWU0Mjc5NDIxYTg0YzRmYjc1ODkwZjQ0ZDIyYzA1OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MTYwODMsIC03OS40NjQ3NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfOGM4MzU3YTNjYTUxNGQxZWIzMzJmOTI1ZmQ2ZWI5YjkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzczYjZhODI3OTRmOTQyZjNiYzQyYmE1ZGE0MjdiODQyID0gJChgPGRpdiBpZD0iaHRtbF83M2I2YTgyNzk0Zjk0MmYzYmM0MmJhNWRhNDI3Yjg0MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGlnaCBQYXJrLCBUaGUgSnVuY3Rpb24gU291dGgsIFdlc3QgVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF84YzgzNTdhM2NhNTE0ZDFlYjMzMmY5MjVmZDZlYjliOS5zZXRDb250ZW50KGh0bWxfNzNiNmE4Mjc5NGY5NDJmM2JjNDJiYTVkYTQyN2I4NDIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzRhZTQyNzk0MjFhODRjNGZiNzU4OTBmNDRkMjJjMDU5LmJpbmRQb3B1cChwb3B1cF84YzgzNTdhM2NhNTE0ZDFlYjMzMmY5MjVmZDZlYjliOSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjQyZTk1ZmVlZGE3NDk3NWEwYmY0ZGMwOWI4YzgwYjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTYzMTksIC03OS41MzIyNDI0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZmVkYTU4ODM3NjkyNDhiNGJhOGJmMGJiN2ZkMDg4MWMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2E5NWRjZjgzZDJkNTQzYjdhYjM2NzhkMGEzNDM0MWUxID0gJChgPGRpdiBpZD0iaHRtbF9hOTVkY2Y4M2QyZDU0M2I3YWIzNjc4ZDBhMzQzNDFlMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdG1vdW50LCBFdG9iaWNva2U8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZmVkYTU4ODM3NjkyNDhiNGJhOGJmMGJiN2ZkMDg4MWMuc2V0Q29udGVudChodG1sX2E5NWRjZjgzZDJkNTQzYjdhYjM2NzhkMGEzNDM0MWUxKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl82NDJlOTVmZWVkYTc0OTc1YTBiZjRkYzA5YjhjODBiMi5iaW5kUG9wdXAocG9wdXBfZmVkYTU4ODM3NjkyNDhiNGJhOGJmMGJiN2ZkMDg4MWMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNjMGY1ZmQwZTg3YjRhNmQ4ZTEyNmI2ZjAwYmU5NTNjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzUwMDcxNTAwMDAwMDA0LCAtNzkuMjk1ODQ5MV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYjJiNGE0ODRhN2UwNDNmNmIwNDYwODMzMGEwYzQ5ZTcgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2E2MTFjNmU4MWNiNjQ2MTk5ZTQ4Y2JiZjcxMGQ3MzlkID0gJChgPGRpdiBpZD0iaHRtbF9hNjExYzZlODFjYjY0NjE5OWU0OGNiYmY3MTBkNzM5ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2V4Zm9yZCwgTWFyeXZhbGUsIFNjYXJib3JvdWdoPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2IyYjRhNDg0YTdlMDQzZjZiMDQ2MDgzMzBhMGM0OWU3LnNldENvbnRlbnQoaHRtbF9hNjExYzZlODFjYjY0NjE5OWU0OGNiYmY3MTBkNzM5ZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfM2MwZjVmZDBlODdiNGE2ZDhlMTI2YjZmMDBiZTk1M2MuYmluZFBvcHVwKHBvcHVwX2IyYjRhNDg0YTdlMDQzZjZiMDQ2MDgzMzBhMGM0OWU3KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wOGNjNjI1ZGYyN2Y0YTg2YWE0MTYxZTg3NDk0YWI1NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4MjczNjQsIC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iODVmNmZmYjI3Zjc0ODU0YTU4NmYyMGZiY2Q1MGQwOSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZjMwODU3ODRlMzAzNDIyODkzZTc1Mzg3ZjViZWJjZjQgPSAkKGA8ZGl2IGlkPSJodG1sX2YzMDg1Nzg0ZTMwMzQyMjg5M2U3NTM4N2Y1YmViY2Y0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XaWxsb3dkYWxlLCBXaWxsb3dkYWxlIFdlc3QsIE5vcnRoIFlvcms8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYjg1ZjZmZmIyN2Y3NDg1NGE1ODZmMjBmYmNkNTBkMDkuc2V0Q29udGVudChodG1sX2YzMDg1Nzg0ZTMwMzQyMjg5M2U3NTM4N2Y1YmViY2Y0KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8wOGNjNjI1ZGYyN2Y0YTg2YWE0MTYxZTg3NDk0YWI1NS5iaW5kUG9wdXAocG9wdXBfYjg1ZjZmZmIyN2Y3NDg1NGE1ODZmMjBmYmNkNTBkMDkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQxYmE0NDRlOGU4YzRkNGJiODViZjE5MzUzNzI3NjQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE1MzgzNCwgLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hM2Y2OGJmMjM4OTQ0YmYwOThhZDUxNGRjMjUxODE1ZiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMmViYjg1ZGFmMmFjNDg5YWIwYmIzYWQwOTg0YzhkNjUgPSAkKGA8ZGl2IGlkPSJodG1sXzJlYmI4NWRhZjJhYzQ4OWFiMGJiM2FkMDk4NGM4ZDY1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBUb3JvbnRvIFdlc3QsIExhd3JlbmNlIFBhcmssIENlbnRyYWwgVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9hM2Y2OGJmMjM4OTQ0YmYwOThhZDUxNGRjMjUxODE1Zi5zZXRDb250ZW50KGh0bWxfMmViYjg1ZGFmMmFjNDg5YWIwYmIzYWQwOTg0YzhkNjUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzQxYmE0NDRlOGU4YzRkNGJiODViZjE5MzUzNzI3NjQyLmJpbmRQb3B1cChwb3B1cF9hM2Y2OGJmMjM4OTQ0YmYwOThhZDUxNGRjMjUxODE1ZikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGE5ZjA2NDZiODNkNGJmMGEzMDFhMjA4NjJlNzI4ZDIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzI3MDk3LCAtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2YzNDE3NWU1ZWQzMDQ5ZTZiNGEyYTM4YjM4N2M3NWFkID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF82MTFlMjYwNGRhODg0ZTE0OGFmN2JlYzNlM2JmZGQxZiA9ICQoYDxkaXYgaWQ9Imh0bWxfNjExZTI2MDRkYTg4NGUxNDhhZjdiZWMzZTNiZmRkMWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBBbm5leCwgTm9ydGggTWlkdG93biwgWW9ya3ZpbGxlLCBDZW50cmFsIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZjM0MTc1ZTVlZDMwNDllNmI0YTJhMzhiMzg3Yzc1YWQuc2V0Q29udGVudChodG1sXzYxMWUyNjA0ZGE4ODRlMTQ4YWY3YmVjM2UzYmZkZDFmKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9kYTlmMDY0NmI4M2Q0YmYwYTMwMWEyMDg2MmU3MjhkMi5iaW5kUG9wdXAocG9wdXBfZjM0MTc1ZTVlZDMwNDllNmI0YTJhMzhiMzg3Yzc1YWQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IzYmU0Y2JjZTI1ODQwM2ZiNWM2ZmFiYjlmNDA3NjY2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywgLTc5LjQ1NjMyNV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMWM5ZWMwMTkxYWMyNDIxMGE4MTk4MDE0MjNiYjg2ZGMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzhmM2RjMmM5MzIwODQ3MDg5MWE2OWRkYjIzY2ZjZWFiID0gJChgPGRpdiBpZD0iaHRtbF84ZjNkYzJjOTMyMDg0NzA4OTFhNjlkZGIyM2NmY2VhYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya2RhbGUsIFJvbmNlc3ZhbGxlcywgV2VzdCBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzFjOWVjMDE5MWFjMjQyMTBhODE5ODAxNDIzYmI4NmRjLnNldENvbnRlbnQoaHRtbF84ZjNkYzJjOTMyMDg0NzA4OTFhNjlkZGIyM2NmY2VhYik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYjNiZTRjYmNlMjU4NDAzZmI1YzZmYWJiOWY0MDc2NjYuYmluZFBvcHVwKHBvcHVwXzFjOWVjMDE5MWFjMjQyMTBhODE5ODAxNDIzYmI4NmRjKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MTExZTk5Y2I4MDY0NDg0YTZkYjg4MmQ3ZDU1YmZhNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjk2NTYsIC03OS42MTU4MTg5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMzU0YmM2ODFiMDNmNDg1NDk0MDI0MGJjMjU2MmViZjQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2MzZjU5ZWE3ODdhNjQzYWY4NjYzNzZiYzViMGZkZmRhID0gJChgPGRpdiBpZD0iaHRtbF9jM2Y1OWVhNzg3YTY0M2FmODY2Mzc2YmM1YjBmZGZkYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2FuYWRhIFBvc3QgR2F0ZXdheSBQcm9jZXNzaW5nIENlbnRyZSwgTWlzc2lzc2F1Z2E8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMzU0YmM2ODFiMDNmNDg1NDk0MDI0MGJjMjU2MmViZjQuc2V0Q29udGVudChodG1sX2MzZjU5ZWE3ODdhNjQzYWY4NjYzNzZiYzViMGZkZmRhKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl82MTExZTk5Y2I4MDY0NDg0YTZkYjg4MmQ3ZDU1YmZhNy5iaW5kUG9wdXAocG9wdXBfMzU0YmM2ODFiMDNmNDg1NDk0MDI0MGJjMjU2MmViZjQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAwZmFhNjRkZGQwYTQyN2M4ZTdhNzRlMDc1MDE3ODY0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg4OTA1NCwgLTc5LjU1NDcyNDQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82MGE3MjM5YzA0MzM0ZTY2YTYyNjViM2NhZWI3NDA1MSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZDk4NzEyMmU2ZTE4NGMyZDkyZTY4NDUxMGQ1ZmE3NWMgPSAkKGA8ZGl2IGlkPSJodG1sX2Q5ODcxMjJlNmUxODRjMmQ5MmU2ODQ1MTBkNWZhNzVjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nc3ZpZXcgVmlsbGFnZSwgU3QuIFBoaWxsaXBzLCBNYXJ0aW4gR3JvdmUgR2FyZGVucywgUmljaHZpZXcgR2FyZGVucywgRXRvYmljb2tlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzYwYTcyMzljMDQzMzRlNjZhNjI2NWIzY2FlYjc0MDUxLnNldENvbnRlbnQoaHRtbF9kOTg3MTIyZTZlMTg0YzJkOTJlNjg0NTEwZDVmYTc1Yyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMDBmYWE2NGRkZDBhNDI3YzhlN2E3NGUwNzUwMTc4NjQuYmluZFBvcHVwKHBvcHVwXzYwYTcyMzljMDQzMzRlNjZhNjI2NWIzY2FlYjc0MDUxKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMTU4M2Q3MjQyOWU0YTVjOTU5MGFlZTc2MmE3ODZkNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc5NDIwMDMsIC03OS4yNjIwMjk0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZjhiZDg4NjNjOTZmNGNmNmJmOGZjY2YwNGZjYjM5OTQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzAzMjFkY2NlZDM4MjQyZDlhYzg1YmEzYjE2ZDAyOWEyID0gJChgPGRpdiBpZD0iaHRtbF8wMzIxZGNjZWQzODI0MmQ5YWM4NWJhM2IxNmQwMjlhMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QWdpbmNvdXJ0LCBTY2FyYm9yb3VnaDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9mOGJkODg2M2M5NmY0Y2Y2YmY4ZmNjZjA0ZmNiMzk5NC5zZXRDb250ZW50KGh0bWxfMDMyMWRjY2VkMzgyNDJkOWFjODViYTNiMTZkMDI5YTIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzExNTgzZDcyNDI5ZTRhNWM5NTkwYWVlNzYyYTc4NmQ2LmJpbmRQb3B1cChwb3B1cF9mOGJkODg2M2M5NmY0Y2Y2YmY4ZmNjZjA0ZmNiMzk5NCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODkxNTEzZWVlYzQ0NDhmOWJkNTY3MGI4MmVhMjkxMzIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDQzMjQ0LCAtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYjIyMjEzMDI4YjZjNDc4ZDhkN2UzYWIwOGVmMWI1NjYgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzc4NjBmZTU1OWI4MjQ3MGM5MGJjNjE5YjgyYjFhMzAzID0gJChgPGRpdiBpZD0iaHRtbF83ODYwZmU1NTliODI0NzBjOTBiYzYxOWI4MmIxYTMwMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2IyMjIxMzAyOGI2YzQ3OGQ4ZDdlM2FiMDhlZjFiNTY2LnNldENvbnRlbnQoaHRtbF83ODYwZmU1NTliODI0NzBjOTBiYzYxOWI4MmIxYTMwMyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfODkxNTEzZWVlYzQ0NDhmOWJkNTY3MGI4MmVhMjkxMzIuYmluZFBvcHVwKHBvcHVwX2IyMjIxMzAyOGI2YzQ3OGQ4ZDdlM2FiMDhlZjFiNTY2KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MTYyNDM0OThiNTI0MTkyYTcxOTYzM2MzNGY0MTI5YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjY5NTYsIC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82MzNlMzI4M2FmMDI0YzNhOGVjMTg3Y2NjNWI2YWVhMiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMmIzNzExZWZiNTBlNDNhMjgyMTIzMDA0OTUxYzE2OTYgPSAkKGA8ZGl2IGlkPSJodG1sXzJiMzcxMWVmYjUwZTQzYTI4MjEyMzAwNDk1MWMxNjk2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIFRvcm9udG8sIEhhcmJvcmQsIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNjMzZTMyODNhZjAyNGMzYThlYzE4N2NjYzViNmFlYTIuc2V0Q29udGVudChodG1sXzJiMzcxMWVmYjUwZTQzYTI4MjEyMzAwNDk1MWMxNjk2KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl85MTYyNDM0OThiNTI0MTkyYTcxOTYzM2MzNGY0MTI5Yy5iaW5kUG9wdXAocG9wdXBfNjMzZTMyODNhZjAyNGMzYThlYzE4N2NjYzViNmFlYTIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRkOTY5OGNkZjVhMzRhNzg5YjYwZjU1NTVkNjAzNTUxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNTcwNiwgLTc5LjQ4NDQ0OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzBhMDdjZjA0ODU2YjRhMmM5YjNiNjFjMmQ1NmU2MDE5ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hNzM4YzYxMTA5MzY0MTdjODgzNTU3YjExOTMwNDVjNyA9ICQoYDxkaXYgaWQ9Imh0bWxfYTczOGM2MTEwOTM2NDE3Yzg4MzU1N2IxMTkzMDQ1YzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJ1bm55bWVkZSwgU3dhbnNlYSwgV2VzdCBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzBhMDdjZjA0ODU2YjRhMmM5YjNiNjFjMmQ1NmU2MDE5LnNldENvbnRlbnQoaHRtbF9hNzM4YzYxMTA5MzY0MTdjODgzNTU3YjExOTMwNDVjNyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNGQ5Njk4Y2RmNWEzNGE3ODliNjBmNTU1NWQ2MDM1NTEuYmluZFBvcHVwKHBvcHVwXzBhMDdjZjA0ODU2YjRhMmM5YjNiNjFjMmQ1NmU2MDE5KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ODA3MjlmOGQ2ODc0MDFlYTcxNmY1MjQ1YzQ2OGMxYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4MTYzNzUsIC03OS4zMDQzMDIxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iYzYxMzg0YjQzMDU0MGFlOGQ1M2Q5NWUzZGIyNTE4YSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfOGI1NGRmYmJlNjk4NDc3YzgzZWVlMGY2ZWQ3NTg4ZGEgPSAkKGA8ZGl2IGlkPSJodG1sXzhiNTRkZmJiZTY5ODQ3N2M4M2VlZTBmNmVkNzU4OGRhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DbGFya3MgQ29ybmVycywgVGFtIE8mIzM5O1NoYW50ZXIsIFN1bGxpdmFuLCBTY2FyYm9yb3VnaDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9iYzYxMzg0YjQzMDU0MGFlOGQ1M2Q5NWUzZGIyNTE4YS5zZXRDb250ZW50KGh0bWxfOGI1NGRmYmJlNjk4NDc3YzgzZWVlMGY2ZWQ3NTg4ZGEpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzk4MDcyOWY4ZDY4NzQwMWVhNzE2ZjUyNDVjNDY4YzFiLmJpbmRQb3B1cChwb3B1cF9iYzYxMzg0YjQzMDU0MGFlOGQ1M2Q5NWUzZGIyNTE4YSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGE1MWRkNDJiZDljNGNmMWE3Y2E0ZmIyMjYyYTUyMzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLCAtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzY5YWU3ZDMzYmU5OTRjZTM4MTZiMjA3OWYxOWEzOTAyID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8xNWYzYzc3NGUzNjQ0ZjA5OWZiOTRlZDIwMDRmYWMyMSA9ICQoYDxkaXYgaWQ9Imh0bWxfMTVmM2M3NzRlMzY0NGYwOTlmYjk0ZWQyMDA0ZmFjMjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vb3JlIFBhcmssIFN1bW1lcmhpbGwgRWFzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzY5YWU3ZDMzYmU5OTRjZTM4MTZiMjA3OWYxOWEzOTAyLnNldENvbnRlbnQoaHRtbF8xNWYzYzc3NGUzNjQ0ZjA5OWZiOTRlZDIwMDRmYWMyMSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfOGE1MWRkNDJiZDljNGNmMWE3Y2E0ZmIyMjYyYTUyMzMuYmluZFBvcHVwKHBvcHVwXzY5YWU3ZDMzYmU5OTRjZTM4MTZiMjA3OWYxOWEzOTAyKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82NWMwNTFjOGY4NDY0NGM5YTYwMjI3NmY1ZTY2ODY2MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzIwNTcsIC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hZGQxNzcwZjVlNGE0ZDFiOTA5NDFmYzliNzhmNzdmZCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYzUzNGQ0YTNjZTgyNDhkNzliOTFkMDY0OTQ3ZWQwODYgPSAkKGA8ZGl2IGlkPSJodG1sX2M1MzRkNGEzY2U4MjQ4ZDc5YjkxZDA2NDk0N2VkMDg2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZW5zaW5ndG9uIE1hcmtldCwgQ2hpbmF0b3duLCBHcmFuZ2UgUGFyaywgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9hZGQxNzcwZjVlNGE0ZDFiOTA5NDFmYzliNzhmNzdmZC5zZXRDb250ZW50KGh0bWxfYzUzNGQ0YTNjZTgyNDhkNzliOTFkMDY0OTQ3ZWQwODYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzY1YzA1MWM4Zjg0NjQ0YzlhNjAyMjc2ZjVlNjY4NjYyLmJpbmRQb3B1cChwb3B1cF9hZGQxNzcwZjVlNGE0ZDFiOTA5NDFmYzliNzhmNzdmZCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGVmNGU4ZDNmMmNjNDNjMzk4YzQxNGRjNGIzZTk0ZWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My44MTUyNTIyLCAtNzkuMjg0NTc3Ml0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzcyODQyN2YxYTM0NGRhM2IxMjljYmIwZDYyYzM2YTYgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzE0ZDBlOWVkM2I3NDQ0MmY5MWJlZTY0MzVjMTI2ZTdjID0gJChgPGRpdiBpZD0iaHRtbF8xNGQwZTllZDNiNzQ0NDJmOTFiZWU2NDM1YzEyNmU3YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWlsbGlrZW4sIEFnaW5jb3VydCBOb3J0aCwgU3RlZWxlcyBFYXN0LCBMJiMzOTtBbW9yZWF1eCBFYXN0LCBTY2FyYm9yb3VnaDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9jNzI4NDI3ZjFhMzQ0ZGEzYjEyOWNiYjBkNjJjMzZhNi5zZXRDb250ZW50KGh0bWxfMTRkMGU5ZWQzYjc0NDQyZjkxYmVlNjQzNWMxMjZlN2MpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzRlZjRlOGQzZjJjYzQzYzM5OGM0MTRkYzRiM2U5NGVkLmJpbmRQb3B1cChwb3B1cF9jNzI4NDI3ZjFhMzQ0ZGEzYjEyOWNiYjBkNjJjMzZhNikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjlhNDY5MTM3NzUyNDllZjgxNDdmNmE2ZTlkYTAyODUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODY0MTIyOTk5OTk5OSwgLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzlhMzk2YTI1MDUxOTQ2MjQ5ZjJmMzk1NjFhZmU1NThiID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF82OGU3NzQ2NDgzNDQ0ODhkYWRiNmU3MWFhYzIwOTc1NyA9ICQoYDxkaXYgaWQ9Imh0bWxfNjhlNzc0NjQ4MzQ0NDg4ZGFkYjZlNzFhYWMyMDk3NTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN1bW1lcmhpbGwgV2VzdCwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBGb3Jlc3QgSGlsbCBTRSwgRGVlciBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfOWEzOTZhMjUwNTE5NDYyNDlmMmYzOTU2MWFmZTU1OGIuc2V0Q29udGVudChodG1sXzY4ZTc3NDY0ODM0NDQ4OGRhZGI2ZTcxYWFjMjA5NzU3KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9iOWE0NjkxMzc3NTI0OWVmODE0N2Y2YTZlOWRhMDI4NS5iaW5kUG9wdXAocG9wdXBfOWEzOTZhMjUwNTE5NDYyNDlmMmYzOTU2MWFmZTU1OGIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJlOWU3Njk3ZGRhYzRhNWJhYWNiZTQ5ZDEyNjg3ODBhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4OTQ2NywgLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzg2NDI2OTdmOWJlZDRiM2Q5NmJkMWJiN2NjNGY5NmU0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9iY2E3ZjMxZjQ3Y2M0ODA1ODRlZjVlNGY3YTQ2YTUyNiA9ICQoYDxkaXYgaWQ9Imh0bWxfYmNhN2YzMWY0N2NjNDgwNTg0ZWY1ZTRmN2E0NmE1MjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNOIFRvd2VyLCBLaW5nIGFuZCBTcGFkaW5hLCBSYWlsd2F5IExhbmRzLCBIYXJib3VyZnJvbnQgV2VzdCwgQmF0aHVyc3QgUXVheSwgU291dGggTmlhZ2FyYSwgSXNsYW5kIGFpcnBvcnQsIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfODY0MjY5N2Y5YmVkNGIzZDk2YmQxYmI3Y2M0Zjk2ZTQuc2V0Q29udGVudChodG1sX2JjYTdmMzFmNDdjYzQ4MDU4NGVmNWU0ZjdhNDZhNTI2KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8yZTllNzY5N2RkYWM0YTViYWFjYmU0OWQxMjY4NzgwYS5iaW5kUG9wdXAocG9wdXBfODY0MjY5N2Y5YmVkNGIzZDk2YmQxYmI3Y2M0Zjk2ZTQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVhNzMwZmI0YjllNTQ2MmViZGM0MzE2MWFiNzIzODRjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjA1NjQ2NiwgLTc5LjUwMTMyMDcwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9mNGQxMWUzOGFmZDg0MzZlYTU3OWE4ZWE4MGI1ZmVhZSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfY2Y0MjgxOWUwZjYwNDFmZTljMDVhMWM2YWZlZjRkM2QgPSAkKGA8ZGl2IGlkPSJodG1sX2NmNDI4MTllMGY2MDQxZmU5YzA1YTFjNmFmZWY0ZDNkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OZXcgVG9yb250bywgTWltaWNvIFNvdXRoLCBIdW1iZXIgQmF5IFNob3JlcywgRXRvYmljb2tlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2Y0ZDExZTM4YWZkODQzNmVhNTc5YThlYTgwYjVmZWFlLnNldENvbnRlbnQoaHRtbF9jZjQyODE5ZTBmNjA0MWZlOWMwNWExYzZhZmVmNGQzZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNWE3MzBmYjRiOWU1NDYyZWJkYzQzMTYxYWI3MjM4NGMuYmluZFBvcHVwKHBvcHVwX2Y0ZDExZTM4YWZkODQzNmVhNTc5YThlYTgwYjVmZWFlKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jM2ZmODExYTJmNjA0NjY0OGU3ZTYwYzI1NGVjMmQzYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTQxNjM5OTk5OTk5NiwgLTc5LjU4ODQzNjldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzAwMTJhOTUyYmVhZDQ2NGZiZmU5NThmYTE2ZTUwZmIxID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kN2QxNmIwNTYzZmE0MTlmYjRjOGIzMmI1YWE3YmRjNyA9ICQoYDxkaXYgaWQ9Imh0bWxfZDdkMTZiMDU2M2ZhNDE5ZmI0YzhiMzJiNWFhN2JkYzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNvdXRoIFN0ZWVsZXMsIFNpbHZlcnN0b25lLCBIdW1iZXJnYXRlLCBKYW1lc3Rvd24sIE1vdW50IE9saXZlLCBCZWF1bW9uZCBIZWlnaHRzLCBUaGlzdGxldG93biwgQWxiaW9uIEdhcmRlbnMsIEV0b2JpY29rZTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8wMDEyYTk1MmJlYWQ0NjRmYmZlOTU4ZmExNmU1MGZiMS5zZXRDb250ZW50KGh0bWxfZDdkMTZiMDU2M2ZhNDE5ZmI0YzhiMzJiNWFhN2JkYzcpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2MzZmY4MTFhMmY2MDQ2NjQ4ZTdlNjBjMjU0ZWMyZDNjLmJpbmRQb3B1cChwb3B1cF8wMDEyYTk1MmJlYWQ0NjRmYmZlOTU4ZmExNmU1MGZiMSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTBjMWQ4ODVkYjQ1NDdhMmIxYTViNzQ4NzNkNzhkZjAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43OTk1MjUyMDAwMDAwMDUsIC03OS4zMTgzODg3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9lNmJlMWJkZjY4NmY0ZjlkOWViZDM3NzU3N2I2MWFiOCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYTg5YWI5Y2JjMzZkNDUxYmI2YTA3Y2VhNzg4MTE4NmIgPSAkKGA8ZGl2IGlkPSJodG1sX2E4OWFiOWNiYzM2ZDQ1MWJiNmEwN2NlYTc4ODExODZiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdGVlbGVzIFdlc3QsIEwmIzM5O0Ftb3JlYXV4IFdlc3QsIFNjYXJib3JvdWdoPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2U2YmUxYmRmNjg2ZjRmOWQ5ZWJkMzc3NTc3YjYxYWI4LnNldENvbnRlbnQoaHRtbF9hODlhYjljYmMzNmQ0NTFiYjZhMDdjZWE3ODgxMTg2Yik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfZTBjMWQ4ODVkYjQ1NDdhMmIxYTViNzQ4NzNkNzhkZjAuYmluZFBvcHVwKHBvcHVwX2U2YmUxYmRmNjg2ZjRmOWQ5ZWJkMzc3NTc3YjYxYWI4KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NmRjMWNkZWM4MmM0NmVkYWNjNGQ0ZWMxODAyYjFhZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU2MjYsIC03OS4zNzc1Mjk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMGRhOWFjYWUzZjFmNDI1MGFkMzgxYmZkOWEzZTIwMmUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2YxMmQyOTE2MjQ1NTRiZjRiNGYwYWJlYWI2NTE1MjM4ID0gJChgPGRpdiBpZD0iaHRtbF9mMTJkMjkxNjI0NTU0YmY0YjRmMGFiZWFiNjUxNTIzOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWRhbGUsIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMGRhOWFjYWUzZjFmNDI1MGFkMzgxYmZkOWEzZTIwMmUuc2V0Q29udGVudChodG1sX2YxMmQyOTE2MjQ1NTRiZjRiNGYwYWJlYWI2NTE1MjM4KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl83NmRjMWNkZWM4MmM0NmVkYWNjNGQ0ZWMxODAyYjFhZS5iaW5kUG9wdXAocG9wdXBfMGRhOWFjYWUzZjFmNDI1MGFkMzgxYmZkOWEzZTIwMmUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE2OTY0MTQwZjAzYjQ4NzRiMjhjMjJjNjU5YjNiMWQ2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwgLTc5LjM3NDg0NTk5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9lNmQ5M2ZlMzg0ZjU0ZDk5YjllMDg0ZDgwZDc3ZmM1NyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMWFjNmVmN2E1YjNhNGFjZGEwZmE3NTM2NDVjNzczZDcgPSAkKGA8ZGl2IGlkPSJodG1sXzFhYzZlZjdhNWIzYTRhY2RhMGZhNzUzNjQ1Yzc3M2Q3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdG4gQSBQTyBCb3hlcywgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9lNmQ5M2ZlMzg0ZjU0ZDk5YjllMDg0ZDgwZDc3ZmM1Ny5zZXRDb250ZW50KGh0bWxfMWFjNmVmN2E1YjNhNGFjZGEwZmE3NTM2NDVjNzczZDcpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzE2OTY0MTQwZjAzYjQ4NzRiMjhjMjJjNjU5YjNiMWQ2LmJpbmRQb3B1cChwb3B1cF9lNmQ5M2ZlMzg0ZjU0ZDk5YjllMDg0ZDgwZDc3ZmM1NykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmZmZGI0MzUyODRhNGNmNGI5M2VjMWFjZmZjMTQwOGMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MDI0MTM3MDAwMDAwMSwgLTc5LjU0MzQ4NDA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8xMWExMjIwYjZkYzM0YjMxYTYxZTBmYjJlM2ViOWRiNSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMjU1MzkzNTBlY2I1NDY0NDlmZDA0OGI4Nzc3YWY5YWIgPSAkKGA8ZGl2IGlkPSJodG1sXzI1NTM5MzUwZWNiNTQ2NDQ5ZmQwNDhiODc3N2FmOWFiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BbGRlcndvb2QsIExvbmcgQnJhbmNoLCBFdG9iaWNva2U8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMTFhMTIyMGI2ZGMzNGIzMWE2MWUwZmIyZTNlYjlkYjUuc2V0Q29udGVudChodG1sXzI1NTM5MzUwZWNiNTQ2NDQ5ZmQwNDhiODc3N2FmOWFiKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl82ZmZkYjQzNTI4NGE0Y2Y0YjkzZWMxYWNmZmMxNDA4Yy5iaW5kUG9wdXAocG9wdXBfMTFhMTIyMGI2ZGMzNGIzMWE2MWUwZmIyZTNlYjlkYjUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U0ZmRlOGIyYzUwYTQ0YjdhMjIxMDAwZGQyMjM5YjFkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2NzQ4Mjk5OTk5OTk0LCAtNzkuNTk0MDU0NF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMzg0ZGFhMmQ0YzQ3NGYyY2FjODlhOTI2MTY5Mzg0MDUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzFiYzk4YmI2N2U5MzRiMzQ4ZTU1MTM3ZjhiYzBmYmViID0gJChgPGRpdiBpZD0iaHRtbF8xYmM5OGJiNjdlOTM0YjM0OGU1NTEzN2Y4YmMwZmJlYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGh3ZXN0LCBXZXN0IEh1bWJlciAtIENsYWlydmlsbGUsIEV0b2JpY29rZTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8zODRkYWEyZDRjNDc0ZjJjYWM4OWE5MjYxNjkzODQwNS5zZXRDb250ZW50KGh0bWxfMWJjOThiYjY3ZTkzNGIzNDhlNTUxMzdmOGJjMGZiZWIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2U0ZmRlOGIyYzUwYTQ0YjdhMjIxMDAwZGQyMjM5YjFkLmJpbmRQb3B1cChwb3B1cF8zODRkYWEyZDRjNDc0ZjJjYWM4OWE5MjYxNjkzODQwNSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2Q0MzM0NmVlMjczNGIzNzkwODA0Y2M2ZjYwZjc5NjMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My44MzYxMjQ3MDAwMDAwMDYsIC03OS4yMDU2MzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYWNmOGEwZDIwZWM4NGEzNThlZTU0YTdmMWFhNWVmYzIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzdkOGIyZDc0MDlkNTRkNTM5MmQzNGZiZjcxNDVmMzk2ID0gJChgPGRpdiBpZD0iaHRtbF83ZDhiMmQ3NDA5ZDU0ZDUzOTJkMzRmYmY3MTQ1ZjM5NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VXBwZXIgUm91Z2UsIFNjYXJib3JvdWdoPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2FjZjhhMGQyMGVjODRhMzU4ZWU1NGE3ZjFhYTVlZmMyLnNldENvbnRlbnQoaHRtbF83ZDhiMmQ3NDA5ZDU0ZDUzOTJkMzRmYmY3MTQ1ZjM5Nik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfN2Q0MzM0NmVlMjczNGIzNzkwODA0Y2M2ZjYwZjc5NjMuYmluZFBvcHVwKHBvcHVwX2FjZjhhMGQyMGVjODRhMzU4ZWU1NGE3ZjFhYTVlZmMyKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iODc3YjVlOTMwMmY0Yjg5OTIyZmJkMzIyNGUyMjMwNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywgLTc5LjM2NzY3NTNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzRiMTM0OWQ4ZjgyYzQ2NDQ5YTdkZDUzNDMwYzU2MzYwID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hNjYwODMzMWNiOTM0NmM1OTFmMDg1Y2RlMzA0ZmIyNyA9ICQoYDxkaXYgaWQ9Imh0bWxfYTY2MDgzMzFjYjkzNDZjNTkxZjA4NWNkZTMwNGZiMjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duLCBDYWJiYWdldG93biwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF80YjEzNDlkOGY4MmM0NjQ0OWE3ZGQ1MzQzMGM1NjM2MC5zZXRDb250ZW50KGh0bWxfYTY2MDgzMzFjYjkzNDZjNTkxZjA4NWNkZTMwNGZiMjcpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2I4NzdiNWU5MzAyZjRiODk5MjJmYmQzMjI0ZTIyMzA2LmJpbmRQb3B1cChwb3B1cF80YjEzNDlkOGY4MmM0NjQ0OWE3ZGQ1MzQzMGM1NjM2MCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTU1MGE1MDhkMzU3NDMzY2FkNTA0OGNlN2IzZGMyYmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg0MjkyLCAtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfOGQwOGZhYWE2YjU1NGIwY2EzMDY2Y2JjZmVmNjQzZDAgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzljMDhkODQ2NTU1ZjQ5M2RiMzk3NGFjMDQxZTRhZTc3ID0gJChgPGRpdiBpZD0iaHRtbF85YzA4ZDg0NjU1NWY0OTNkYjM5NzRhYzA0MWU0YWU3NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rmlyc3QgQ2FuYWRpYW4gUGxhY2UsIFVuZGVyZ3JvdW5kIGNpdHksIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfOGQwOGZhYWE2YjU1NGIwY2EzMDY2Y2JjZmVmNjQzZDAuc2V0Q29udGVudChodG1sXzljMDhkODQ2NTU1ZjQ5M2RiMzk3NGFjMDQxZTRhZTc3KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9hNTUwYTUwOGQzNTc0MzNjYWQ1MDQ4Y2U3YjNkYzJiYy5iaW5kUG9wdXAocG9wdXBfOGQwOGZhYWE2YjU1NGIwY2EzMDY2Y2JjZmVmNjQzZDApCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIzZmNjNWU5OTM0NDQ5ODBhMzU1YmE0NDA4ZTRiNTAxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzNjUzNjAwMDAwMDA1LCAtNzkuNTA2OTQzNl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzA1NWQxZDEyZjIwNGY4Nzg4ZmM4YjdjN2E4ZGEwODMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2FiYTIxZDg1ZTdiZDQ5Yzk5Mzg1MWVlYzNkMTQxODc2ID0gJChgPGRpdiBpZD0iaHRtbF9hYmEyMWQ4NWU3YmQ0OWM5OTM4NTFlZWMzZDE0MTg3NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEtpbmdzd2F5LCBNb250Z29tZXJ5IFJvYWQsIE9sZCBNaWxsIE5vcnRoLCBFdG9iaWNva2U8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYzA1NWQxZDEyZjIwNGY4Nzg4ZmM4YjdjN2E4ZGEwODMuc2V0Q29udGVudChodG1sX2FiYTIxZDg1ZTdiZDQ5Yzk5Mzg1MWVlYzNkMTQxODc2KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8yM2ZjYzVlOTkzNDQ0OTgwYTM1NWJhNDQwOGU0YjUwMS5iaW5kUG9wdXAocG9wdXBfYzA1NWQxZDEyZjIwNGY4Nzg4ZmM4YjdjN2E4ZGEwODMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VlMjYxZTIxZGExYzQyOTliNjNjYjBiZjNjOGYzNzI0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY1ODU5OSwgLTc5LjM4MzE1OTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9lNmE4NGMxY2Y0YmI0MDA0YWM2MDNmMmExMGYyNjM4NCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfOTczMjIwODhiYjJjNGM2Mjg3N2E1ZDU0ODgyNjhkYjQgPSAkKGA8ZGl2IGlkPSJodG1sXzk3MzIyMDg4YmIyYzRjNjI4NzdhNWQ1NDg4MjY4ZGI0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaHVyY2ggYW5kIFdlbGxlc2xleSwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9lNmE4NGMxY2Y0YmI0MDA0YWM2MDNmMmExMGYyNjM4NC5zZXRDb250ZW50KGh0bWxfOTczMjIwODhiYjJjNGM2Mjg3N2E1ZDU0ODgyNjhkYjQpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2VlMjYxZTIxZGExYzQyOTliNjNjYjBiZjNjOGYzNzI0LmJpbmRQb3B1cChwb3B1cF9lNmE4NGMxY2Y0YmI0MDA0YWM2MDNmMmExMGYyNjM4NCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWZkNTBjMmE0NWU2NDRlM2JiMGEwZGY4ZDc3ZTQ4NWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI3NDM5LCAtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJtaWRuaWdodGJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAibWFyb29uIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2RjYWRmZDE4YTFkZDQyMTZhZjA4MDcwMGI4NWIzZmNlKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8zNjJmYzgzMTFlYWQ0N2E4YmMyOTFlNWNiYjllNWYwYSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMmUzNmVhOGQyMTEyNDNhZDlmYTQ0ODU5YmNhZTVkMzcgPSAkKGA8ZGl2IGlkPSJodG1sXzJlMzZlYThkMjExMjQzYWQ5ZmE0NDg1OWJjYWU1ZDM3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CdXNpbmVzcyByZXBseSBtYWlsIFByb2Nlc3NpbmcgQ2VudHJlLCBTb3V0aCBDZW50cmFsIExldHRlciBQcm9jZXNzaW5nIFBsYW50IFRvcm9udG8sIEVhc3QgVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8zNjJmYzgzMTFlYWQ0N2E4YmMyOTFlNWNiYjllNWYwYS5zZXRDb250ZW50KGh0bWxfMmUzNmVhOGQyMTEyNDNhZDlmYTQ0ODU5YmNhZTVkMzcpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2FmZDUwYzJhNDVlNjQ0ZTNiYjBhMGRmOGQ3N2U0ODVmLmJpbmRQb3B1cChwb3B1cF8zNjJmYzgzMTFlYWQ0N2E4YmMyOTFlNWNiYjllNWYwYSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjg3NGY0YjliZmU1NGRiMTlhNDUzZTJjYTJjOTAzNDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzYyNTc5LCAtNzkuNDk4NTA5MDk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIm1pZG5pZ2h0Ymx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICJtYXJvb24iLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZGNhZGZkMThhMWRkNDIxNmFmMDgwNzAwYjg1YjNmY2UpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2QyMzdlMGE3N2M4NDRhMTdhNDJmZjA2Y2VjMmYxMjNlID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8xZGQ2ZmEyZGFjMzk0ODk2Yjc1ODMxMzZiMmVjMWM2ZSA9ICQoYDxkaXYgaWQ9Imh0bWxfMWRkNmZhMmRhYzM5NDg5NmI3NTgzMTM2YjJlYzFjNmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk9sZCBNaWxsIFNvdXRoLCBLaW5nJiMzOTtzIE1pbGwgUGFyaywgU3VubnlsZWEsIEh1bWJlciBCYXksIE1pbWljbyBORSwgVGhlIFF1ZWVuc3dheSBFYXN0LCBSb3lhbCBZb3JrIFNvdXRoIEVhc3QsIEtpbmdzd2F5IFBhcmsgU291dGggRWFzdCwgRXRvYmljb2tlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2QyMzdlMGE3N2M4NDRhMTdhNDJmZjA2Y2VjMmYxMjNlLnNldENvbnRlbnQoaHRtbF8xZGQ2ZmEyZGFjMzk0ODk2Yjc1ODMxMzZiMmVjMWM2ZSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMjg3NGY0YjliZmU1NGRiMTlhNDUzZTJjYTJjOTAzNDkuYmluZFBvcHVwKHBvcHVwX2QyMzdlMGE3N2M4NDRhMTdhNDJmZjA2Y2VjMmYxMjNlKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83OTdjMmM3ZDljYjA0MTA1YmM5ZGUwNzAzMzI5MDZkZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODg0MDgsIC03OS41MjA5OTk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAibWlkbmlnaHRibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIm1hcm9vbiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kY2FkZmQxOGExZGQ0MjE2YWYwODA3MDBiODViM2ZjZSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNzMzNDgzM2QzMWExNDNiZmIzMzAyZGM1YmVjMzc5NDQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzBiZTVjZDdiNzZlZjQ2MzRhNDIwZmU0MDE5ZWFkZWEzID0gJChgPGRpdiBpZD0iaHRtbF8wYmU1Y2Q3Yjc2ZWY0NjM0YTQyMGZlNDAxOWVhZGVhMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWltaWNvIE5XLCBUaGUgUXVlZW5zd2F5IFdlc3QsIFNvdXRoIG9mIEJsb29yLCBLaW5nc3dheSBQYXJrIFNvdXRoIFdlc3QsIFJveWFsIFlvcmsgU291dGggV2VzdCwgRXRvYmljb2tlPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzczMzQ4MzNkMzFhMTQzYmZiMzMwMmRjNWJlYzM3OTQ0LnNldENvbnRlbnQoaHRtbF8wYmU1Y2Q3Yjc2ZWY0NjM0YTQyMGZlNDAxOWVhZGVhMyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNzk3YzJjN2Q5Y2IwNDEwNWJjOWRlMDcwMzMyOTA2ZGQuYmluZFBvcHVwKHBvcHVwXzczMzQ4MzNkMzFhMTQzYmZiMzMwMmRjNWJlYzM3OTQ0KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKPC9zY3JpcHQ+ onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
CLIENT_ID = 'SECRET' # your Foursquare ID
CLIENT_SECRET = 'SECRET' # your Foursquare Secret
VERSION = '20200606' # Foursquare API version
```

Generate the function that makes a list with the name and coordinates of the nearby locations of each postal code (with limit of 200 locations for each neighborhood and a radius of 1km):


```python
LIMIT = 200
def getNearbyVenues(names, latitudes, longitudes, radius=1000):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        #print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        try:
            results = requests.get(url).json()["response"]['groups'][0]['items']
        except: continue
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Postal Code', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```


```python
toronto_venues = getNearbyVenues(names=df_t3["Postal Code"],
                                latitudes= df_t3["Latitude"],
                                longitudes=df_t3["Longitude"])
toronto_venues.shape
```




    (4881, 7)



In the following dataframe is presented the first 5 venues found with the **foursquare API**:


```python
print(toronto_venues.shape)
toronto_venues.head()

```

    (4881, 7)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>43.753259</td>
      <td>-79.329656</td>
      <td>Allwyn's Bakery</td>
      <td>43.759840</td>
      <td>-79.324719</td>
      <td>Caribbean Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M3A</td>
      <td>43.753259</td>
      <td>-79.329656</td>
      <td>Brookbanks Park</td>
      <td>43.751976</td>
      <td>-79.332140</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>43.753259</td>
      <td>-79.329656</td>
      <td>Tim Hortons</td>
      <td>43.760668</td>
      <td>-79.326368</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M3A</td>
      <td>43.753259</td>
      <td>-79.329656</td>
      <td>Bruno's valu-mart</td>
      <td>43.746143</td>
      <td>-79.324630</td>
      <td>Grocery Store</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M3A</td>
      <td>43.753259</td>
      <td>-79.329656</td>
      <td>High Street Fish &amp; Chips</td>
      <td>43.745260</td>
      <td>-79.324949</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))
```

    There are 331 uniques categories.
    

The following script generates a table that identifies each venue according to its category (there are 331 different categories):


```python
# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")
cols = list(toronto_onehot.columns)

# add neighborhood column back to dataframe
toronto_onehot['Postal Code'] = toronto_venues['Postal Code'] 

fixed_columns = [toronto_onehot.columns[-1]]+ list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]
```


```python
print(toronto_onehot.shape)
print(len(toronto_onehot["Postal Code"].unique()))
toronto_onehot.head()
```

    (4881, 332)
    101
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Accessories Store</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>Airport</th>
      <th>American Restaurant</th>
      <th>Amphitheater</th>
      <th>Animal Shelter</th>
      <th>Antique Shop</th>
      <th>Aquarium</th>
      <th>Art Gallery</th>
      <th>Art Museum</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Dealership</th>
      <th>Auto Garage</th>
      <th>Auto Workshop</th>
      <th>Automotive Shop</th>
      <th>BBQ Joint</th>
      <th>Baby Store</th>
      <th>Badminton Court</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Baseball Field</th>
      <th>Baseball Stadium</th>
      <th>Basketball Stadium</th>
      <th>Beach</th>
      <th>Beach Bar</th>
      <th>Beer Bar</th>
      <th>Beer Store</th>
      <th>Belgian Restaurant</th>
      <th>Bike Shop</th>
      <th>Bistro</th>
      <th>Bookstore</th>
      <th>Botanical Garden</th>
      <th>Boutique</th>
      <th>Bowling Alley</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>Brewery</th>
      <th>Bridal Shop</th>
      <th>Bridge</th>
      <th>Bubble Tea Shop</th>
      <th>Buffet</th>
      <th>Burger Joint</th>
      <th>Burrito Place</th>
      <th>Bus Line</th>
      <th>Bus Station</th>
      <th>Bus Stop</th>
      <th>Business Service</th>
      <th>Butcher</th>
      <th>Cafeteria</th>
      <th>Café</th>
      <th>Cajun / Creole Restaurant</th>
      <th>Camera Store</th>
      <th>Candy Store</th>
      <th>Cantonese Restaurant</th>
      <th>Caribbean Restaurant</th>
      <th>Castle</th>
      <th>Cemetery</th>
      <th>Cheese Shop</th>
      <th>Chinese Restaurant</th>
      <th>Chiropractor</th>
      <th>Chocolate Shop</th>
      <th>Church</th>
      <th>Churrascaria</th>
      <th>Climbing Gym</th>
      <th>Clothing Store</th>
      <th>Cocktail Bar</th>
      <th>Coffee Shop</th>
      <th>College Gym</th>
      <th>College Quad</th>
      <th>College Rec Center</th>
      <th>College Stadium</th>
      <th>College Theater</th>
      <th>Comedy Club</th>
      <th>Comfort Food Restaurant</th>
      <th>Comic Shop</th>
      <th>Community Center</th>
      <th>Concert Hall</th>
      <th>Construction &amp; Landscaping</th>
      <th>Convenience Store</th>
      <th>Cosmetics Shop</th>
      <th>Coworking Space</th>
      <th>Creperie</th>
      <th>Cuban Restaurant</th>
      <th>Cupcake Shop</th>
      <th>Curling Ice</th>
      <th>Dance Studio</th>
      <th>Deli / Bodega</th>
      <th>Dentist's Office</th>
      <th>Department Store</th>
      <th>Design Studio</th>
      <th>Dessert Shop</th>
      <th>Dim Sum Restaurant</th>
      <th>Diner</th>
      <th>Discount Store</th>
      <th>Distribution Center</th>
      <th>Dive Bar</th>
      <th>Dog Run</th>
      <th>Doner Restaurant</th>
      <th>Donut Shop</th>
      <th>Drugstore</th>
      <th>Dumpling Restaurant</th>
      <th>Eastern European Restaurant</th>
      <th>Electronics Store</th>
      <th>Ethiopian Restaurant</th>
      <th>Event Space</th>
      <th>Falafel Restaurant</th>
      <th>Farm</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Field</th>
      <th>Filipino Restaurant</th>
      <th>Fireworks Store</th>
      <th>Fish &amp; Chips Shop</th>
      <th>Fish Market</th>
      <th>Flea Market</th>
      <th>Flower Shop</th>
      <th>Food</th>
      <th>Food &amp; Drink Shop</th>
      <th>Food Court</th>
      <th>Food Truck</th>
      <th>Fountain</th>
      <th>Frame Store</th>
      <th>French Restaurant</th>
      <th>Fried Chicken Joint</th>
      <th>Frozen Yogurt Shop</th>
      <th>Fruit &amp; Vegetable Store</th>
      <th>Furniture / Home Store</th>
      <th>Gaming Cafe</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>Gas Station</th>
      <th>Gastropub</th>
      <th>Gay Bar</th>
      <th>General Entertainment</th>
      <th>General Travel</th>
      <th>German Restaurant</th>
      <th>Gift Shop</th>
      <th>Golf Course</th>
      <th>Golf Driving Range</th>
      <th>Gourmet Shop</th>
      <th>Greek Restaurant</th>
      <th>Grocery Store</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Gym Pool</th>
      <th>Hakka Restaurant</th>
      <th>Harbor / Marina</th>
      <th>Hardware Store</th>
      <th>Hawaiian Restaurant</th>
      <th>Health &amp; Beauty Service</th>
      <th>Health Food Store</th>
      <th>Historic Site</th>
      <th>History Museum</th>
      <th>Hobby Shop</th>
      <th>Hockey Arena</th>
      <th>Home Service</th>
      <th>Hookah Bar</th>
      <th>Hostel</th>
      <th>Hot Dog Joint</th>
      <th>Hotel</th>
      <th>Hotel Bar</th>
      <th>Hotpot Restaurant</th>
      <th>IT Services</th>
      <th>Ice Cream Shop</th>
      <th>Indian Chinese Restaurant</th>
      <th>Indian Restaurant</th>
      <th>Indie Movie Theater</th>
      <th>Indie Theater</th>
      <th>Indonesian Restaurant</th>
      <th>Intersection</th>
      <th>Irish Pub</th>
      <th>Italian Restaurant</th>
      <th>Japanese Restaurant</th>
      <th>Jazz Club</th>
      <th>Jewelry Store</th>
      <th>Jewish Restaurant</th>
      <th>Juice Bar</th>
      <th>Karaoke Bar</th>
      <th>Kitchen Supply Store</th>
      <th>Korean Restaurant</th>
      <th>Lake</th>
      <th>Latin American Restaurant</th>
      <th>Laundry Service</th>
      <th>Light Rail Station</th>
      <th>Lighting Store</th>
      <th>Lingerie Store</th>
      <th>Liquor Store</th>
      <th>Lounge</th>
      <th>Mac &amp; Cheese Joint</th>
      <th>Malay Restaurant</th>
      <th>Marijuana Dispensary</th>
      <th>Market</th>
      <th>Martial Arts Dojo</th>
      <th>Massage Studio</th>
      <th>Mattress Store</th>
      <th>Medical Center</th>
      <th>Mediterranean Restaurant</th>
      <th>Men's Store</th>
      <th>Metro Station</th>
      <th>Mexican Restaurant</th>
      <th>Middle Eastern Restaurant</th>
      <th>Miscellaneous Shop</th>
      <th>Mobile Phone Shop</th>
      <th>Modern European Restaurant</th>
      <th>Monument / Landmark</th>
      <th>Moroccan Restaurant</th>
      <th>Movie Theater</th>
      <th>Moving Target</th>
      <th>Museum</th>
      <th>Music School</th>
      <th>Music Store</th>
      <th>Music Venue</th>
      <th>Nail Salon</th>
      <th>Neighborhood</th>
      <th>New American Restaurant</th>
      <th>Nightclub</th>
      <th>Noodle House</th>
      <th>Office</th>
      <th>Optical Shop</th>
      <th>Organic Grocery</th>
      <th>Other Great Outdoors</th>
      <th>Other Repair Shop</th>
      <th>Pakistani Restaurant</th>
      <th>Paper / Office Supplies Store</th>
      <th>Park</th>
      <th>Pastry Shop</th>
      <th>Performing Arts Venue</th>
      <th>Persian Restaurant</th>
      <th>Pet Store</th>
      <th>Pharmacy</th>
      <th>Photography Lab</th>
      <th>Pide Place</th>
      <th>Pie Shop</th>
      <th>Pilates Studio</th>
      <th>Pizza Place</th>
      <th>Playground</th>
      <th>Plaza</th>
      <th>Poke Place</th>
      <th>Pool</th>
      <th>Pool Hall</th>
      <th>Portuguese Restaurant</th>
      <th>Poutine Place</th>
      <th>Print Shop</th>
      <th>Pub</th>
      <th>Ramen Restaurant</th>
      <th>Record Shop</th>
      <th>Recreation Center</th>
      <th>Rental Car Location</th>
      <th>Residential Building (Apartment / Condo)</th>
      <th>Restaurant</th>
      <th>River</th>
      <th>Road</th>
      <th>Rock Climbing Spot</th>
      <th>Rock Club</th>
      <th>Roof Deck</th>
      <th>Sake Bar</th>
      <th>Salad Place</th>
      <th>Salon / Barbershop</th>
      <th>Sandwich Place</th>
      <th>Scenic Lookout</th>
      <th>School</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shoe Store</th>
      <th>Shop &amp; Service</th>
      <th>Shopping Mall</th>
      <th>Shopping Plaza</th>
      <th>Skate Park</th>
      <th>Skating Rink</th>
      <th>Ski Area</th>
      <th>Ski Chalet</th>
      <th>Smoke Shop</th>
      <th>Smoothie Shop</th>
      <th>Snack Place</th>
      <th>Soccer Field</th>
      <th>Soccer Stadium</th>
      <th>Social Club</th>
      <th>Soup Place</th>
      <th>South American Restaurant</th>
      <th>Souvlaki Shop</th>
      <th>Spa</th>
      <th>Speakeasy</th>
      <th>Sporting Goods Shop</th>
      <th>Sports Bar</th>
      <th>Sports Club</th>
      <th>Sri Lankan Restaurant</th>
      <th>Stationery Store</th>
      <th>Steakhouse</th>
      <th>Storage Facility</th>
      <th>Street Art</th>
      <th>Supermarket</th>
      <th>Supplement Shop</th>
      <th>Sushi Restaurant</th>
      <th>Syrian Restaurant</th>
      <th>Taco Place</th>
      <th>Tailor Shop</th>
      <th>Taiwanese Restaurant</th>
      <th>Tanning Salon</th>
      <th>Tapas Restaurant</th>
      <th>Tea Room</th>
      <th>Tech Startup</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Theme Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Tibetan Restaurant</th>
      <th>Toy / Game Store</th>
      <th>Track</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Tree</th>
      <th>Turkish Restaurant</th>
      <th>University</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Video Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Warehouse Store</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M3A</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M3A</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M3A</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



This dataframe contains the mean of the frequency of occurrence of each category:


```python
toronto_g = toronto_onehot.groupby("Postal Code").mean().reset_index()
print(toronto_g.shape)
toronto_g.head()
```

    (101, 332)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Accessories Store</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>Airport</th>
      <th>American Restaurant</th>
      <th>Amphitheater</th>
      <th>Animal Shelter</th>
      <th>Antique Shop</th>
      <th>Aquarium</th>
      <th>Art Gallery</th>
      <th>Art Museum</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Dealership</th>
      <th>Auto Garage</th>
      <th>Auto Workshop</th>
      <th>Automotive Shop</th>
      <th>BBQ Joint</th>
      <th>Baby Store</th>
      <th>Badminton Court</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Baseball Field</th>
      <th>Baseball Stadium</th>
      <th>Basketball Stadium</th>
      <th>Beach</th>
      <th>Beach Bar</th>
      <th>Beer Bar</th>
      <th>Beer Store</th>
      <th>Belgian Restaurant</th>
      <th>Bike Shop</th>
      <th>Bistro</th>
      <th>Bookstore</th>
      <th>Botanical Garden</th>
      <th>Boutique</th>
      <th>Bowling Alley</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>Brewery</th>
      <th>Bridal Shop</th>
      <th>Bridge</th>
      <th>Bubble Tea Shop</th>
      <th>Buffet</th>
      <th>Burger Joint</th>
      <th>Burrito Place</th>
      <th>Bus Line</th>
      <th>Bus Station</th>
      <th>Bus Stop</th>
      <th>Business Service</th>
      <th>Butcher</th>
      <th>Cafeteria</th>
      <th>Café</th>
      <th>Cajun / Creole Restaurant</th>
      <th>Camera Store</th>
      <th>Candy Store</th>
      <th>Cantonese Restaurant</th>
      <th>Caribbean Restaurant</th>
      <th>Castle</th>
      <th>Cemetery</th>
      <th>Cheese Shop</th>
      <th>Chinese Restaurant</th>
      <th>Chiropractor</th>
      <th>Chocolate Shop</th>
      <th>Church</th>
      <th>Churrascaria</th>
      <th>Climbing Gym</th>
      <th>Clothing Store</th>
      <th>Cocktail Bar</th>
      <th>Coffee Shop</th>
      <th>College Gym</th>
      <th>College Quad</th>
      <th>College Rec Center</th>
      <th>College Stadium</th>
      <th>College Theater</th>
      <th>Comedy Club</th>
      <th>Comfort Food Restaurant</th>
      <th>Comic Shop</th>
      <th>Community Center</th>
      <th>Concert Hall</th>
      <th>Construction &amp; Landscaping</th>
      <th>Convenience Store</th>
      <th>Cosmetics Shop</th>
      <th>Coworking Space</th>
      <th>Creperie</th>
      <th>Cuban Restaurant</th>
      <th>Cupcake Shop</th>
      <th>Curling Ice</th>
      <th>Dance Studio</th>
      <th>Deli / Bodega</th>
      <th>Dentist's Office</th>
      <th>Department Store</th>
      <th>Design Studio</th>
      <th>Dessert Shop</th>
      <th>Dim Sum Restaurant</th>
      <th>Diner</th>
      <th>Discount Store</th>
      <th>Distribution Center</th>
      <th>Dive Bar</th>
      <th>Dog Run</th>
      <th>Doner Restaurant</th>
      <th>Donut Shop</th>
      <th>Drugstore</th>
      <th>Dumpling Restaurant</th>
      <th>Eastern European Restaurant</th>
      <th>Electronics Store</th>
      <th>Ethiopian Restaurant</th>
      <th>Event Space</th>
      <th>Falafel Restaurant</th>
      <th>Farm</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Field</th>
      <th>Filipino Restaurant</th>
      <th>Fireworks Store</th>
      <th>Fish &amp; Chips Shop</th>
      <th>Fish Market</th>
      <th>Flea Market</th>
      <th>Flower Shop</th>
      <th>Food</th>
      <th>Food &amp; Drink Shop</th>
      <th>Food Court</th>
      <th>Food Truck</th>
      <th>Fountain</th>
      <th>Frame Store</th>
      <th>French Restaurant</th>
      <th>Fried Chicken Joint</th>
      <th>Frozen Yogurt Shop</th>
      <th>Fruit &amp; Vegetable Store</th>
      <th>Furniture / Home Store</th>
      <th>Gaming Cafe</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>Gas Station</th>
      <th>Gastropub</th>
      <th>Gay Bar</th>
      <th>General Entertainment</th>
      <th>General Travel</th>
      <th>German Restaurant</th>
      <th>Gift Shop</th>
      <th>Golf Course</th>
      <th>Golf Driving Range</th>
      <th>Gourmet Shop</th>
      <th>Greek Restaurant</th>
      <th>Grocery Store</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Gym Pool</th>
      <th>Hakka Restaurant</th>
      <th>Harbor / Marina</th>
      <th>Hardware Store</th>
      <th>Hawaiian Restaurant</th>
      <th>Health &amp; Beauty Service</th>
      <th>Health Food Store</th>
      <th>Historic Site</th>
      <th>History Museum</th>
      <th>Hobby Shop</th>
      <th>Hockey Arena</th>
      <th>Home Service</th>
      <th>Hookah Bar</th>
      <th>Hostel</th>
      <th>Hot Dog Joint</th>
      <th>Hotel</th>
      <th>Hotel Bar</th>
      <th>Hotpot Restaurant</th>
      <th>IT Services</th>
      <th>Ice Cream Shop</th>
      <th>Indian Chinese Restaurant</th>
      <th>Indian Restaurant</th>
      <th>Indie Movie Theater</th>
      <th>Indie Theater</th>
      <th>Indonesian Restaurant</th>
      <th>Intersection</th>
      <th>Irish Pub</th>
      <th>Italian Restaurant</th>
      <th>Japanese Restaurant</th>
      <th>Jazz Club</th>
      <th>Jewelry Store</th>
      <th>Jewish Restaurant</th>
      <th>Juice Bar</th>
      <th>Karaoke Bar</th>
      <th>Kitchen Supply Store</th>
      <th>Korean Restaurant</th>
      <th>Lake</th>
      <th>Latin American Restaurant</th>
      <th>Laundry Service</th>
      <th>Light Rail Station</th>
      <th>Lighting Store</th>
      <th>Lingerie Store</th>
      <th>Liquor Store</th>
      <th>Lounge</th>
      <th>Mac &amp; Cheese Joint</th>
      <th>Malay Restaurant</th>
      <th>Marijuana Dispensary</th>
      <th>Market</th>
      <th>Martial Arts Dojo</th>
      <th>Massage Studio</th>
      <th>Mattress Store</th>
      <th>Medical Center</th>
      <th>Mediterranean Restaurant</th>
      <th>Men's Store</th>
      <th>Metro Station</th>
      <th>Mexican Restaurant</th>
      <th>Middle Eastern Restaurant</th>
      <th>Miscellaneous Shop</th>
      <th>Mobile Phone Shop</th>
      <th>Modern European Restaurant</th>
      <th>Monument / Landmark</th>
      <th>Moroccan Restaurant</th>
      <th>Movie Theater</th>
      <th>Moving Target</th>
      <th>Museum</th>
      <th>Music School</th>
      <th>Music Store</th>
      <th>Music Venue</th>
      <th>Nail Salon</th>
      <th>Neighborhood</th>
      <th>New American Restaurant</th>
      <th>Nightclub</th>
      <th>Noodle House</th>
      <th>Office</th>
      <th>Optical Shop</th>
      <th>Organic Grocery</th>
      <th>Other Great Outdoors</th>
      <th>Other Repair Shop</th>
      <th>Pakistani Restaurant</th>
      <th>Paper / Office Supplies Store</th>
      <th>Park</th>
      <th>Pastry Shop</th>
      <th>Performing Arts Venue</th>
      <th>Persian Restaurant</th>
      <th>Pet Store</th>
      <th>Pharmacy</th>
      <th>Photography Lab</th>
      <th>Pide Place</th>
      <th>Pie Shop</th>
      <th>Pilates Studio</th>
      <th>Pizza Place</th>
      <th>Playground</th>
      <th>Plaza</th>
      <th>Poke Place</th>
      <th>Pool</th>
      <th>Pool Hall</th>
      <th>Portuguese Restaurant</th>
      <th>Poutine Place</th>
      <th>Print Shop</th>
      <th>Pub</th>
      <th>Ramen Restaurant</th>
      <th>Record Shop</th>
      <th>Recreation Center</th>
      <th>Rental Car Location</th>
      <th>Residential Building (Apartment / Condo)</th>
      <th>Restaurant</th>
      <th>River</th>
      <th>Road</th>
      <th>Rock Climbing Spot</th>
      <th>Rock Club</th>
      <th>Roof Deck</th>
      <th>Sake Bar</th>
      <th>Salad Place</th>
      <th>Salon / Barbershop</th>
      <th>Sandwich Place</th>
      <th>Scenic Lookout</th>
      <th>School</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shoe Store</th>
      <th>Shop &amp; Service</th>
      <th>Shopping Mall</th>
      <th>Shopping Plaza</th>
      <th>Skate Park</th>
      <th>Skating Rink</th>
      <th>Ski Area</th>
      <th>Ski Chalet</th>
      <th>Smoke Shop</th>
      <th>Smoothie Shop</th>
      <th>Snack Place</th>
      <th>Soccer Field</th>
      <th>Soccer Stadium</th>
      <th>Social Club</th>
      <th>Soup Place</th>
      <th>South American Restaurant</th>
      <th>Souvlaki Shop</th>
      <th>Spa</th>
      <th>Speakeasy</th>
      <th>Sporting Goods Shop</th>
      <th>Sports Bar</th>
      <th>Sports Club</th>
      <th>Sri Lankan Restaurant</th>
      <th>Stationery Store</th>
      <th>Steakhouse</th>
      <th>Storage Facility</th>
      <th>Street Art</th>
      <th>Supermarket</th>
      <th>Supplement Shop</th>
      <th>Sushi Restaurant</th>
      <th>Syrian Restaurant</th>
      <th>Taco Place</th>
      <th>Tailor Shop</th>
      <th>Taiwanese Restaurant</th>
      <th>Tanning Salon</th>
      <th>Tapas Restaurant</th>
      <th>Tea Room</th>
      <th>Tech Startup</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Theme Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Tibetan Restaurant</th>
      <th>Toy / Game Store</th>
      <th>Track</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Tree</th>
      <th>Turkish Restaurant</th>
      <th>University</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Video Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Warehouse Store</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1C</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.200000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.200000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1E</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.040000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.080000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.080000</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.040000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.040000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.120000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.080000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.040000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1G</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.111111</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.222222</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.111111</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.111111</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.111111</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.222222</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.111111</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1H</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.096774</td>
      <td>0.064516</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.096774</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.064516</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.064516</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.064516</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.000000</td>
      <td>0.032258</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1J</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.076923</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.076923</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.076923</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.076923</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.076923</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.153846</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.076923</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.076923</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.076923</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.076923</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.076923</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Here are the top 5 most frequent venue categories for each postal code location:


```python
#num_top_venues = 5

#for hood in toronto_g['Postal Code']:
#    print("----"+hood+"----")
#    temp = toronto_g[toronto_g['Postal Code'] == hood].T.reset_index()
#    temp.columns = ['venue','freq']
#    temp = temp.iloc[1:]
#    temp['freq'] = temp['freq'].astype(float)
#    temp = temp.round({'freq': 2})
#    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
#    print('\n')
```


```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```

This dataframe shows the top 20 mosto common venue category for each neighborhood:


```python
num_top_venues = 20

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Postal Code']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Postal Code'] = toronto_g['Postal Code']

for ind in np.arange(toronto_g.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_g.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
      <th>11th Most Common Venue</th>
      <th>12th Most Common Venue</th>
      <th>13th Most Common Venue</th>
      <th>14th Most Common Venue</th>
      <th>15th Most Common Venue</th>
      <th>16th Most Common Venue</th>
      <th>17th Most Common Venue</th>
      <th>18th Most Common Venue</th>
      <th>19th Most Common Venue</th>
      <th>20th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1C</td>
      <td>Italian Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Playground</td>
      <td>Burger Joint</td>
      <td>Park</td>
      <td>Zoo</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Field</td>
      <td>Fast Food Restaurant</td>
      <td>Drugstore</td>
      <td>Filipino Restaurant</td>
      <td>Fireworks Store</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fish Market</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1E</td>
      <td>Pizza Place</td>
      <td>Coffee Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Restaurant</td>
      <td>Bank</td>
      <td>Filipino Restaurant</td>
      <td>Pharmacy</td>
      <td>Smoothie Shop</td>
      <td>Food &amp; Drink Shop</td>
      <td>Sports Bar</td>
      <td>Beer Store</td>
      <td>Liquor Store</td>
      <td>Sandwich Place</td>
      <td>Greek Restaurant</td>
      <td>Moving Target</td>
      <td>Chinese Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Discount Store</td>
      <td>Supermarket</td>
      <td>Ethiopian Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1G</td>
      <td>Park</td>
      <td>Coffee Shop</td>
      <td>Chinese Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Pharmacy</td>
      <td>Mobile Phone Shop</td>
      <td>Dance Studio</td>
      <td>Farmers Market</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Creperie</td>
      <td>Drugstore</td>
      <td>Field</td>
      <td>Filipino Restaurant</td>
      <td>Fireworks Store</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1H</td>
      <td>Bakery</td>
      <td>Coffee Shop</td>
      <td>Bank</td>
      <td>Indian Restaurant</td>
      <td>Gas Station</td>
      <td>Grocery Store</td>
      <td>Gym / Fitness Center</td>
      <td>Burger Joint</td>
      <td>Fast Food Restaurant</td>
      <td>Pizza Place</td>
      <td>Sporting Goods Shop</td>
      <td>Fried Chicken Joint</td>
      <td>Caribbean Restaurant</td>
      <td>Music Store</td>
      <td>Chinese Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Bus Line</td>
      <td>Wings Joint</td>
      <td>Athletics &amp; Sports</td>
      <td>Pharmacy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1J</td>
      <td>Ice Cream Shop</td>
      <td>Coffee Shop</td>
      <td>Train Station</td>
      <td>Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Bowling Alley</td>
      <td>Convenience Store</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Women's Store</td>
      <td>Grocery Store</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Drugstore</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Zoo</td>
      <td>Falafel Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



## 3. K-means to cluster Neighborhoods
The following scripts runs a *k-means* algorithm to cluster in five groups all the neighborhoods according to the frequency of the different categories of venues in each neighborhood:


```python
# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_g.drop('Postal Code', axis = 1)
toronto_grouped_clustering.head()
# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
check1 = list(kmeans.labels_)
len(check1)
```




    101




```python
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df_t3

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Postal Code'), on='Postal Code')
print(toronto_merged["Cluster Labels"].unique())
toronto_merged.head() # check the last columns!
toronto_merged.replace("nan", np.nan, inplace = True)
toronto_merged.dropna(inplace=True)
print(toronto_merged["Cluster Labels"].unique())
```

    [ 2.  3.  0. nan  1.  4.]
    [2. 3. 0. 1. 4.]
    

### Clustered Neighborhoods Map
This map shows the 5 different clusters of neighborhoods, indicating which neighborhoods are more similar between them:


```python
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=10)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
#print(colors_array)
rainbow = [colors.rgb2hex(i) for i in colors_array]
#print(rainbow)
# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    #print(cluster)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster)],
        fill=True,
        fill_color=rainbow[int(cluster)],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgCiAgICAgICAgPHNjcmlwdD4KICAgICAgICAgICAgTF9OT19UT1VDSCA9IGZhbHNlOwogICAgICAgICAgICBMX0RJU0FCTEVfM0QgPSBmYWxzZTsKICAgICAgICA8L3NjcmlwdD4KICAgIAogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjYuMC9kaXN0L2xlYWZsZXQuanMiPjwvc2NyaXB0PgogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY29kZS5qcXVlcnkuY29tL2pxdWVyeS0xLjEyLjQubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9qcy9ib290c3RyYXAubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5qcyI+PC9zY3JpcHQ+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjYuMC9kaXN0L2xlYWZsZXQuY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vYm9vdHN0cmFwLzMuMi4wL2Nzcy9ib290c3RyYXAubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLXRoZW1lLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9mb250LWF3ZXNvbWUvNC42LjMvY3NzL2ZvbnQtYXdlc29tZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuanMuY2xvdWRmbGFyZS5jb20vYWpheC9saWJzL0xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLzIuMC4yL2xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL3Jhd2Nkbi5naXRoYWNrLmNvbS9weXRob24tdmlzdWFsaXphdGlvbi9mb2xpdW0vbWFzdGVyL2ZvbGl1bS90ZW1wbGF0ZXMvbGVhZmxldC5hd2Vzb21lLnJvdGF0ZS5jc3MiLz4KICAgIDxzdHlsZT5odG1sLCBib2R5IHt3aWR0aDogMTAwJTtoZWlnaHQ6IDEwMCU7bWFyZ2luOiAwO3BhZGRpbmc6IDA7fTwvc3R5bGU+CiAgICA8c3R5bGU+I21hcCB7cG9zaXRpb246YWJzb2x1dGU7dG9wOjA7Ym90dG9tOjA7cmlnaHQ6MDtsZWZ0OjA7fTwvc3R5bGU+CiAgICAKICAgICAgICAgICAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwKICAgICAgICAgICAgICAgIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxlPTEuMCwgdXNlci1zY2FsYWJsZT1ubyIgLz4KICAgICAgICAgICAgPHN0eWxlPgogICAgICAgICAgICAgICAgI21hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1YyB7CiAgICAgICAgICAgICAgICAgICAgcG9zaXRpb246IHJlbGF0aXZlOwogICAgICAgICAgICAgICAgICAgIHdpZHRoOiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgbGVmdDogMC4wJTsKICAgICAgICAgICAgICAgICAgICB0b3A6IDAuMCU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIDwvc3R5bGU+CiAgICAgICAgCjwvaGVhZD4KPGJvZHk+ICAgIAogICAgCiAgICAgICAgICAgIDxkaXYgY2xhc3M9ImZvbGl1bS1tYXAiIGlkPSJtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMiID48L2Rpdj4KICAgICAgICAKPC9ib2R5Pgo8c2NyaXB0PiAgICAKICAgIAogICAgICAgICAgICB2YXIgbWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjID0gTC5tYXAoCiAgICAgICAgICAgICAgICAibWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjIiwKICAgICAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgICAgICBjZW50ZXI6IFs0My42NTM0ODE3LCAtNzkuMzgzOTM0N10sCiAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NywKICAgICAgICAgICAgICAgICAgICB6b29tOiAxMCwKICAgICAgICAgICAgICAgICAgICB6b29tQ29udHJvbDogdHJ1ZSwKICAgICAgICAgICAgICAgICAgICBwcmVmZXJDYW52YXM6IGZhbHNlLAogICAgICAgICAgICAgICAgfQogICAgICAgICAgICApOwoKICAgICAgICAgICAgCgogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzgxOWM1ZjU2ZjBjNzRjYjk4YTIwNjNiZjlmODE1MmI0ID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAiaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmciLAogICAgICAgICAgICAgICAgeyJhdHRyaWJ1dGlvbiI6ICJEYXRhIGJ5IFx1MDAyNmNvcHk7IFx1MDAzY2EgaHJlZj1cImh0dHA6Ly9vcGVuc3RyZWV0bWFwLm9yZ1wiXHUwMDNlT3BlblN0cmVldE1hcFx1MDAzYy9hXHUwMDNlLCB1bmRlciBcdTAwM2NhIGhyZWY9XCJodHRwOi8vd3d3Lm9wZW5zdHJlZXRtYXAub3JnL2NvcHlyaWdodFwiXHUwMDNlT0RiTFx1MDAzYy9hXHUwMDNlLiIsICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwgIm1heE5hdGl2ZVpvb20iOiAxOCwgIm1heFpvb20iOiAxOCwgIm1pblpvb20iOiAwLCAibm9XcmFwIjogZmFsc2UsICJvcGFjaXR5IjogMSwgInN1YmRvbWFpbnMiOiAiYWJjIiwgInRtcyI6IGZhbHNlfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmUzMjY2MmI2YWRhNDRkOWIxOWJkZGFiMmQyODdiOTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTMyNTg2LCAtNzkuMzI5NjU2NV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8xZmIyYmE0ZmE2Mjg0ZjQwYWY3ZmY3YzUzZjk5ZTZkYyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNmQ3Y2ZhY2EzZjk5NDI3ODhjOGU4MjMyYjM3OWY4MmMgPSAkKGA8ZGl2IGlkPSJodG1sXzZkN2NmYWNhM2Y5OTQyNzg4YzhlODIzMmIzNzlmODJjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrd29vZHMgQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMWZiMmJhNGZhNjI4NGY0MGFmN2ZmN2M1M2Y5OWU2ZGMuc2V0Q29udGVudChodG1sXzZkN2NmYWNhM2Y5OTQyNzg4YzhlODIzMmIzNzlmODJjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8yZTMyNjYyYjZhZGE0NGQ5YjE5YmRkYWIyZDI4N2I5MC5iaW5kUG9wdXAocG9wdXBfMWZiMmJhNGZhNjI4NGY0MGFmN2ZmN2M1M2Y5OWU2ZGMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FiMDc5Y2RmNWZlMjQzNWNhMTQyOGQ2NGVhZjgzNzBkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI1ODgyMjk5OTk5OTk1LCAtNzkuMzE1NTcxNTk5OTk5OThdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZmZjMWRkNjlhNTc3NGU4OWI5YWVkZmRmMWJjOGJkYWMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2I1ZDIyYTg0MmU2MDQ5OTdhMzJlMjlmODUyNWI1NGJjID0gJChgPGRpdiBpZD0iaHRtbF9iNWQyMmE4NDJlNjA0OTk3YTMyZTI5Zjg1MjViNTRiYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VmljdG9yaWEgVmlsbGFnZSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9mZmMxZGQ2OWE1Nzc0ZTg5YjlhZWRmZGYxYmM4YmRhYy5zZXRDb250ZW50KGh0bWxfYjVkMjJhODQyZTYwNDk5N2EzMmUyOWY4NTI1YjU0YmMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2FiMDc5Y2RmNWZlMjQzNWNhMTQyOGQ2NGVhZjgzNzBkLmJpbmRQb3B1cChwb3B1cF9mZmMxZGQ2OWE1Nzc0ZTg5YjlhZWRmZGYxYmM4YmRhYykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzg4NmQ2ZGQ5YTc0NDgxNWE1MTRiYWIyZTVjY2MxNDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTQyNTk5LCAtNzkuMzYwNjM1OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82NjYwNGQ5Y2E3MTg0YmJmYWMyODdiNjk5YTNiM2Q3OCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZmNjMDc4Yjk3ZGM3NGVlZThiZjBhYjE3NTMyYmRmMGIgPSAkKGA8ZGl2IGlkPSJodG1sX2ZjYzA3OGI5N2RjNzRlZWU4YmYwYWIxNzUzMmJkZjBiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SZWdlbnQgUGFyaywgSGFyYm91cmZyb250IENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzY2NjA0ZDljYTcxODRiYmZhYzI4N2I2OTlhM2IzZDc4LnNldENvbnRlbnQoaHRtbF9mY2MwNzhiOTdkYzc0ZWVlOGJmMGFiMTc1MzJiZGYwYik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYzg4NmQ2ZGQ5YTc0NDgxNWE1MTRiYWIyZTVjY2MxNDkuYmluZFBvcHVwKHBvcHVwXzY2NjA0ZDljYTcxODRiYmZhYzI4N2I2OTlhM2IzZDc4KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNzkyNzJkYTQyNDc0YmRiODNkNWFmMmMyZDczYmU3MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxODUxNzk5OTk5OTk5NiwgLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzA4MmU3NTIwZWNlMjQxOTA4NWVjZDg4ZmVkN2UxYjFkID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hZTM0NzI1NmZmY2Y0MzEzYWFiNDU1MTlhZmYyNTM4MCA9ICQoYDxkaXYgaWQ9Imh0bWxfYWUzNDcyNTZmZmNmNDMxM2FhYjQ1NTE5YWZmMjUzODAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxhd3JlbmNlIE1hbm9yLCBMYXdyZW5jZSBIZWlnaHRzIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzA4MmU3NTIwZWNlMjQxOTA4NWVjZDg4ZmVkN2UxYjFkLnNldENvbnRlbnQoaHRtbF9hZTM0NzI1NmZmY2Y0MzEzYWFiNDU1MTlhZmYyNTM4MCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYjc5MjcyZGE0MjQ3NGJkYjgzZDVhZjJjMmQ3M2JlNzAuYmluZFBvcHVwKHBvcHVwXzA4MmU3NTIwZWNlMjQxOTA4NWVjZDg4ZmVkN2UxYjFkKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNTdkNGYzYThmZGI0MWJjYjdhNjc5ODI1NTFhMWFmNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjMwMTUsIC03OS4zODk0OTM4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2I1OTcyNDIzNzNkMTQ4OGI5M2RjZDNlYTRiNzc1YmEyID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kODAyMTRmZmMwOWM0YTllODkxYmI2NzBlNTE0N2NlNiA9ICQoYDxkaXYgaWQ9Imh0bWxfZDgwMjE0ZmZjMDljNGE5ZTg5MWJiNjcwZTUxNDdjZTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlF1ZWVuJiMzOTtzIFBhcmssIE9udGFyaW8gUHJvdmluY2lhbCBHb3Zlcm5tZW50IENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2I1OTcyNDIzNzNkMTQ4OGI5M2RjZDNlYTRiNzc1YmEyLnNldENvbnRlbnQoaHRtbF9kODAyMTRmZmMwOWM0YTllODkxYmI2NzBlNTE0N2NlNik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYzU3ZDRmM2E4ZmRiNDFiY2I3YTY3OTgyNTUxYTFhZjcuYmluZFBvcHVwKHBvcHVwX2I1OTcyNDIzNzNkMTQ4OGI5M2RjZDNlYTRiNzc1YmEyKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NDAxZGJmMTI1Y2I0YzZmODljNzk1NmJhMDRjZGNkYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzg1NTYsIC03OS41MzIyNDI0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwMDBmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODAwMGZmIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8zZDI3Mjk4MTlmZGI0ZWYwYTQ0M2UyYmU5NDE1YzUxZiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZTZmNjEzOGEzMjQ0NDhlODhlYTY4YWJmYzBkZmJkMDQgPSAkKGA8ZGl2IGlkPSJodG1sX2U2ZjYxMzhhMzI0NDQ4ZTg4ZWE2OGFiZmMwZGZiZDA0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Jc2xpbmd0b24gQXZlbnVlLCBIdW1iZXIgVmFsbGV5IFZpbGxhZ2UgQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfM2QyNzI5ODE5ZmRiNGVmMGE0NDNlMmJlOTQxNWM1MWYuc2V0Q29udGVudChodG1sX2U2ZjYxMzhhMzI0NDQ4ZTg4ZWE2OGFiZmMwZGZiZDA0KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl85NDAxZGJmMTI1Y2I0YzZmODljNzk1NmJhMDRjZGNkYS5iaW5kUG9wdXAocG9wdXBfM2QyNzI5ODE5ZmRiNGVmMGE0NDNlMmJlOTQxNWM1MWYpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA3Y2NkY2I4ZWNiMzQ1NTM5ZjRhZTFkZWE3MTI5N2I0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzQ1OTA1Nzk5OTk5OTk2LCAtNzkuMzUyMTg4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzUyN2IxOTY2MTBiMTQ3MGViY2ExZDg2ZWFlMjlhMjcwID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9iNDU3Y2ZiZWQ5M2E0ZGYyYjZkODBmYjlhYmU2NGM0NiA9ICQoYDxkaXYgaWQ9Imh0bWxfYjQ1N2NmYmVkOTNhNGRmMmI2ZDgwZmI5YWJlNjRjNDYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvbiBNaWxscyBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF81MjdiMTk2NjEwYjE0NzBlYmNhMWQ4NmVhZTI5YTI3MC5zZXRDb250ZW50KGh0bWxfYjQ1N2NmYmVkOTNhNGRmMmI2ZDgwZmI5YWJlNjRjNDYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzA3Y2NkY2I4ZWNiMzQ1NTM5ZjRhZTFkZWE3MTI5N2I0LmJpbmRQb3B1cChwb3B1cF81MjdiMTk2NjEwYjE0NzBlYmNhMWQ4NmVhZTI5YTI3MCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGU2YTdmMGZmMTcyNGE0YjkyZjQzZDgwOGFlODE5YzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDYzOTcyLCAtNzkuMzA5OTM3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODBmZmI0IiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MGZmYjQiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2NhYzA5Y2I1Mzc5YjQ2MTI5ZjA5YmVkYmNjZGM2YWRiID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8wNzMyYTE2ODY0NzA0ZGU0YjRjMGEzMDA4NWY1ODE2ZCA9ICQoYDxkaXYgaWQ9Imh0bWxfMDczMmExNjg2NDcwNGRlNGI0YzBhMzAwODVmNTgxNmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmt2aWV3IEhpbGwsIFdvb2RiaW5lIEdhcmRlbnMgQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfY2FjMDljYjUzNzliNDYxMjlmMDliZWRiY2NkYzZhZGIuc2V0Q29udGVudChodG1sXzA3MzJhMTY4NjQ3MDRkZTRiNGMwYTMwMDg1ZjU4MTZkKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl84ZTZhN2YwZmYxNzI0YTRiOTJmNDNkODA4YWU4MTljMy5iaW5kUG9wdXAocG9wdXBfY2FjMDljYjUzNzliNDYxMjlmMDliZWRiY2NkYzZhZGIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FlMDQ4MDcyYmJkOTQzMjI4Y2FmOTBiNTRhYjlhZjQ1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwgLTc5LjM3ODkzNzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzI5MjdlNmIxMGQ4ZDQ2ZDJhMTNmMjY2NWQyNmQ1ZDI4ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8yOTM0NmMxNmJhMzA0YmZlOWU4ZDcyYTEwYWU2Y2I4MyA9ICQoYDxkaXYgaWQ9Imh0bWxfMjkzNDZjMTZiYTMwNGJmZTllOGQ3MmExMGFlNmNiODMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdhcmRlbiBEaXN0cmljdCwgUnllcnNvbiBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8yOTI3ZTZiMTBkOGQ0NmQyYTEzZjI2NjVkMjZkNWQyOC5zZXRDb250ZW50KGh0bWxfMjkzNDZjMTZiYTMwNGJmZTllOGQ3MmExMGFlNmNiODMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2FlMDQ4MDcyYmJkOTQzMjI4Y2FmOTBiNTRhYjlhZjQ1LmJpbmRQb3B1cChwb3B1cF8yOTI3ZTZiMTBkOGQ0NmQyYTEzZjI2NjVkMjZkNWQyOCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDkyNGIyZThiOTY2NDVjODg5YWYxODk4ZDczOTM5ZTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDk1NzcsIC03OS40NDUwNzI1OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9kMGIzNzFkMGY3NGM0Yjc3OTdkZWI2NzM5MjdjMjMzMyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMzJiNzc1YTNmYTk5NGRiZmE3ZWVjMzM2ODY5ZmUzNzAgPSAkKGA8ZGl2IGlkPSJodG1sXzMyYjc3NWEzZmE5OTRkYmZhN2VlYzMzNjg2OWZlMzcwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HbGVuY2Fpcm4gQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZDBiMzcxZDBmNzRjNGI3Nzk3ZGViNjczOTI3YzIzMzMuc2V0Q29udGVudChodG1sXzMyYjc3NWEzZmE5OTRkYmZhN2VlYzMzNjg2OWZlMzcwKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9kOTI0YjJlOGI5NjY0NWM4ODlhZjE4OThkNzM5MzllMS5iaW5kUG9wdXAocG9wdXBfZDBiMzcxZDBmNzRjNGI3Nzk3ZGViNjczOTI3YzIzMzMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UxMDc1NTBmZWRmNzRmMzc5MGJhZDk0NzJjNjhhZjE3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwOTQzMiwgLTc5LjU1NDcyNDQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODBmZmI0IiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MGZmYjQiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzYwOTMxNjhiYjU2YTQ1Y2M4MTVjYzdhNWMxZDE4OGIwID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kYmE2ODdmN2U1MDk0N2I3YmJkODNlYzRjZDMyM2FmOSA9ICQoYDxkaXYgaWQ9Imh0bWxfZGJhNjg3ZjdlNTA5NDdiN2JiZDgzZWM0Y2QzMjNhZjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3QgRGVhbmUgUGFyaywgUHJpbmNlc3MgR2FyZGVucywgTWFydGluIEdyb3ZlLCBJc2xpbmd0b24sIENsb3ZlcmRhbGUgQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNjA5MzE2OGJiNTZhNDVjYzgxNWNjN2E1YzFkMTg4YjAuc2V0Q29udGVudChodG1sX2RiYTY4N2Y3ZTUwOTQ3YjdiYmQ4M2VjNGNkMzIzYWY5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9lMTA3NTUwZmVkZjc0ZjM3OTBiYWQ5NDcyYzY4YWYxNy5iaW5kUG9wdXAocG9wdXBfNjA5MzE2OGJiNTZhNDVjYzgxNWNjN2E1YzFkMTg4YjApCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IyMGEyZGY3NTBiYTRkZGRiNWE0NmVkMzBhNmM3MTQzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzg0NTM1MSwgLTc5LjE2MDQ5NzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODAwMGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MDAwZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzE3M2YzNTc0OWU2NDQ4NTFhYTU3NjM0OWE2OTRmMzZhID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF80ZTJkZjRmOTdjMTc0ZGU2YWJlYTczN2FjMDY3NDRmNCA9ICQoYDxkaXYgaWQ9Imh0bWxfNGUyZGY0Zjk3YzE3NGRlNmFiZWE3MzdhYzA2NzQ0ZjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvdWdlIEhpbGwsIFBvcnQgVW5pb24sIEhpZ2hsYW5kIENyZWVrIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzE3M2YzNTc0OWU2NDQ4NTFhYTU3NjM0OWE2OTRmMzZhLnNldENvbnRlbnQoaHRtbF80ZTJkZjRmOTdjMTc0ZGU2YWJlYTczN2FjMDY3NDRmNCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYjIwYTJkZjc1MGJhNGRkZGI1YTQ2ZWQzMGE2YzcxNDMuYmluZFBvcHVwKHBvcHVwXzE3M2YzNTc0OWU2NDQ4NTFhYTU3NjM0OWE2OTRmMzZhKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZjZhNTNmYmQ1ZDc0YWI4OThjMTI2ZGNkMDI5Njk1ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNTg5OTcwMDAwMDAxLCAtNzkuMzQwOTIzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzg0NGI1NjY3NGZiODQ1Zjc5NDk1ZjNmNjc0MThmOWE0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9lMTE3ZTg5MWRiYjY0MmYyOWQ2MWM3YWFkMWRhMGQ5MyA9ICQoYDxkaXYgaWQ9Imh0bWxfZTExN2U4OTFkYmI2NDJmMjlkNjFjN2FhZDFkYTBkOTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvbiBNaWxscyBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF84NDRiNTY2NzRmYjg0NWY3OTQ5NWYzZjY3NDE4ZjlhNC5zZXRDb250ZW50KGh0bWxfZTExN2U4OTFkYmI2NDJmMjlkNjFjN2FhZDFkYTBkOTMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzJmNmE1M2ZiZDVkNzRhYjg5OGMxMjZkY2QwMjk2OTVlLmJpbmRQb3B1cChwb3B1cF84NDRiNTY2NzRmYjg0NWY3OTQ5NWYzZjY3NDE4ZjlhNCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjYyY2E0MjFkNWEyNGQ0Njg2OTkxNTMwY2YwMmRmYWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTUzNDM5MDAwMDAwMDUsIC03OS4zMTgzODg3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzFlNTI3NDAwMmIwMTQ4NGY4ZjMzN2ZjMTk0OTAwYzhlID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF82MjhhODY3NWQyYzQ0ZWMzYTI2MDY1YWE5OWQ4YmJlNiA9ICQoYDxkaXYgaWQ9Imh0bWxfNjI4YTg2NzVkMmM0NGVjM2EyNjA2NWFhOTlkOGJiZTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldvb2RiaW5lIEhlaWdodHMgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMWU1Mjc0MDAyYjAxNDg0ZjhmMzM3ZmMxOTQ5MDBjOGUuc2V0Q29udGVudChodG1sXzYyOGE4Njc1ZDJjNDRlYzNhMjYwNjVhYTk5ZDhiYmU2KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9mNjJjYTQyMWQ1YTI0ZDQ2ODY5OTE1MzBjZjAyZGZhZS5iaW5kUG9wdXAocG9wdXBfMWU1Mjc0MDAyYjAxNDg0ZjhmMzM3ZmMxOTQ5MDBjOGUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJhOWEzZTU1YWI0NDQ5NTFiZWY2OWQyMDU1YTQ5M2U1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNDkzOSwgLTc5LjM3NTQxNzldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMjk3NDMwNmI5NDNiNDQ3OWJkZDdjMDVjNzlmMzc4MzkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzg0ZmFlNTllYjJjYjRiMTE4ZDllNTFmZmFhYjJmZDA5ID0gJChgPGRpdiBpZD0iaHRtbF84NGZhZTU5ZWIyY2I0YjExOGQ5ZTUxZmZhYWIyZmQwOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24gQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMjk3NDMwNmI5NDNiNDQ3OWJkZDdjMDVjNzlmMzc4Mzkuc2V0Q29udGVudChodG1sXzg0ZmFlNTllYjJjYjRiMTE4ZDllNTFmZmFhYjJmZDA5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8yYTlhM2U1NWFiNDQ0OTUxYmVmNjlkMjA1NWE0OTNlNS5iaW5kUG9wdXAocG9wdXBfMjk3NDMwNmI5NDNiNDQ3OWJkZDdjMDVjNzlmMzc4MzkpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU1YTFkMDMzYzRmMzQ4Yzk5MDViMjc0YmFkMzkyYjY1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkzNzgxMywgLTc5LjQyODE5MTQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODBmZmI0IiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MGZmYjQiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzQ0YjQzYWZhMTNkODQzNTRiNTk0YmIxMGJhZTAyZDdhID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF81MjFjZGUzNGYyMjc0NDk1Yjk4NDZkNTc4YTFmNDE1NCA9ICQoYDxkaXYgaWQ9Imh0bWxfNTIxY2RlMzRmMjI3NDQ5NWI5ODQ2ZDU3OGExZjQxNTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWV3b29kLUNlZGFydmFsZSBDbHVzdGVyIDIuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF80NGI0M2FmYTEzZDg0MzU0YjU5NGJiMTBiYWUwMmQ3YS5zZXRDb250ZW50KGh0bWxfNTIxY2RlMzRmMjI3NDQ5NWI5ODQ2ZDU3OGExZjQxNTQpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzU1YTFkMDMzYzRmMzQ4Yzk5MDViMjc0YmFkMzkyYjY1LmJpbmRQb3B1cChwb3B1cF80NGI0M2FmYTEzZDg0MzU0YjU5NGJiMTBiYWUwMmQ3YSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWM5NDZkMzZlNjI0NDg2N2E5NmNmZWEzMDQxMWU0ZDIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDM1MTUyLCAtNzkuNTc3MjAwNzk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMWU0YTQxNjllNzA2NGIzNmI1YmMxOTk0NGZiOWZhNzAgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2YzNDMyNmQ1ODhhODRkOGViYjVmYTE1MzliNWE5ZThjID0gJChgPGRpdiBpZD0iaHRtbF9mMzQzMjZkNTg4YTg0ZDhlYmI1ZmExNTM5YjVhOWU4YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RXJpbmdhdGUsIEJsb29yZGFsZSBHYXJkZW5zLCBPbGQgQnVybmhhbXRob3JwZSwgTWFya2xhbmQgV29vZCBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8xZTRhNDE2OWU3MDY0YjM2YjViYzE5OTQ0ZmI5ZmE3MC5zZXRDb250ZW50KGh0bWxfZjM0MzI2ZDU4OGE4NGQ4ZWJiNWZhMTUzOWI1YTllOGMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzVjOTQ2ZDM2ZTYyNDQ4NjdhOTZjZmVhMzA0MTFlNGQyLmJpbmRQb3B1cChwb3B1cF8xZTRhNDE2OWU3MDY0YjM2YjViYzE5OTQ0ZmI5ZmE3MCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGViMTMyNDU2YmQ1NDFlN2EyNjk3NDFmMTNmM2RkMzIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NjM1NzI2LCAtNzkuMTg4NzExNV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF81M2NjMzBiNGIzMDA0Y2JjYTYzZWVjNDAxYmZmN2EwYSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZmE2NThhMjg5NmZkNGJlMjlmOGUwNzRmZjQwZGY1YjcgPSAkKGA8ZGl2IGlkPSJodG1sX2ZhNjU4YTI4OTZmZDRiZTI5ZjhlMDc0ZmY0MGRmNWI3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HdWlsZHdvb2QsIE1vcm5pbmdzaWRlLCBXZXN0IEhpbGwgQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNTNjYzMwYjRiMzAwNGNiY2E2M2VlYzQwMWJmZjdhMGEuc2V0Q29udGVudChodG1sX2ZhNjU4YTI4OTZmZDRiZTI5ZjhlMDc0ZmY0MGRmNWI3KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8wZWIxMzI0NTZiZDU0MWU3YTI2OTc0MWYxM2YzZGQzMi5iaW5kUG9wdXAocG9wdXBfNTNjYzMwYjRiMzAwNGNiY2E2M2VlYzQwMWJmZjdhMGEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAxZjFhYzZmYzhkYjQ1MTNhOGEyZGQzMTdlMTY5ZWM2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc2MzU3Mzk5OTk5OTksIC03OS4yOTMwMzEyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2VkYTc2MmRmZjhjNTRiYmE5OTFmNmFlYTNlMzZjMDhhID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8wNDMxMjc3ZjdmYzM0N2NiOWZkZDYyODQ1YWVhNjQ1ZCA9ICQoYDxkaXYgaWQ9Imh0bWxfMDQzMTI3N2Y3ZmMzNDdjYjlmZGQ2Mjg0NWFlYTY0NWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2VkYTc2MmRmZjhjNTRiYmE5OTFmNmFlYTNlMzZjMDhhLnNldENvbnRlbnQoaHRtbF8wNDMxMjc3ZjdmYzM0N2NiOWZkZDYyODQ1YWVhNjQ1ZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMDFmMWFjNmZjOGRiNDUxM2E4YTJkZDMxN2UxNjllYzYuYmluZFBvcHVwKHBvcHVwX2VkYTc2MmRmZjhjNTRiYmE5OTFmNmFlYTNlMzZjMDhhKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83OTY0MGU1YTVkYTg0Nzg2OTFiYTM2Yjg0YTAyZThkYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NDc3MDc5OTk5OTk5NiwgLTc5LjM3MzMwNjRdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNjNjZjI2M2ZhMjY5NGE1OGFmMTlkNjdiMWFkOWYyM2QgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2E1M2I4MjI0ODVhNzRjYjJhODEwMTZjNTNmMGIyY2M0ID0gJChgPGRpdiBpZD0iaHRtbF9hNTNiODIyNDg1YTc0Y2IyYTgxMDE2YzUzZjBiMmNjNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVyY3p5IFBhcmsgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNjNjZjI2M2ZhMjY5NGE1OGFmMTlkNjdiMWFkOWYyM2Quc2V0Q29udGVudChodG1sX2E1M2I4MjI0ODVhNzRjYjJhODEwMTZjNTNmMGIyY2M0KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl83OTY0MGU1YTVkYTg0Nzg2OTFiYTM2Yjg0YTAyZThkYS5iaW5kUG9wdXAocG9wdXBfNjNjZjI2M2ZhMjY5NGE1OGFmMTlkNjdiMWFkOWYyM2QpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA4NTVhM2ViOGVmYjQxNjU4OWU0MjRjMTU3NTU5ZDBkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5MDI1NiwgLTc5LjQ1MzUxMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8wNmJmZjVjNjY4YjE0YThlOTIyZDM1MDNlNGUxZGY3ZiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZDAyY2QyZTE1M2JiNDI2N2I3MjZjM2JjNjM4YmUyZTUgPSAkKGA8ZGl2IGlkPSJodG1sX2QwMmNkMmUxNTNiYjQyNjdiNzI2YzNiYzYzOGJlMmU1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYWxlZG9uaWEtRmFpcmJhbmtzIENsdXN0ZXIgMi4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzA2YmZmNWM2NjhiMTRhOGU5MjJkMzUwM2U0ZTFkZjdmLnNldENvbnRlbnQoaHRtbF9kMDJjZDJlMTUzYmI0MjY3YjcyNmMzYmM2MzhiZTJlNSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMDg1NWEzZWI4ZWZiNDE2NTg5ZTQyNGMxNTc1NTlkMGQuYmluZFBvcHVwKHBvcHVwXzA2YmZmNWM2NjhiMTRhOGU5MjJkMzUwM2U0ZTFkZjdmKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zZjEwYTcyOGI4Nzg0YWM3ODlhMjZhMGExZTIzY2Y0OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc3MDk5MjEsIC03OS4yMTY5MTc0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8zOWIwZWM2Mzc1YTE0OTI5YTEyMDZmNDkwMjdjNzY3YiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMGYyNTRhM2YyODUyNGQ3OGE5NzdlM2EzNzk1YjE2NDEgPSAkKGA8ZGl2IGlkPSJodG1sXzBmMjU0YTNmMjg1MjRkNzhhOTc3ZTNhMzc5NWIxNjQxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Xb2J1cm4gQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMzliMGVjNjM3NWExNDkyOWExMjA2ZjQ5MDI3Yzc2N2Iuc2V0Q29udGVudChodG1sXzBmMjU0YTNmMjg1MjRkNzhhOTc3ZTNhMzc5NWIxNjQxKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8zZjEwYTcyOGI4Nzg0YWM3ODlhMjZhMGExZTIzY2Y0OS5iaW5kUG9wdXAocG9wdXBfMzliMGVjNjM3NWExNDkyOWExMjA2ZjQ5MDI3Yzc2N2IpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUzODJlNmJjNWVmZjQ5YzM4ZTZiOGVkYzNkM2U5ZmFiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA5MDYwNCwgLTc5LjM2MzQ1MTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZDI2MWQ3ZGNkYjgwNGNlNmIzYTA5NGEyZmI0OWVlZWIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzgxYjI2Yjk4ZDU5YTRmYmFiNWQ5N2RlN2Y3NDYzODFkID0gJChgPGRpdiBpZD0iaHRtbF84MWIyNmI5OGQ1OWE0ZmJhYjVkOTdkZTdmNzQ2MzgxZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGVhc2lkZSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9kMjYxZDdkY2RiODA0Y2U2YjNhMDk0YTJmYjQ5ZWVlYi5zZXRDb250ZW50KGh0bWxfODFiMjZiOThkNTlhNGZiYWI1ZDk3ZGU3Zjc0NjM4MWQpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzUzODJlNmJjNWVmZjQ5YzM4ZTZiOGVkYzNkM2U5ZmFiLmJpbmRQb3B1cChwb3B1cF9kMjYxZDdkY2RiODA0Y2U2YjNhMDk0YTJmYjQ5ZWVlYikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTVmMjI0NTFhNmRmNGY5Yzk0ODMyYTg3ODEyNzA4NWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTc5NTI0LCAtNzkuMzg3MzgyNl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82OTA1MDYyZWY3NmU0NWMyYTkwOWY0NzEwNWQ4MGU3YSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMDlkOTQ2YjIwODg4NGI3ZTkzOTk1NjNjMGNlMzI0OGEgPSAkKGA8ZGl2IGlkPSJodG1sXzA5ZDk0NmIyMDg4ODRiN2U5Mzk5NTYzYzBjZTMyNDhhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIEJheSBTdHJlZXQgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNjkwNTA2MmVmNzZlNDVjMmE5MDlmNDcxMDVkODBlN2Euc2V0Q29udGVudChodG1sXzA5ZDk0NmIyMDg4ODRiN2U5Mzk5NTYzYzBjZTMyNDhhKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8xNWYyMjQ1MWE2ZGY0ZjljOTQ4MzJhODc4MTI3MDg1YS5iaW5kUG9wdXAocG9wdXBfNjkwNTA2MmVmNzZlNDVjMmE5MDlmNDcxMDVkODBlN2EpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M2OTUxMTU0NmQwMTRhNjM5MTZlN2VlOWE2ZGY5ZTQ4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5NTQyLCAtNzkuNDIyNTYzN10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8yZTYxYjZkNDM4MTI0ZTA2YTU4OTgzNGJlMmI4MDJiYyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMDRiNDJhYjg1MTIxNGJmYWIxNTE3MmE3OGE4ZTliNzIgPSAkKGA8ZGl2IGlkPSJodG1sXzA0YjQyYWI4NTEyMTRiZmFiMTUxNzJhNzhhOGU5YjcyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaHJpc3RpZSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8yZTYxYjZkNDM4MTI0ZTA2YTU4OTgzNGJlMmI4MDJiYy5zZXRDb250ZW50KGh0bWxfMDRiNDJhYjg1MTIxNGJmYWIxNTE3MmE3OGE4ZTliNzIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2M2OTUxMTU0NmQwMTRhNjM5MTZlN2VlOWE2ZGY5ZTQ4LmJpbmRQb3B1cChwb3B1cF8yZTYxYjZkNDM4MTI0ZTA2YTU4OTgzNGJlMmI4MDJiYykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODhhMTI2YjQ0YmQ4NDM1MmJiYzY3Y2MyOTJhMzZjYmIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzMxMzYsIC03OS4yMzk0NzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF81NmRjMmM2NDg1MWE0OWJlYmQwNjAzOWE3OTcyOGQ3NiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYWU3MTM3ODMwM2NmNGFhOWFjNGRiYmM0YmI1NDdiYTQgPSAkKGA8ZGl2IGlkPSJodG1sX2FlNzEzNzgzMDNjZjRhYTlhYzRkYmJjNGJiNTQ3YmE0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZWRhcmJyYWUgQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNTZkYzJjNjQ4NTFhNDliZWJkMDYwMzlhNzk3MjhkNzYuc2V0Q29udGVudChodG1sX2FlNzEzNzgzMDNjZjRhYTlhYzRkYmJjNGJiNTQ3YmE0KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl84OGExMjZiNDRiZDg0MzUyYmJjNjdjYzI5MmEzNmNiYi5iaW5kUG9wdXAocG9wdXBfNTZkYzJjNjQ4NTFhNDliZWJkMDYwMzlhNzk3MjhkNzYpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZlZTUxYzI2ZWI4YTQ0Mjk5ZmU4ZmM0NDc0MzU4NjdiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODAzNzYyMiwgLTc5LjM2MzQ1MTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MGZmYjQiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMTFjNzQwNTFkZTk3NDIwOWJmMjY0MGEzMzAzMGM0NTIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2U4ZjlkNGZhM2FlNDRmMDY5Yjg4Yjg3NjRkYzU4MTEwID0gJChgPGRpdiBpZD0iaHRtbF9lOGY5ZDRmYTNhZTQ0ZjA2OWI4OGI4NzY0ZGM1ODExMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGlsbGNyZXN0IFZpbGxhZ2UgQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMTFjNzQwNTFkZTk3NDIwOWJmMjY0MGEzMzAzMGM0NTIuc2V0Q29udGVudChodG1sX2U4ZjlkNGZhM2FlNDRmMDY5Yjg4Yjg3NjRkYzU4MTEwKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl82ZWU1MWMyNmViOGE0NDI5OWZlOGZjNDQ3NDM1ODY3Yi5iaW5kUG9wdXAocG9wdXBfMTFjNzQwNTFkZTk3NDIwOWJmMjY0MGEzMzAzMGM0NTIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U2NDkyYTMxNzcwNDQ0YjhiYjU1MWQ5MzdiNzhlZWMyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU0MzI4MywgLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MGZmYjQiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYTJiYTBiMDY2MDNkNGQyMjgxYTgxOTY1YmM5YzBlNDcgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2YzMTkyYzYwYjc4MDQ2NzFiYmMwY2VjODdhODExYzgyID0gJChgPGRpdiBpZD0iaHRtbF9mMzE5MmM2MGI3ODA0NjcxYmJjMGNlYzg3YTgxMWM4MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF0aHVyc3QgTWFub3IsIFdpbHNvbiBIZWlnaHRzLCBEb3duc3ZpZXcgTm9ydGggQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYTJiYTBiMDY2MDNkNGQyMjgxYTgxOTY1YmM5YzBlNDcuc2V0Q29udGVudChodG1sX2YzMTkyYzYwYjc4MDQ2NzFiYmMwY2VjODdhODExYzgyKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9lNjQ5MmEzMTc3MDQ0NGI4YmI1NTFkOTM3Yjc4ZWVjMi5iaW5kUG9wdXAocG9wdXBfYTJiYTBiMDY2MDNkNGQyMjgxYTgxOTY1YmM5YzBlNDcpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg5NzdkYjU1ODBhNTRkMGU4MjYxMTc2N2E4ZDVjMmM5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA1MzY4OSwgLTc5LjM0OTM3MTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzYxNTIzNmRkZDVjMDQyNDk5NmE1OTJiNDk2NzEyNjlkID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9iODIzMjcxMTNkNTk0NmQxODNmYjkyYzQ4NGU3ZWE3YSA9ICQoYDxkaXYgaWQ9Imh0bWxfYjgyMzI3MTEzZDU5NDZkMTgzZmI5MmM0ODRlN2VhN2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRob3JuY2xpZmZlIFBhcmsgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNjE1MjM2ZGRkNWMwNDI0OTk2YTU5MmI0OTY3MTI2OWQuc2V0Q29udGVudChodG1sX2I4MjMyNzExM2Q1OTQ2ZDE4M2ZiOTJjNDg0ZTdlYTdhKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl84OTc3ZGI1NTgwYTU0ZDBlODI2MTE3NjdhOGQ1YzJjOS5iaW5kUG9wdXAocG9wdXBfNjE1MjM2ZGRkNWMwNDI0OTk2YTU5MmI0OTY3MTI2OWQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk3MDY4YTc2MmZhZTQ1MzBhODA5Y2RjYjE3OWVmYjExID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsIC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2FiZDM1OWQ5ZWYwNTQ1NmU5NGExNzg2YWU1MzM2MTE5ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kZjhiNWY2NTVlMzM0NzBhYWE3YWU0YzI2YjM5ZWVhNiA9ICQoYDxkaXYgaWQ9Imh0bWxfZGY4YjVmNjU1ZTMzNDcwYWFhN2FlNGMyNmIzOWVlYTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJpY2htb25kLCBBZGVsYWlkZSwgS2luZyBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9hYmQzNTlkOWVmMDU0NTZlOTRhMTc4NmFlNTMzNjExOS5zZXRDb250ZW50KGh0bWxfZGY4YjVmNjU1ZTMzNDcwYWFhN2FlNGMyNmIzOWVlYTYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzk3MDY4YTc2MmZhZTQ1MzBhODA5Y2RjYjE3OWVmYjExLmJpbmRQb3B1cChwb3B1cF9hYmQzNTlkOWVmMDU0NTZlOTRhMTc4NmFlNTMzNjExOSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTdjNjU5MTY5NTUyNDJmNTgwYjAxMWE1MmZmMWMyNWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjkwMDUxMDAwMDAwMSwgLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMmE3MDkzMTA1MzMwNGE4MTljZDlmNzg4ZTNiMTNiNDQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzE3NTY1MzhhZDMxYzQxZjFiZTBiMDE1YzdiYjBiZmRmID0gJChgPGRpdiBpZD0iaHRtbF8xNzU2NTM4YWQzMWM0MWYxYmUwYjAxNWM3YmIwYmZkZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RHVmZmVyaW4sIERvdmVyY291cnQgVmlsbGFnZSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8yYTcwOTMxMDUzMzA0YTgxOWNkOWY3ODhlM2IxM2I0NC5zZXRDb250ZW50KGh0bWxfMTc1NjUzOGFkMzFjNDFmMWJlMGIwMTVjN2JiMGJmZGYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2E3YzY1OTE2OTU1MjQyZjU4MGIwMTFhNTJmZjFjMjViLmJpbmRQb3B1cChwb3B1cF8yYTcwOTMxMDUzMzA0YTgxOWNkOWY3ODhlM2IxM2I0NCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMThmZDgwZWQyNDAxNDQwODllZDMzN2U5NzI3YjY1ZTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NDQ3MzQyLCAtNzkuMjM5NDc2MDk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MGZmYjQiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNTk5MjBiNWJkMDc3NDY0OWJiMjI3Nzg5YWIzMWUxMjEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzc2Mzk3ZWI3YjNmNDQ4OTk5ZDNmYmFmYmU0ZDk2OTJiID0gJChgPGRpdiBpZD0iaHRtbF83NjM5N2ViN2IzZjQ0ODk5OWQzZmJhZmJlNGQ5NjkyYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U2NhcmJvcm91Z2ggVmlsbGFnZSBDbHVzdGVyIDIuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF81OTkyMGI1YmQwNzc0NjQ5YmIyMjc3ODlhYjMxZTEyMS5zZXRDb250ZW50KGh0bWxfNzYzOTdlYjdiM2Y0NDg5OTlkM2ZiYWZiZTRkOTY5MmIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzE4ZmQ4MGVkMjQwMTQ0MDg5ZWQzMzdlOTcyN2I2NWU2LmJpbmRQb3B1cChwb3B1cF81OTkyMGI1YmQwNzc0NjQ5YmIyMjc3ODlhYjMxZTEyMSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWRhMmI4YjQyYTRiNDhkOThkZmMzYjA0ZmQ1MzY1OTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Nzg1MTc1LCAtNzkuMzQ2NTU1N10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF80YzZiNjllMmQ4MGQ0OTU4YWZhNzBiZGMxN2JkZmMzZiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNjExYTkwZTZiYThkNDkzZGFjMzIyNWJhMDNjYTVjNzAgPSAkKGA8ZGl2IGlkPSJodG1sXzYxMWE5MGU2YmE4ZDQ5M2RhYzMyMjViYTAzY2E1YzcwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GYWlydmlldywgSGVucnkgRmFybSwgT3Jpb2xlIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzRjNmI2OWUyZDgwZDQ5NThhZmE3MGJkYzE3YmRmYzNmLnNldENvbnRlbnQoaHRtbF82MTFhOTBlNmJhOGQ0OTNkYWMzMjI1YmEwM2NhNWM3MCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYWRhMmI4YjQyYTRiNDhkOThkZmMzYjA0ZmQ1MzY1OTAuYmluZFBvcHVwKHBvcHVwXzRjNmI2OWUyZDgwZDQ5NThhZmE3MGJkYzE3YmRmYzNmKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MGM4NzRlNzQ5YzY0M2M2OTNlZjY4ZjAxZDU2OWVhNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc2Nzk4MDMsIC03OS40ODcyNjE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8xM2UzYTEzNzhmODg0YTMzYTNjNmE0NDcyODg0MmQ2YSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNjMyYmI1M2RmZWZmNDY2YWI4YWI5ZTdmMTBhZDQ2ZTAgPSAkKGA8ZGl2IGlkPSJodG1sXzYzMmJiNTNkZmVmZjQ2NmFiOGFiOWU3ZjEwYWQ0NmUwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aHdvb2QgUGFyaywgWW9yayBVbml2ZXJzaXR5IENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzEzZTNhMTM3OGY4ODRhMzNhM2M2YTQ0NzI4ODQyZDZhLnNldENvbnRlbnQoaHRtbF82MzJiYjUzZGZlZmY0NjZhYjhhYjllN2YxMGFkNDZlMCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfOTBjODc0ZTc0OWM2NDNjNjkzZWY2OGYwMWQ1NjllYTYuYmluZFBvcHVwKHBvcHVwXzEzZTNhMTM3OGY4ODRhMzNhM2M2YTQ0NzI4ODQyZDZhKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NmI2ZDJkZWI2NDc0NDcwYjQxMzZhMGM2ZTAwZGFkZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4NTM0NywgLTc5LjMzODEwNjVdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMjE4N2YwMTE2ODNhNDhjZmIzYzc2MWEwZjAyODBhNmIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzJiNmZmODRmYTA2NDRlNzNhMTliNjU3NjljZGE0M2Y2ID0gJChgPGRpdiBpZD0iaHRtbF8yYjZmZjg0ZmEwNjQ0ZTczYTE5YjY1NzY5Y2RhNDNmNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvLCBCcm9hZHZpZXcgTm9ydGggKE9sZCBFYXN0IFlvcmspIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzIxODdmMDExNjgzYTQ4Y2ZiM2M3NjFhMGYwMjgwYTZiLnNldENvbnRlbnQoaHRtbF8yYjZmZjg0ZmEwNjQ0ZTczYTE5YjY1NzY5Y2RhNDNmNik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNzZiNmQyZGViNjQ3NDQ3MGI0MTM2YTBjNmUwMGRhZGQuYmluZFBvcHVwKHBvcHVwXzIxODdmMDExNjgzYTQ4Y2ZiM2M3NjFhMGYwMjgwYTZiKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kYTllYmUyMDU2NGY0OGUyYTY4NTc0N2U2NzE3YzY3OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MDgxNTcsIC03OS4zODE3NTIyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF81OTFiYmMzMWM1YTc0NDlhOTJhNTY0OGExODViYTFlYSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfN2UxYzdhMDc0ZDg3NDFlN2I3NjA4OWY5MWEyNDEyODQgPSAkKGA8ZGl2IGlkPSJodG1sXzdlMWM3YTA3NGQ4NzQxZTdiNzYwODlmOTFhMjQxMjg0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQgRWFzdCwgVW5pb24gU3RhdGlvbiwgVG9yb250byBJc2xhbmRzIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzU5MWJiYzMxYzVhNzQ0OWE5MmE1NjQ4YTE4NWJhMWVhLnNldENvbnRlbnQoaHRtbF83ZTFjN2EwNzRkODc0MWU3Yjc2MDg5ZjkxYTI0MTI4NCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfZGE5ZWJlMjA1NjRmNDhlMmE2ODU3NDdlNjcxN2M2NzkuYmluZFBvcHVwKHBvcHVwXzU5MWJiYzMxYzVhNzQ0OWE5MmE1NjQ4YTE4NWJhMWVhKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iN2Y0ZTBmMmVhYzM0ZjdhYTYwYjEwZWY4YWVmNDQ2ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwgLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZjQyNmNmNzllYmRiNGJiZGJhNmIzZDI1ODI2YjljZDYgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzBiZmIxYTk4OWQ5ZDRkMDc5ODA0OWY4ZDU2YTQwM2FmID0gJChgPGRpdiBpZD0iaHRtbF8wYmZiMWE5ODlkOWQ0ZDA3OTgwNDlmOGQ1NmE0MDNhZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGl0dGxlIFBvcnR1Z2FsLCBUcmluaXR5IENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2Y0MjZjZjc5ZWJkYjRiYmRiYTZiM2QyNTgyNmI5Y2Q2LnNldENvbnRlbnQoaHRtbF8wYmZiMWE5ODlkOWQ0ZDA3OTgwNDlmOGQ1NmE0MDNhZik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYjdmNGUwZjJlYWMzNGY3YWE2MGIxMGVmOGFlZjQ0NmQuYmluZFBvcHVwKHBvcHVwX2Y0MjZjZjc5ZWJkYjRiYmRiYTZiM2QyNTgyNmI5Y2Q2KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MGEzY2I0M2RhZjM0YmY1YTg2YTlmNWUyODgwN2EwNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNzkyOTIsIC03OS4yNjIwMjk0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hN2Q2OThmMTc2OTk0Y2UxOTMxMzIwNDliNTc1ZmFlNCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNTkzNThiODFiN2JhNDZkZmJkMGRjMzY5YWZlZjhlZjIgPSAkKGA8ZGl2IGlkPSJodG1sXzU5MzU4YjgxYjdiYTQ2ZGZiZDBkYzM2OWFmZWY4ZWYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZW5uZWR5IFBhcmssIElvbnZpZXcsIEVhc3QgQmlyY2htb3VudCBQYXJrIENsdXN0ZXIgMi4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2E3ZDY5OGYxNzY5OTRjZTE5MzEzMjA0OWI1NzVmYWU0LnNldENvbnRlbnQoaHRtbF81OTM1OGI4MWI3YmE0NmRmYmQwZGMzNjlhZmVmOGVmMik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfOTBhM2NiNDNkYWYzNGJmNWE4NmE5ZjVlMjg4MDdhMDcuYmluZFBvcHVwKHBvcHVwX2E3ZDY5OGYxNzY5OTRjZTE5MzEzMjA0OWI1NzVmYWU0KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zNTc1YzAxNDVhODY0MjY0OThmNWUwOTk5NzNmYjI4NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4Njk0NzMsIC03OS4zODU5NzVdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MGZmYjQiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNWU0OGI0OTRkM2FhNGM0OGJhNmRkOGFjMTIxMzEwMDEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzE4Y2MwNTExMGU5NzRjZTZiYmQzYThjMmY3ZDFkMjkxID0gJChgPGRpdiBpZD0iaHRtbF8xOGNjMDUxMTBlOTc0Y2U2YmJkM2E4YzJmN2QxZDI5MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF5dmlldyBWaWxsYWdlIENsdXN0ZXIgMi4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzVlNDhiNDk0ZDNhYTRjNDhiYTZkZDhhYzEyMTMxMDAxLnNldENvbnRlbnQoaHRtbF8xOGNjMDUxMTBlOTc0Y2U2YmJkM2E4YzJmN2QxZDI5MSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMzU3NWMwMTQ1YTg2NDI2NDk4ZjVlMDk5OTczZmIyODYuYmluZFBvcHVwKHBvcHVwXzVlNDhiNDk0ZDNhYTRjNDhiYTZkZDhhYzEyMTMxMDAxKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MmE0YzE2NTA1Nzk0ZDYzOGYwZDE4ZDcyZTE3OWIzNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczNzQ3MzIwMDAwMDAwNCwgLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2YxNDdiZGEzMWJmMTQyZDZiODZmNzdiYWRhYzMzOGZlID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hYWJjNzQxNTFjNjI0YzcwOTYyNWFmNWQyZTM2OTkxNiA9ICQoYDxkaXYgaWQ9Imh0bWxfYWFiYzc0MTUxYzYyNGM3MDk2MjVhZjVkMmUzNjk5MTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldyBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9mMTQ3YmRhMzFiZjE0MmQ2Yjg2Zjc3YmFkYWMzMzhmZS5zZXRDb250ZW50KGh0bWxfYWFiYzc0MTUxYzYyNGM3MDk2MjVhZjVkMmUzNjk5MTYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzkyYTRjMTY1MDU3OTRkNjM4ZjBkMThkNzJlMTc5YjM0LmJpbmRQb3B1cChwb3B1cF9mMTQ3YmRhMzFiZjE0MmQ2Yjg2Zjc3YmFkYWMzMzhmZSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGQ2ZDI4NmE3MTAxNGE4Mzk2YWI4ZmRlNWIwNGMwODkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NTcxLCAtNzkuMzUyMTg4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzk3NjU5MWYwMTA5NzRhYTk5NTQyNTA1MTNlZThkYmNkID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9lM2IxOWUxZWExNjI0YTg0YWNhMTNlOTRkMWI0OWI1MSA9ICQoYDxkaXYgaWQ9Imh0bWxfZTNiMTllMWVhMTYyNGE4NGFjYTEzZTk0ZDFiNDliNTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBEYW5mb3J0aCBXZXN0LCBSaXZlcmRhbGUgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfOTc2NTkxZjAxMDk3NGFhOTk1NDI1MDUxM2VlOGRiY2Quc2V0Q29udGVudChodG1sX2UzYjE5ZTFlYTE2MjRhODRhY2ExM2U5NGQxYjQ5YjUxKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl84ZDZkMjg2YTcxMDE0YTgzOTZhYjhmZGU1YjA0YzA4OS5iaW5kUG9wdXAocG9wdXBfOTc2NTkxZjAxMDk3NGFhOTk1NDI1MDUxM2VlOGRiY2QpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhhZmI3NjEyYTY4NjQzNzVhMWE5NTlhMDg5MjZlN2I5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ3MTc2OCwgLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzg4ODBkM2MxODIxMTQ0ZTFiZWU1MmU3NDU5ZmZkNTg2ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hMDJhYTFkMjZmZTc0ZGMxYjk5ZGM0YmE4Y2NjZmQ2NyA9ICQoYDxkaXYgaWQ9Imh0bWxfYTAyYWExZDI2ZmU3NGRjMWI5OWRjNGJhOGNjY2ZkNjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRvcm9udG8gRG9taW5pb24gQ2VudHJlLCBEZXNpZ24gRXhjaGFuZ2UgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfODg4MGQzYzE4MjExNDRlMWJlZTUyZTc0NTlmZmQ1ODYuc2V0Q29udGVudChodG1sX2EwMmFhMWQyNmZlNzRkYzFiOTlkYzRiYThjY2NmZDY3KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl84YWZiNzYxMmE2ODY0Mzc1YTFhOTU5YTA4OTI2ZTdiOS5iaW5kUG9wdXAocG9wdXBfODg4MGQzYzE4MjExNDRlMWJlZTUyZTc0NTlmZmQ1ODYpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ1NGFmNDYzM2FhNTRmNTRhNGMxNGZkYTU2ZDgxNzNlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2ODQ3MiwgLTc5LjQyODE5MTQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2EzYjhjMWY4NGQ4NjRiYjhhNGEzNzI2MjNhODdlMmQwID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8zZWRmMzE4YmE4YjI0ZTU5YmFiZjQ1ZmM3ZTU1ZGQwYSA9ICQoYDxkaXYgaWQ9Imh0bWxfM2VkZjMxOGJhOGIyNGU1OWJhYmY0NWZjN2U1NWRkMGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrdG9uLCBQYXJrZGFsZSBWaWxsYWdlLCBFeGhpYml0aW9uIFBsYWNlIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2EzYjhjMWY4NGQ4NjRiYjhhNGEzNzI2MjNhODdlMmQwLnNldENvbnRlbnQoaHRtbF8zZWRmMzE4YmE4YjI0ZTU5YmFiZjQ1ZmM3ZTU1ZGQwYSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNDU0YWY0NjMzYWE1NGY1NGE0YzE0ZmRhNTZkODE3M2UuYmluZFBvcHVwKHBvcHVwX2EzYjhjMWY4NGQ4NjRiYjhhNGEzNzI2MjNhODdlMmQwKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zZGJlOWE4NDYzNTY0MWY1OTJkZGQzYzVkMWE0NmQ1MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMTExMTcwMDAwMDAwNCwgLTc5LjI4NDU3NzJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MGZmYjQiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfN2U4NDRiYjA5MTc5NGI2Yzk2N2FhODJiMDljMWViYTcgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzAzMjYwNGJlNzRmNjRhMTE5Yjc1N2M5MzkzYWIwZTU2ID0gJChgPGRpdiBpZD0iaHRtbF8wMzI2MDRiZTc0ZjY0YTExOWI3NTdjOTM5M2FiMGU1NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R29sZGVuIE1pbGUsIENsYWlybGVhLCBPYWtyaWRnZSBDbHVzdGVyIDIuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF83ZTg0NGJiMDkxNzk0YjZjOTY3YWE4MmIwOWMxZWJhNy5zZXRDb250ZW50KGh0bWxfMDMyNjA0YmU3NGY2NGExMTliNzU3YzkzOTNhYjBlNTYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzNkYmU5YTg0NjM1NjQxZjU5MmRkZDNjNWQxYTQ2ZDUxLmJpbmRQb3B1cChwb3B1cF83ZTg0NGJiMDkxNzk0YjZjOTY3YWE4MmIwOWMxZWJhNykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjRlZDdiMmNiYzRjNGYxZmI1ZTQzNTg0MDBkNGZmODEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTc0OTAyLCAtNzkuMzc0NzE0MDk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiMwMGI1ZWIiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzAwYjVlYiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYWQ4YmRmMjc5ZTA1NDYxNjhkNDQ0ZGE5YWVjYzY5Y2IgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzY2ZjQ2NTYzODEyNzQ0OWFiZTNlODU3MmQ0Zjk5MTYxID0gJChgPGRpdiBpZD0iaHRtbF82NmY0NjU2MzgxMjc0NDlhYmUzZTg1NzJkNGY5OTE2MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+WW9yayBNaWxscywgU2lsdmVyIEhpbGxzIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2FkOGJkZjI3OWUwNTQ2MTY4ZDQ0NGRhOWFlY2M2OWNiLnNldENvbnRlbnQoaHRtbF82NmY0NjU2MzgxMjc0NDlhYmUzZTg1NzJkNGY5OTE2MSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNjRlZDdiMmNiYzRjNGYxZmI1ZTQzNTg0MDBkNGZmODEuYmluZFBvcHVwKHBvcHVwX2FkOGJkZjI3OWUwNTQ2MTY4ZDQ0NGRhOWFlY2M2OWNiKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mOGFlNDg2NTc2YzA0NjY3OGQ2Mjk5YmNmZDFhM2Q4MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTAxNDYsIC03OS41MDY5NDM2XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODBmZmI0IiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MGZmYjQiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzZmMjIyZjY0MjliNzQ3YTc4ZTNjMTlkYjMxODEwMTMyID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF85YWViODc3MjFjZTc0NGJiYWU3MTMzZGIyYzVlOTkyYSA9ICQoYDxkaXYgaWQ9Imh0bWxfOWFlYjg3NzIxY2U3NDRiYmFlNzEzM2RiMmM1ZTk5MmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldyBDbHVzdGVyIDIuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF82ZjIyMmY2NDI5Yjc0N2E3OGUzYzE5ZGIzMTgxMDEzMi5zZXRDb250ZW50KGh0bWxfOWFlYjg3NzIxY2U3NDRiYmFlNzEzM2RiMmM1ZTk5MmEpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2Y4YWU0ODY1NzZjMDQ2Njc4ZDYyOTliY2ZkMWEzZDgyLmJpbmRQb3B1cChwb3B1cF82ZjIyMmY2NDI5Yjc0N2E3OGUzYzE5ZGIzMTgxMDEzMikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjBlOTc2ZDgwZWQ2NGEyZmJhOGJjYmQwOWU1NWNiODEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njg5OTg1LCAtNzkuMzE1NTcxNTk5OTk5OThdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNmVmZTRiNjExMzllNDI1Yjg4NzFjMzVkZTY1ZDc0MDAgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzY4Nzg4MTA4MDIxMzQ4ZDViZWExNGNkOWMxN2YyOTkyID0gJChgPGRpdiBpZD0iaHRtbF82ODc4ODEwODAyMTM0OGQ1YmVhMTRjZDljMTdmMjk5MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SW5kaWEgQmF6YWFyLCBUaGUgQmVhY2hlcyBXZXN0IENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzZlZmU0YjYxMTM5ZTQyNWI4ODcxYzM1ZGU2NWQ3NDAwLnNldENvbnRlbnQoaHRtbF82ODc4ODEwODAyMTM0OGQ1YmVhMTRjZDljMTdmMjk5Mik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNjBlOTc2ZDgwZWQ2NGEyZmJhOGJjYmQwOWU1NWNiODEuYmluZFBvcHVwKHBvcHVwXzZlZmU0YjYxMTM5ZTQyNWI4ODcxYzM1ZGU2NWQ3NDAwKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yOGQ2NTQyZDZmMDk0Njc1YTMzM2I3MTM5NWNlNjZmZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODE5ODUsIC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8wZTllZjliODdlYWQ0NjlhYWM0NWE2ZWUzM2VhM2ZjMSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNjgyNmU1NTNiYWVjNDJjYWFjNTY2MzdlZmU4MjlhNjYgPSAkKGA8ZGl2IGlkPSJodG1sXzY4MjZlNTUzYmFlYzQyY2FhYzU2NjM3ZWZlODI5YTY2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Db21tZXJjZSBDb3VydCwgVmljdG9yaWEgSG90ZWwgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMGU5ZWY5Yjg3ZWFkNDY5YWFjNDVhNmVlMzNlYTNmYzEuc2V0Q29udGVudChodG1sXzY4MjZlNTUzYmFlYzQyY2FhYzU2NjM3ZWZlODI5YTY2KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8yOGQ2NTQyZDZmMDk0Njc1YTMzM2I3MTM5NWNlNjZmZS5iaW5kUG9wdXAocG9wdXBfMGU5ZWY5Yjg3ZWFkNDY5YWFjNDVhNmVlMzNlYTNmYzEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NjMTM3NzMyMDA1MjRjMjc5NzI4ZWNlYTBlNmJjNWM5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEzNzU2MjAwMDAwMDA2LCAtNzkuNDkwMDczOF0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9kMDUzMzViNTRmNDM0NDhhODFlN2FkYjYwMDk1OWVmMyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNjY2MGZmNjJkYWUzNDIyMmE3ZmM5YTdiYzdiNGU4MWUgPSAkKGA8ZGl2IGlkPSJodG1sXzY2NjBmZjYyZGFlMzQyMjJhN2ZjOWE3YmM3YjRlODFlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBQYXJrLCBNYXBsZSBMZWFmIFBhcmssIFVwd29vZCBQYXJrIENsdXN0ZXIgMi4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2QwNTMzNWI1NGY0MzQ0OGE4MWU3YWRiNjAwOTU5ZWYzLnNldENvbnRlbnQoaHRtbF82NjYwZmY2MmRhZTM0MjIyYTdmYzlhN2JjN2I0ZTgxZSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfY2MxMzc3MzIwMDUyNGMyNzk3MjhlY2VhMGU2YmM1YzkuYmluZFBvcHVwKHBvcHVwX2QwNTMzNWI1NGY0MzQ0OGE4MWU3YWRiNjAwOTU5ZWYzKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MzkwZDcxYzkxYTY0NWFjOWU5ZDA0M2VkMWQ3NzFlNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1NjMwMzMsIC03OS41NjU5NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwMDBmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODAwMGZmIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iMjRlOWE2NWJmNjA0Y2FiYWJmMTU2NTAyZTU2MjdiMiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfODE3ZGIwY2IyNTE2NDRiNWE3Y2FlNjlkZjQwY2Y4ZjMgPSAkKGA8ZGl2IGlkPSJodG1sXzgxN2RiMGNiMjUxNjQ0YjVhN2NhZTY5ZGY0MGNmOGYzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IdW1iZXIgU3VtbWl0IENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2IyNGU5YTY1YmY2MDRjYWJhYmYxNTY1MDJlNTYyN2IyLnNldENvbnRlbnQoaHRtbF84MTdkYjBjYjI1MTY0NGI1YTdjYWU2OWRmNDBjZjhmMyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNjM5MGQ3MWM5MWE2NDVhYzllOWQwNDNlZDFkNzcxZTYuYmluZFBvcHVwKHBvcHVwX2IyNGU5YTY1YmY2MDRjYWJhYmYxNTY1MDJlNTYyN2IyKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZmEyOWNiNzY4NmU0YTEzOWZjOWIxMGFlMDEyZGVmNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxNjMxNiwgLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODBmZmI0IiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MGZmYjQiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzVhYTU3ODQ5MmFlMDRkZjg4NDIxNmVkOTdhMmY1N2QzID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF80ZDM5NDJhOTZjMzY0ZjJkOTBmY2Q5NTg5MDllZGZiMSA9ICQoYDxkaXYgaWQ9Imh0bWxfNGQzOTQyYTk2YzM2NGYyZDkwZmNkOTU4OTA5ZWRmYjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNsaWZmc2lkZSwgQ2xpZmZjcmVzdCwgU2NhcmJvcm91Z2ggVmlsbGFnZSBXZXN0IENsdXN0ZXIgMi4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzVhYTU3ODQ5MmFlMDRkZjg4NDIxNmVkOTdhMmY1N2QzLnNldENvbnRlbnQoaHRtbF80ZDM5NDJhOTZjMzY0ZjJkOTBmY2Q5NTg5MDllZGZiMSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYWZhMjljYjc2ODZlNGExMzlmYzliMTBhZTAxMmRlZjcuYmluZFBvcHVwKHBvcHVwXzVhYTU3ODQ5MmFlMDRkZjg4NDIxNmVkOTdhMmY1N2QzKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xYmI4Yjg1ZjZiMjc0MTIzYTMxY2ExOGUxY2RlNWVkOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4OTA1MywgLTc5LjQwODQ5Mjc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2E1ZGM1ZWRlOTY5YzQ2NGU5ZWE4NGQzZGE0YjY4MTBmID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8zMDNiZWQ1NjE3MTc0NzlkYjFhYzRkMmJlMGNkNDZmYSA9ICQoYDxkaXYgaWQ9Imh0bWxfMzAzYmVkNTYxNzE3NDc5ZGIxYWM0ZDJiZTBjZDQ2ZmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldpbGxvd2RhbGUsIE5ld3RvbmJyb29rIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2E1ZGM1ZWRlOTY5YzQ2NGU5ZWE4NGQzZGE0YjY4MTBmLnNldENvbnRlbnQoaHRtbF8zMDNiZWQ1NjE3MTc0NzlkYjFhYzRkMmJlMGNkNDZmYSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMWJiOGI4NWY2YjI3NDEyM2EzMWNhMThlMWNkZTVlZDguYmluZFBvcHVwKHBvcHVwX2E1ZGM1ZWRlOTY5YzQ2NGU5ZWE4NGQzZGE0YjY4MTBmKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYjA5MmJhNThiMzI0Y2ZiYmFkYWQxMzI2ODEzNDQ2MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODQ5NjQsIC03OS40OTU2OTc0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmMDAwMCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8zZTM3MGVmMmZhYzU0Yzk3OTdlZmI1ODU3MzMzOTgxNCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZjQ4MTBkNjk5NGQ5NDY3OWI2M2ZjMmIxZDQwNzI0MzEgPSAkKGA8ZGl2IGlkPSJodG1sX2Y0ODEwZDY5OTRkOTQ2NzliNjNmYzJiMWQ0MDcyNDMxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcgQ2x1c3RlciA0LjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfM2UzNzBlZjJmYWM1NGM5Nzk3ZWZiNTg1NzMzMzk4MTQuc2V0Q29udGVudChodG1sX2Y0ODEwZDY5OTRkOTQ2NzliNjNmYzJiMWQ0MDcyNDMxKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8yYjA5MmJhNThiMzI0Y2ZiYmFkYWQxMzI2ODEzNDQ2Mi5iaW5kUG9wdXAocG9wdXBfM2UzNzBlZjJmYWM1NGM5Nzk3ZWZiNTg1NzMzMzk4MTQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc1NWQwMGFiNThiOTRiMjk5MTI4Y2I1NWQ4MjE1YzY3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU5NTI1NSwgLTc5LjM0MDkyM10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF83OTNkOGUyYWViYjM0ZjliODIwMGU2Yzk1MzMxODdmZCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZTZmYjJlYzUzMGRjNDJiYTg0YWExMzIxYTI4ZjNhNmQgPSAkKGA8ZGl2IGlkPSJodG1sX2U2ZmIyZWM1MzBkYzQyYmE4NGFhMTMyMWEyOGYzYTZkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdHVkaW8gRGlzdHJpY3QgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNzkzZDhlMmFlYmIzNGY5YjgyMDBlNmM5NTMzMTg3ZmQuc2V0Q29udGVudChodG1sX2U2ZmIyZWM1MzBkYzQyYmE4NGFhMTMyMWEyOGYzYTZkKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl83NTVkMDBhYjU4Yjk0YjI5OTEyOGNiNTVkODIxNWM2Ny5iaW5kUG9wdXAocG9wdXBfNzkzZDhlMmFlYmIzNGY5YjgyMDBlNmM5NTMzMTg3ZmQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg2NDQzNmI1MzJkZDQ2ZDM5ZWFiNzg0Y2UyZmI4YWMyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzMzMjgyNSwgLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMDU2OGRiYmI2NzkyNDZkNmJkOTUyODM4Y2I0ODRlNzQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzk3NGU5YTIxNmRmMDRlNGU5ZGNhNTJlYWY1ZDcyZTYwID0gJChgPGRpdiBpZD0iaHRtbF85NzRlOWEyMTZkZjA0ZTRlOWRjYTUyZWFmNWQ3MmU2MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVkZm9yZCBQYXJrLCBMYXdyZW5jZSBNYW5vciBFYXN0IENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzA1NjhkYmJiNjc5MjQ2ZDZiZDk1MjgzOGNiNDg0ZTc0LnNldENvbnRlbnQoaHRtbF85NzRlOWEyMTZkZjA0ZTRlOWRjYTUyZWFmNWQ3MmU2MCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfODY0NDM2YjUzMmRkNDZkMzllYWI3ODRjZTJmYjhhYzIuYmluZFBvcHVwKHBvcHVwXzA1NjhkYmJiNjc5MjQ2ZDZiZDk1MjgzOGNiNDg0ZTc0KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ZGJmZmRlNzg4MTQ0MmI1YTRjNTQwMThkMDU0MTEzZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5MTExNTgsIC03OS40NzYwMTMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF84YmI4ZDViYTAxMWY0NjY0YWJiZjM2NDM1YTIwN2JhYyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMmU5ZjhiYjlhZDI5NDRhMTk4MGQ4NDIzYjExNDA4MmYgPSAkKGA8ZGl2IGlkPSJodG1sXzJlOWY4YmI5YWQyOTQ0YTE5ODBkODQyM2IxMTQwODJmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZWwgUmF5LCBNb3VudCBEZW5uaXMsIEtlZWxzZGFsZSBhbmQgU2lsdmVydGhvcm4gQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfOGJiOGQ1YmEwMTFmNDY2NGFiYmYzNjQzNWEyMDdiYWMuc2V0Q29udGVudChodG1sXzJlOWY4YmI5YWQyOTQ0YTE5ODBkODQyM2IxMTQwODJmKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl80ZGJmZmRlNzg4MTQ0MmI1YTRjNTQwMThkMDU0MTEzZi5iaW5kUG9wdXAocG9wdXBfOGJiOGQ1YmEwMTFmNDY2NGFiYmYzNjQzNWEyMDdiYWMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IyMzIzZDZlNmFmNDQ1YThiNjFkN2IyN2QzMDcyOWRjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI0NzY1OSwgLTc5LjUzMjI0MjQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODAwMGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MDAwZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzdhN2JlYzU0NmZmNDQ1ZGZiZmIzNzliZmU3M2UwY2M1ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9jNDZlNjlkMzNkZjU0MDkwYmFjZDM1OTdhN2Q0ZjdkYyA9ICQoYDxkaXYgaWQ9Imh0bWxfYzQ2ZTY5ZDMzZGY1NDA5MGJhY2QzNTk3YTdkNGY3ZGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWJlcmxlYSwgRW1lcnkgQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfN2E3YmVjNTQ2ZmY0NDVkZmJmYjM3OWJmZTczZTBjYzUuc2V0Q29udGVudChodG1sX2M0NmU2OWQzM2RmNTQwOTBiYWNkMzU5N2E3ZDRmN2RjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9iMjMyM2Q2ZTZhZjQ0NWE4YjYxZDdiMjdkMzA3MjlkYy5iaW5kUG9wdXAocG9wdXBfN2E3YmVjNTQ2ZmY0NDVkZmJmYjM3OWJmZTczZTBjYzUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhiZGYxYmY3Y2MyMTQ5ZjliOTgxNGIxN2M0ZTgxNTJhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkyNjU3MDAwMDAwMDA0LCAtNzkuMjY0ODQ4MV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwMDBmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODAwMGZmIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF84MGM4YTZlZWMzYzU0MWFkOGU3ZTI2NjliZmQxMDRmZSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZTljODcwZDE3YzRiNDlkNGI1Nzk5ZGRjNDUzZGQyNWYgPSAkKGA8ZGl2IGlkPSJodG1sX2U5Yzg3MGQxN2M0YjQ5ZDRiNTc5OWRkYzQ1M2RkMjVmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CaXJjaCBDbGlmZiwgQ2xpZmZzaWRlIFdlc3QgQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfODBjOGE2ZWVjM2M1NDFhZDhlN2UyNjY5YmZkMTA0ZmUuc2V0Q29udGVudChodG1sX2U5Yzg3MGQxN2M0YjQ5ZDRiNTc5OWRkYzQ1M2RkMjVmKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl84YmRmMWJmN2NjMjE0OWY5Yjk4MTRiMTdjNGU4MTUyYS5iaW5kUG9wdXAocG9wdXBfODBjOGE2ZWVjM2M1NDFhZDhlN2UyNjY5YmZkMTA0ZmUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYwNWE5N2MyYzQ0MTQyNDE5Y2Y1ODM1N2E0NmYxNzQ0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzcwMTE5OSwgLTc5LjQwODQ5Mjc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzBiMzJjZTkzMTE4ZTRjZmY4MjRjYjJjZGVjY2JiOTRlID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9jMzgwNzY4MWE2MzY0MjZmOTVlN2ZhMTIwYzA4NWY1ZSA9ICQoYDxkaXYgaWQ9Imh0bWxfYzM4MDc2ODFhNjM2NDI2Zjk1ZTdmYTEyMGMwODVmNWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldpbGxvd2RhbGUsIFdpbGxvd2RhbGUgRWFzdCBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8wYjMyY2U5MzExOGU0Y2ZmODI0Y2IyY2RlY2NiYjk0ZS5zZXRDb250ZW50KGh0bWxfYzM4MDc2ODFhNjM2NDI2Zjk1ZTdmYTEyMGMwODVmNWUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzYwNWE5N2MyYzQ0MTQyNDE5Y2Y1ODM1N2E0NmYxNzQ0LmJpbmRQb3B1cChwb3B1cF8wYjMyY2U5MzExOGU0Y2ZmODI0Y2IyY2RlY2NiYjk0ZSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjZmZjZkZmEwNDAyNGYyMjg1YTI2YzY1M2FlM2VlZTMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NjE2MzEzLCAtNzkuNTIwOTk5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MGZmYjQiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMTQxM2I4NzE1MjlkNDEwYjg1OWY0YWYyY2VlMzQyOGQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzQyMmU4YTA0NmU1OTQyYjRhNzlhZDIxYTc1NjdiNjU0ID0gJChgPGRpdiBpZD0iaHRtbF80MjJlOGEwNDZlNTk0MmI0YTc5YWQyMWE3NTY3YjY1NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnN2aWV3IENsdXN0ZXIgMi4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzE0MTNiODcxNTI5ZDQxMGI4NTlmNGFmMmNlZTM0MjhkLnNldENvbnRlbnQoaHRtbF80MjJlOGEwNDZlNTk0MmI0YTc5YWQyMWE3NTY3YjY1NCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMjZmZjZkZmEwNDAyNGYyMjg1YTI2YzY1M2FlM2VlZTMuYmluZFBvcHVwKHBvcHVwXzE0MTNiODcxNTI5ZDQxMGI4NTlmNGFmMmNlZTM0MjhkKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MGQwNTVhYmQ5Yzg0NDgxOTZkODdlZGU0OGE3N2Q5ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODAyMDUsIC03OS4zODg3OTAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2JkNmUwMzBmY2VlMzRmYjliNzUxMTdlZjg2M2U4MGE2ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8xOTlkNmI0MGZiMmM0MjNjODEyMDg1MzFkMzlmMTE4ZiA9ICQoYDxkaXYgaWQ9Imh0bWxfMTk5ZDZiNDBmYjJjNDIzYzgxMjA4NTMxZDM5ZjExOGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxhd3JlbmNlIFBhcmsgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYmQ2ZTAzMGZjZWUzNGZiOWI3NTExN2VmODYzZTgwYTYuc2V0Q29udGVudChodG1sXzE5OWQ2YjQwZmIyYzQyM2M4MTIwODUzMWQzOWYxMThmKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl85MGQwNTVhYmQ5Yzg0NDgxOTZkODdlZGU0OGE3N2Q5ZC5iaW5kUG9wdXAocG9wdXBfYmQ2ZTAzMGZjZWUzNGZiOWI3NTExN2VmODYzZTgwYTYpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2JiYjc3OWViMzUxYjRmYmNiMzA3OGI0MzMxYjlkNDI1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzExNjk0OCwgLTc5LjQxNjkzNTU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzBmMmEzNzc4ZjUwODRjZDJhMDI1NjFhNWIxYjM5YmJlID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kZjM2ZTQ2YWVhNTc0ZTU1YjdkZTNkZDJjNDQ5MTM5ZCA9ICQoYDxkaXYgaWQ9Imh0bWxfZGYzNmU0NmFlYTU3NGU1NWI3ZGUzZGQyYzQ0OTEzOWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VsYXduIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzBmMmEzNzc4ZjUwODRjZDJhMDI1NjFhNWIxYjM5YmJlLnNldENvbnRlbnQoaHRtbF9kZjM2ZTQ2YWVhNTc0ZTU1YjdkZTNkZDJjNDQ5MTM5ZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYmJiNzc5ZWIzNTFiNGZiY2IzMDc4YjQzMzFiOWQ0MjUuYmluZFBvcHVwKHBvcHVwXzBmMmEzNzc4ZjUwODRjZDJhMDI1NjFhNWIxYjM5YmJlKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNTVmMzlkMGJkNTI0MjAxYTE0ZjQzNGZhMjgwZThiMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MzE4NTI5OTk5OTk5LCAtNzkuNDg3MjYxOTAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MGZmYjQiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNzZiYmRlNzIyOTM3NGU1YzljNTM2ZDdkYzQyNDRkN2IgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2FhNTg2YjExNjgzNjRjMGI5N2FkYTQwMTRkOGY4YTgxID0gJChgPGRpdiBpZD0iaHRtbF9hYTU4NmIxMTY4MzY0YzBiOTdhZGE0MDE0ZDhmOGE4MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBUaGUgSnVuY3Rpb24gTm9ydGggQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNzZiYmRlNzIyOTM3NGU1YzljNTM2ZDdkYzQyNDRkN2Iuc2V0Q29udGVudChodG1sX2FhNTg2YjExNjgzNjRjMGI5N2FkYTQwMTRkOGY4YTgxKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9jNTVmMzlkMGJkNTI0MjAxYTE0ZjQzNGZhMjgwZThiMi5iaW5kUG9wdXAocG9wdXBfNzZiYmRlNzIyOTM3NGU1YzljNTM2ZDdkYzQyNDRkN2IpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJhOGU0OGI0MGViNzRlYzg4YTdkZTk5MmRhMjI0N2EyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2ODc2LCAtNzkuNTE4MTg4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZDg0NDllMmVmMzNlNDE5NWE1NjkyZjYwMWM3Mzg2YzUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzkxZjJjNGE1YzUxYzQ1ODRhNzUyZDUxYjI4ZDBhOTZjID0gJChgPGRpdiBpZD0iaHRtbF85MWYyYzRhNWM1MWM0NTg0YTc1MmQ1MWIyOGQwYTk2YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdG9uIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2Q4NDQ5ZTJlZjMzZTQxOTVhNTY5MmY2MDFjNzM4NmM1LnNldENvbnRlbnQoaHRtbF85MWYyYzRhNWM1MWM0NTg0YTc1MmQ1MWIyOGQwYTk2Yyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMmE4ZTQ4YjQwZWI3NGVjODhhN2RlOTkyZGEyMjQ3YTIuYmluZFBvcHVwKHBvcHVwX2Q4NDQ5ZTJlZjMzZTQxOTVhNTY5MmY2MDFjNzM4NmM1KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84ZjM3ZDgyNTExN2M0OGJiODIwYzE4NDRlMjhjNWMwNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1NzQwOTYsIC03OS4yNzMzMDQwMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iZjZjODg2NWUzN2Q0MTFjYmI3ODA5MzIzNjYyM2YzNyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMzg1MGUzN2Y5M2NjNDg4NzlmMjgwYjA3MzA3ZmQ4ZjAgPSAkKGA8ZGl2IGlkPSJodG1sXzM4NTBlMzdmOTNjYzQ4ODc5ZjI4MGIwNzMwN2ZkOGYwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3JzZXQgUGFyaywgV2V4Zm9yZCBIZWlnaHRzLCBTY2FyYm9yb3VnaCBUb3duIENlbnRyZSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9iZjZjODg2NWUzN2Q0MTFjYmI3ODA5MzIzNjYyM2YzNy5zZXRDb250ZW50KGh0bWxfMzg1MGUzN2Y5M2NjNDg4NzlmMjgwYjA3MzA3ZmQ4ZjApOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzhmMzdkODI1MTE3YzQ4YmI4MjBjMTg0NGUyOGM1YzA1LmJpbmRQb3B1cChwb3B1cF9iZjZjODg2NWUzN2Q0MTFjYmI3ODA5MzIzNjYyM2YzNykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzc3ZGViN2NiMWJlNDIxZmEwMGIwODk4ZTYyODE1YWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTI3NTgyOTk5OTk5OTYsIC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODAwMGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MDAwZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2M5YTcwZmJhYzA1ZTQ5NjNiM2ZjOTJhYWE5NGY1OTcxID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF82MTVkNWFiYzIyNjQ0MDNjYWZiMzIzOWZkYmZjMDc4NSA9ICQoYDxkaXYgaWQ9Imh0bWxfNjE1ZDVhYmMyMjY0NDAzY2FmYjMyMzlmZGJmYzA3ODUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPllvcmsgTWlsbHMgV2VzdCBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9jOWE3MGZiYWMwNWU0OTYzYjNmYzkyYWFhOTRmNTk3MS5zZXRDb250ZW50KGh0bWxfNjE1ZDVhYmMyMjY0NDAzY2FmYjMyMzlmZGJmYzA3ODUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzc3N2RlYjdjYjFiZTQyMWZhMDBiMDg5OGU2MjgxNWFhLmJpbmRQb3B1cChwb3B1cF9jOWE3MGZiYWMwNWU0OTYzYjNmYzkyYWFhOTRmNTk3MSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGI1MTdlMzVkOWUwNDNiNDkyYjVhNTk1N2I1MzIyMDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTI3NTExLCAtNzkuMzkwMTk3NV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF8zODUxZDAwNzhmZGE0MDQwYWM1NzExOTQyZjhmNjA4YSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMWRkZGI2MmVlZmFmNDBiZGE2YjgxOTRlN2FlZGVlNTIgPSAkKGA8ZGl2IGlkPSJodG1sXzFkZGRiNjJlZWZhZjQwYmRhNmI4MTk0ZTdhZWRlZTUyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EYXZpc3ZpbGxlIE5vcnRoIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzM4NTFkMDA3OGZkYTQwNDBhYzU3MTE5NDJmOGY2MDhhLnNldENvbnRlbnQoaHRtbF8xZGRkYjYyZWVmYWY0MGJkYTZiODE5NGU3YWVkZWU1Mik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfZGI1MTdlMzVkOWUwNDNiNDkyYjVhNTk1N2I1MzIyMDEuYmluZFBvcHVwKHBvcHVwXzM4NTFkMDA3OGZkYTQwNDBhYzU3MTE5NDJmOGY2MDhhKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82Yjc5YWYyODM2OTI0MjhlYWQ1NmY4NTgyNWM1MjRhZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5Njk0NzYsIC03OS40MTEzMDcyMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF84YTcxZjY3MmI3Mjc0YzExYjZlZTAyNDQyNzlhNTExYSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZTA2MWIzODcwYzczNDA1MzlhMzE3OWJlZTk2NjIxNWUgPSAkKGA8ZGl2IGlkPSJodG1sX2UwNjFiMzg3MGM3MzQwNTM5YTMxNzliZWU5NjYyMTVlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Gb3Jlc3QgSGlsbCBOb3J0aCAmYW1wOyBXZXN0LCBGb3Jlc3QgSGlsbCBSb2FkIFBhcmsgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfOGE3MWY2NzJiNzI3NGMxMWI2ZWUwMjQ0Mjc5YTUxMWEuc2V0Q29udGVudChodG1sX2UwNjFiMzg3MGM3MzQwNTM5YTMxNzliZWU5NjYyMTVlKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl82Yjc5YWYyODM2OTI0MjhlYWQ1NmY4NTgyNWM1MjRhZi5iaW5kUG9wdXAocG9wdXBfOGE3MWY2NzJiNzI3NGMxMWI2ZWUwMjQ0Mjc5YTUxMWEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc0MWY1YjE1YTAwZDQwMTM4MmI0ZDFkMjIzMjk2MTNkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYxNjA4MywgLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2U0ODcwMTQwZDBkNDRjNmRhZmZhNzU0OWNhN2JhZDAzID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9jMWE2YTBmYzhkN2Y0OGU1YjBiYWM2NGU0MDA4NGQ5NyA9ICQoYDxkaXYgaWQ9Imh0bWxfYzFhNmEwZmM4ZDdmNDhlNWIwYmFjNjRlNDAwODRkOTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhpZ2ggUGFyaywgVGhlIEp1bmN0aW9uIFNvdXRoIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2U0ODcwMTQwZDBkNDRjNmRhZmZhNzU0OWNhN2JhZDAzLnNldENvbnRlbnQoaHRtbF9jMWE2YTBmYzhkN2Y0OGU1YjBiYWM2NGU0MDA4NGQ5Nyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNzQxZjViMTVhMDBkNDAxMzgyYjRkMWQyMjMyOTYxM2QuYmluZFBvcHVwKHBvcHVwX2U0ODcwMTQwZDBkNDRjNmRhZmZhNzU0OWNhN2JhZDAzKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZjI2M2M4ZDI1Mjg0Mjg0YjNlZDNhOTFlYzQ3NTFiNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5NjMxOSwgLTc5LjUzMjI0MjQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODBmZmI0IiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MGZmYjQiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzQwYmFmNmNkYmU5YjRjNWI4YjNhMGRjYjNmYzdmMTlmID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF81NTg1NzRjZWM2Mjc0MzQ0ODk0NWUwMTUxYTkwOTM1NyA9ICQoYDxkaXYgaWQ9Imh0bWxfNTU4NTc0Y2VjNjI3NDM0NDg5NDVlMDE1MWE5MDkzNTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3Rtb3VudCBDbHVzdGVyIDIuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF80MGJhZjZjZGJlOWI0YzViOGIzYTBkY2IzZmM3ZjE5Zi5zZXRDb250ZW50KGh0bWxfNTU4NTc0Y2VjNjI3NDM0NDg5NDVlMDE1MWE5MDkzNTcpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzlmMjYzYzhkMjUyODQyODRiM2VkM2E5MWVjNDc1MWI2LmJpbmRQb3B1cChwb3B1cF80MGJhZjZjZGJlOWI0YzViOGIzYTBkY2IzZmM3ZjE5ZikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDEwN2I3NmQ2Y2YwNGViNzk3MDgwNWM3OGZiNjQwZWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTAwNzE1MDAwMDAwMDQsIC03OS4yOTU4NDkxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODBmZmI0IiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MGZmYjQiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzZlMWY5YThiZTc0YTRjNGE4NDM4MjI3NWU4MDEyYWNjID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9iZDkxM2JkZmQ2YzA0NmQ0OGMwYzZhMWIzN2E4ZWJjMiA9ICQoYDxkaXYgaWQ9Imh0bWxfYmQ5MTNiZGZkNmMwNDZkNDhjMGM2YTFiMzdhOGViYzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldleGZvcmQsIE1hcnl2YWxlIENsdXN0ZXIgMi4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzZlMWY5YThiZTc0YTRjNGE4NDM4MjI3NWU4MDEyYWNjLnNldENvbnRlbnQoaHRtbF9iZDkxM2JkZmQ2YzA0NmQ0OGMwYzZhMWIzN2E4ZWJjMik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMDEwN2I3NmQ2Y2YwNGViNzk3MDgwNWM3OGZiNjQwZWQuYmluZFBvcHVwKHBvcHVwXzZlMWY5YThiZTc0YTRjNGE4NDM4MjI3NWU4MDEyYWNjKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hOWVhZjVhMjI3MmY0MzAyYjQxMTZhYjFiMGYwNTE3ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4MjczNjQsIC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODBmZmI0IiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MGZmYjQiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2ViMmU3NzU2ZjE3ZjQ3MGJiN2E1YTAwNjJiMDAyMTU4ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9kM2ZiN2QyMjNmZDM0YWZhYWRjNDk5Yzg2MTVhMWU4MCA9ICQoYDxkaXYgaWQ9Imh0bWxfZDNmYjdkMjIzZmQzNGFmYWFkYzQ5OWM4NjE1YTFlODAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldpbGxvd2RhbGUsIFdpbGxvd2RhbGUgV2VzdCBDbHVzdGVyIDIuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9lYjJlNzc1NmYxN2Y0NzBiYjdhNWEwMDYyYjAwMjE1OC5zZXRDb250ZW50KGh0bWxfZDNmYjdkMjIzZmQzNGFmYWFkYzQ5OWM4NjE1YTFlODApOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2E5ZWFmNWEyMjcyZjQzMDJiNDExNmFiMWIwZjA1MTdlLmJpbmRQb3B1cChwb3B1cF9lYjJlNzc1NmYxN2Y0NzBiYjdhNWEwMDYyYjAwMjE1OCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGU1ODJjOTNkYWNiNDczODkwZDk0NzgwMzMyZTk1MmIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTUzODM0LCAtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNTdiNDg0MjA1NzY1NGQ0ZGI0NTA4OWRkZDRiNWNkNzAgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2U4ZTUzYzcxNzdjODQzN2Y5MDg4NjY2NzMzYTY3NWU5ID0gJChgPGRpdiBpZD0iaHRtbF9lOGU1M2M3MTc3Yzg0MzdmOTA4ODY2NjczM2E2NzVlOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGggVG9yb250byBXZXN0LCBMYXdyZW5jZSBQYXJrIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzU3YjQ4NDIwNTc2NTRkNGRiNDUwODlkZGQ0YjVjZDcwLnNldENvbnRlbnQoaHRtbF9lOGU1M2M3MTc3Yzg0MzdmOTA4ODY2NjczM2E2NzVlOSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMGU1ODJjOTNkYWNiNDczODkwZDk0NzgwMzMyZTk1MmIuYmluZFBvcHVwKHBvcHVwXzU3YjQ4NDIwNTc2NTRkNGRiNDUwODlkZGQ0YjVjZDcwKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZmFkOWNlMGY5ZmE0MzAzYTM0NWYxYTM3ODRmMmRhZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MjcwOTcsIC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF80YzZiMzI4ZmRiNzQ0ZGM1OGZhMTE2YTMxMmY1N2VjNSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYjQ5ZDU5NDU0OTA4NDM4OWI3NmFhNTA4MmU3NDQ1MjUgPSAkKGA8ZGl2IGlkPSJodG1sX2I0OWQ1OTQ1NDkwODQzODliNzZhYTUwODJlNzQ0NTI1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgQW5uZXgsIE5vcnRoIE1pZHRvd24sIFlvcmt2aWxsZSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF80YzZiMzI4ZmRiNzQ0ZGM1OGZhMTE2YTMxMmY1N2VjNS5zZXRDb250ZW50KGh0bWxfYjQ5ZDU5NDU0OTA4NDM4OWI3NmFhNTA4MmU3NDQ1MjUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2VmYWQ5Y2UwZjlmYTQzMDNhMzQ1ZjFhMzc4NGYyZGFkLmJpbmRQb3B1cChwb3B1cF80YzZiMzI4ZmRiNzQ0ZGM1OGZhMTE2YTMxMmY1N2VjNSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTVkNmQwNDY3MTc1NDFhMTk3Nzg0MWUxZWY4N2EzZjUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg5NTk3LCAtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzU2NDQxOTViM2FhYzRmOGQ5NDA0Mjk4MzA3NjU4OTM0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8yZTE1MmEzMDQ3YjM0NTJjYTg2OWUyNTYzNGMyYzUwMCA9ICQoYDxkaXYgaWQ9Imh0bWxfMmUxNTJhMzA0N2IzNDUyY2E4NjllMjU2MzRjMmM1MDAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmtkYWxlLCBSb25jZXN2YWxsZXMgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNTY0NDE5NWIzYWFjNGY4ZDk0MDQyOTgzMDc2NTg5MzQuc2V0Q29udGVudChodG1sXzJlMTUyYTMwNDdiMzQ1MmNhODY5ZTI1NjM0YzJjNTAwKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9lNWQ2ZDA0NjcxNzU0MWExOTc3ODQxZTFlZjg3YTNmNS5iaW5kUG9wdXAocG9wdXBfNTY0NDE5NWIzYWFjNGY4ZDk0MDQyOTgzMDc2NTg5MzQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE0NmQ0YjM2NDZkMTRkNGE4NjBmODJmNTUwMmE4ZGNlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2OTY1NiwgLTc5LjYxNTgxODk5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2I1M2ZlYmYxNjViNzRiMWM4YmQ1NTE2Yjk3MzVjNTVjID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF82OGNjYjc5YzVlN2Y0MjE1Yjc5Njg1OWNlZGQ3ZjI4NyA9ICQoYDxkaXYgaWQ9Imh0bWxfNjhjY2I3OWM1ZTdmNDIxNWI3OTY4NTljZWRkN2YyODciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhbmFkYSBQb3N0IEdhdGV3YXkgUHJvY2Vzc2luZyBDZW50cmUgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYjUzZmViZjE2NWI3NGIxYzhiZDU1MTZiOTczNWM1NWMuc2V0Q29udGVudChodG1sXzY4Y2NiNzljNWU3ZjQyMTViNzk2ODU5Y2VkZDdmMjg3KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8xNDZkNGIzNjQ2ZDE0ZDRhODYwZjgyZjU1MDJhOGRjZS5iaW5kUG9wdXAocG9wdXBfYjUzZmViZjE2NWI3NGIxYzhiZDU1MTZiOTczNWM1NWMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRjMTIwNzhhNjkyZTQ2ZDViNzc5ZDQ0YzQ2ODBmM2QyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg4OTA1NCwgLTc5LjU1NDcyNDQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODBmZmI0IiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MGZmYjQiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzgxZTE4NWQ4YjM1NjQxYmFhMjA0NGZkM2FjNTlkZmM5ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8yYzNmYjM0MWVkZWY0MmFhYjVhYmRhNTdhOTllNjY5ZCA9ICQoYDxkaXYgaWQ9Imh0bWxfMmMzZmIzNDFlZGVmNDJhYWI1YWJkYTU3YTk5ZTY2OWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktpbmdzdmlldyBWaWxsYWdlLCBTdC4gUGhpbGxpcHMsIE1hcnRpbiBHcm92ZSBHYXJkZW5zLCBSaWNodmlldyBHYXJkZW5zIENsdXN0ZXIgMi4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzgxZTE4NWQ4YjM1NjQxYmFhMjA0NGZkM2FjNTlkZmM5LnNldENvbnRlbnQoaHRtbF8yYzNmYjM0MWVkZWY0MmFhYjVhYmRhNTdhOTllNjY5ZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNGMxMjA3OGE2OTJlNDZkNWI3NzlkNDRjNDY4MGYzZDIuYmluZFBvcHVwKHBvcHVwXzgxZTE4NWQ4YjM1NjQxYmFhMjA0NGZkM2FjNTlkZmM5KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80Zjc2NTE3Njg4MjE0NTFmYjczNzZmYjdlNTdjZjE4ZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc5NDIwMDMsIC03OS4yNjIwMjk0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9kNjUzZmIxNjRiNmU0NjQ5YmIwNjRkYWU0MzkxMWFiYyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMGI5OWZmMTk1ZWM0NGYxMWI5YmU3NGVmOGQ2NmEwZGYgPSAkKGA8ZGl2IGlkPSJodG1sXzBiOTlmZjE5NWVjNDRmMTFiOWJlNzRlZjhkNjZhMGRmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BZ2luY291cnQgQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZDY1M2ZiMTY0YjZlNDY0OWJiMDY0ZGFlNDM5MTFhYmMuc2V0Q29udGVudChodG1sXzBiOTlmZjE5NWVjNDRmMTFiOWJlNzRlZjhkNjZhMGRmKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl80Zjc2NTE3Njg4MjE0NTFmYjczNzZmYjdlNTdjZjE4Zi5iaW5kUG9wdXAocG9wdXBfZDY1M2ZiMTY0YjZlNDY0OWJiMDY0ZGFlNDM5MTFhYmMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZhOWFiY2ExOWI3NjRiNGQ4NTA1ZWM1NDZmM2JjNTkzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA0MzI0NCwgLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYmM1Njc5ZTAzYTUzNGE1ZjlkN2M4MjJhNGFjMDUzOTEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2IxNTdkNjdjOWIzNjQ3YWFiY2YwNjg3ZGZkODhmODUzID0gJChgPGRpdiBpZD0iaHRtbF9iMTU3ZDY3YzliMzY0N2FhYmNmMDY4N2RmZDg4Zjg1MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9iYzU2NzllMDNhNTM0YTVmOWQ3YzgyMmE0YWMwNTM5MS5zZXRDb250ZW50KGh0bWxfYjE1N2Q2N2M5YjM2NDdhYWJjZjA2ODdkZmQ4OGY4NTMpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzZhOWFiY2ExOWI3NjRiNGQ4NTA1ZWM1NDZmM2JjNTkzLmJpbmRQb3B1cChwb3B1cF9iYzU2NzllMDNhNTM0YTVmOWQ3YzgyMmE0YWMwNTM5MSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTk5NmMwNjllODRlNDM2N2I5ZDE2YWU4YTM2MTBlOWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LCAtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF83NDA3N2VkMWJiNDU0Y2QxYmJjNDEyZDgwOWQxMTdhMSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNTEyZmVkZTdiMDA3NDBlOThlMGU0ZDVkOTUxM2ZhNTcgPSAkKGA8ZGl2IGlkPSJodG1sXzUxMmZlZGU3YjAwNzQwZTk4ZTBlNGQ1ZDk1MTNmYTU3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIFRvcm9udG8sIEhhcmJvcmQgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNzQwNzdlZDFiYjQ1NGNkMWJiYzQxMmQ4MDlkMTE3YTEuc2V0Q29udGVudChodG1sXzUxMmZlZGU3YjAwNzQwZTk4ZTBlNGQ1ZDk1MTNmYTU3KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl81OTk2YzA2OWU4NGU0MzY3YjlkMTZhZThhMzYxMGU5Yi5iaW5kUG9wdXAocG9wdXBfNzQwNzdlZDFiYjQ1NGNkMWJiYzQxMmQ4MDlkMTE3YTEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkyOTQ5YjEzNjI2YTRiNDRiNGJiM2I0Y2UxMTRlNTQxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNTcwNiwgLTc5LjQ4NDQ0OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfM2YwMWQ3MTEwNTRhNGYyNmJhNTA4ODdhNjlhYTQ2NTkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzk0MWU2ZDM1YmFiNDQ5MWRhZTcyMDAxN2Y3ZGIwNjQ2ID0gJChgPGRpdiBpZD0iaHRtbF85NDFlNmQzNWJhYjQ0OTFkYWU3MjAwMTdmN2RiMDY0NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBTd2Fuc2VhIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzNmMDFkNzExMDU0YTRmMjZiYTUwODg3YTY5YWE0NjU5LnNldENvbnRlbnQoaHRtbF85NDFlNmQzNWJhYjQ0OTFkYWU3MjAwMTdmN2RiMDY0Nik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfOTI5NDliMTM2MjZhNGI0NGI0YmIzYjRjZTExNGU1NDEuYmluZFBvcHVwKHBvcHVwXzNmMDFkNzExMDU0YTRmMjZiYTUwODg3YTY5YWE0NjU5KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yMDY0NDQ4NjdmMWI0ZTg4OTA2MDg0ZWNkMDk5MmExOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4MTYzNzUsIC03OS4zMDQzMDIxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODBmZmI0IiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MGZmYjQiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzA2Mjc3Mzc0YjgwZjRjNzQ5MTdlYzMxMzFhODFmNWYyID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9iZDIyMzQ3ZjNhNDE0YTlhYjM5YmQ3NzQyYjRiZWM4YiA9ICQoYDxkaXYgaWQ9Imh0bWxfYmQyMjM0N2YzYTQxNGE5YWIzOWJkNzc0MmI0YmVjOGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNsYXJrcyBDb3JuZXJzLCBUYW0gTyYjMzk7U2hhbnRlciwgU3VsbGl2YW4gQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfMDYyNzczNzRiODBmNGM3NDkxN2VjMzEzMWE4MWY1ZjIuc2V0Q29udGVudChodG1sX2JkMjIzNDdmM2E0MTRhOWFiMzliZDc3NDJiNGJlYzhiKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8yMDY0NDQ4NjdmMWI0ZTg4OTA2MDg0ZWNkMDk5MmExOS5iaW5kUG9wdXAocG9wdXBfMDYyNzczNzRiODBmNGM3NDkxN2VjMzEzMWE4MWY1ZjIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E1NDg0MTljYWNhZTQ0NjM4YTkyYTIzYjNhZjk4NTAxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5NTc0MywgLTc5LjM4MzE1OTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzZlYzE2ZGQ2NmUwNDQ2ZTFhYmNiMDY3MGMwZDMwZjljID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9iMjBjOTY3ZmE1MTk0ZTdlYTUxMTQ4MGE1NWJiYjAyOCA9ICQoYDxkaXYgaWQ9Imh0bWxfYjIwYzk2N2ZhNTE5NGU3ZWE1MTE0ODBhNTViYmIwMjgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vb3JlIFBhcmssIFN1bW1lcmhpbGwgRWFzdCBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF82ZWMxNmRkNjZlMDQ0NmUxYWJjYjA2NzBjMGQzMGY5Yy5zZXRDb250ZW50KGh0bWxfYjIwYzk2N2ZhNTE5NGU3ZWE1MTE0ODBhNTViYmIwMjgpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2E1NDg0MTljYWNhZTQ0NjM4YTkyYTIzYjNhZjk4NTAxLmJpbmRQb3B1cChwb3B1cF82ZWMxNmRkNjZlMDQ0NmUxYWJjYjA2NzBjMGQzMGY5YykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmZmZGNjZGRlZTc2NDllNmJlZDlmNjM0M2EzMTJiNmIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LCAtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82YjQ4ZDliZTQyYmY0NjAxYWM1YTg0NThmMWY5M2QwYiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMTRmMzIyYmUwMmQ1NGI1ZjkyNjViNGExYTdmNmMwZDEgPSAkKGA8ZGl2IGlkPSJodG1sXzE0ZjMyMmJlMDJkNTRiNWY5MjY1YjRhMWE3ZjZjMGQxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZW5zaW5ndG9uIE1hcmtldCwgQ2hpbmF0b3duLCBHcmFuZ2UgUGFyayBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF82YjQ4ZDliZTQyYmY0NjAxYWM1YTg0NThmMWY5M2QwYi5zZXRDb250ZW50KGh0bWxfMTRmMzIyYmUwMmQ1NGI1ZjkyNjViNGExYTdmNmMwZDEpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2ZmZmRjY2RkZWU3NjQ5ZTZiZWQ5ZjYzNDNhMzEyYjZiLmJpbmRQb3B1cChwb3B1cF82YjQ4ZDliZTQyYmY0NjAxYWM1YTg0NThmMWY5M2QwYikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDRhMjdiM2Y1ODA4NGFkNWJlZmY4OTk5N2E0ZjU4ZWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My44MTUyNTIyLCAtNzkuMjg0NTc3Ml0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9jM2NiZDNlNWMyMDg0MTJiOGU1MWQ3ZjllODk5YzM4YSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZjFjNDA5YzNmZTNlNDNhZWExNmE2Y2NlYWVmZmFkNDQgPSAkKGA8ZGl2IGlkPSJodG1sX2YxYzQwOWMzZmUzZTQzYWVhMTZhNmNjZWFlZmZhZDQ0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NaWxsaWtlbiwgQWdpbmNvdXJ0IE5vcnRoLCBTdGVlbGVzIEVhc3QsIEwmIzM5O0Ftb3JlYXV4IEVhc3QgQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYzNjYmQzZTVjMjA4NDEyYjhlNTFkN2Y5ZTg5OWMzOGEuc2V0Q29udGVudChodG1sX2YxYzQwOWMzZmUzZTQzYWVhMTZhNmNjZWFlZmZhZDQ0KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9kNGEyN2IzZjU4MDg0YWQ1YmVmZjg5OTk3YTRmNThlZi5iaW5kUG9wdXAocG9wdXBfYzNjYmQzZTVjMjA4NDEyYjhlNTFkN2Y5ZTg5OWMzOGEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NiMmM4ZDAyMDUyOTQ4MGU4ZWQwZjEwNmExOTJmOGJlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg2NDEyMjk5OTk5OTksIC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2M2ZDlkMjliYjA0ODQ0Yjc5ZGYwY2QxN2FmNjBmODVjID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF83NGY5YmRjZmE5MDc0ZDg1YmFlMzI4ZmJjODhmZDBmMiA9ICQoYDxkaXYgaWQ9Imh0bWxfNzRmOWJkY2ZhOTA3NGQ4NWJhZTMyOGZiYzg4ZmQwZjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN1bW1lcmhpbGwgV2VzdCwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBGb3Jlc3QgSGlsbCBTRSwgRGVlciBQYXJrIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2M2ZDlkMjliYjA0ODQ0Yjc5ZGYwY2QxN2FmNjBmODVjLnNldENvbnRlbnQoaHRtbF83NGY5YmRjZmE5MDc0ZDg1YmFlMzI4ZmJjODhmZDBmMik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfY2IyYzhkMDIwNTI5NDgwZThlZDBmMTA2YTE5MmY4YmUuYmluZFBvcHVwKHBvcHVwX2M2ZDlkMjliYjA0ODQ0Yjc5ZGYwY2QxN2FmNjBmODVjKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84Yzc3YzIzNGRhMTk0MzQ4OGJmZjA1YzNjN2E1MTVlNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsIC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2Y0NDgxMGM4YTc0NzRiYjdiZGU1NjNkMTgyMjc5YmMzID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF83N2JlZDQzMWI5ZDU0NjQ0OTY0NDE3MzRlYTk3ODE4MCA9ICQoYDxkaXYgaWQ9Imh0bWxfNzdiZWQ0MzFiOWQ1NDY0NDk2NDQxNzM0ZWE5NzgxODAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNOIFRvd2VyLCBLaW5nIGFuZCBTcGFkaW5hLCBSYWlsd2F5IExhbmRzLCBIYXJib3VyZnJvbnQgV2VzdCwgQmF0aHVyc3QgUXVheSwgU291dGggTmlhZ2FyYSwgSXNsYW5kIGFpcnBvcnQgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZjQ0ODEwYzhhNzQ3NGJiN2JkZTU2M2QxODIyNzliYzMuc2V0Q29udGVudChodG1sXzc3YmVkNDMxYjlkNTQ2NDQ5NjQ0MTczNGVhOTc4MTgwKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl84Yzc3YzIzNGRhMTk0MzQ4OGJmZjA1YzNjN2E1MTVlNy5iaW5kUG9wdXAocG9wdXBfZjQ0ODEwYzhhNzQ3NGJiN2JkZTU2M2QxODIyNzliYzMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Q4YjcyNTAwY2YzYjRmNDZiNGRmZmE4ODMwYTk3YzZjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjA1NjQ2NiwgLTc5LjUwMTMyMDcwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODAwMGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MDAwZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzNkZTE3M2JhMjMwNjRhNDRiMzU2MjI0NDQ0YzQzMzQ4ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8xZWQ0Nzg0MTU4N2U0OTdmODhmODYxYmYzYWI3ODg5OCA9ICQoYDxkaXYgaWQ9Imh0bWxfMWVkNDc4NDE1ODdlNDk3Zjg4Zjg2MWJmM2FiNzg4OTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5ldyBUb3JvbnRvLCBNaW1pY28gU291dGgsIEh1bWJlciBCYXkgU2hvcmVzIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzNkZTE3M2JhMjMwNjRhNDRiMzU2MjI0NDQ0YzQzMzQ4LnNldENvbnRlbnQoaHRtbF8xZWQ0Nzg0MTU4N2U0OTdmODhmODYxYmYzYWI3ODg5OCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfZDhiNzI1MDBjZjNiNGY0NmI0ZGZmYTg4MzBhOTdjNmMuYmluZFBvcHVwKHBvcHVwXzNkZTE3M2JhMjMwNjRhNDRiMzU2MjI0NDQ0YzQzMzQ4KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kY2JlN2JhNTZkNjQ0MzM1OWZhMDJlZDRiM2QwNjZkZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTQxNjM5OTk5OTk5NiwgLTc5LjU4ODQzNjldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MGZmYjQiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMWU0Njk1ZTk1NTU5NGU2OGIyYTcxNWM3ODhjMTRkZmUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzJlMWRlZjNmMWJjNjQ2OTA4NDEzMWUzMmQwY2IwYWM2ID0gJChgPGRpdiBpZD0iaHRtbF8yZTFkZWYzZjFiYzY0NjkwODQxMzFlMzJkMGNiMGFjNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U291dGggU3RlZWxlcywgU2lsdmVyc3RvbmUsIEh1bWJlcmdhdGUsIEphbWVzdG93biwgTW91bnQgT2xpdmUsIEJlYXVtb25kIEhlaWdodHMsIFRoaXN0bGV0b3duLCBBbGJpb24gR2FyZGVucyBDbHVzdGVyIDIuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8xZTQ2OTVlOTU1NTk0ZTY4YjJhNzE1Yzc4OGMxNGRmZS5zZXRDb250ZW50KGh0bWxfMmUxZGVmM2YxYmM2NDY5MDg0MTMxZTMyZDBjYjBhYzYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2RjYmU3YmE1NmQ2NDQzMzU5ZmEwMmVkNGIzZDA2NmRlLmJpbmRQb3B1cChwb3B1cF8xZTQ2OTVlOTU1NTk0ZTY4YjJhNzE1Yzc4OGMxNGRmZSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTVmZTJmYTJmZTlkNGJkYWJiODI3YTgxY2QzODFjNTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43OTk1MjUyMDAwMDAwMDUsIC03OS4zMTgzODg3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODBmZmI0IiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MGZmYjQiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzBjZmM4N2Y3OGE3YjQ2NzZiYTJkZTMwYmI4YThmZWFjID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF80NDdhMTNmYjMwZTI0YzYxYWExMTZhYTcwZThiZTE1NSA9ICQoYDxkaXYgaWQ9Imh0bWxfNDQ3YTEzZmIzMGUyNGM2MWFhMTE2YWE3MGU4YmUxNTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0ZWVsZXMgV2VzdCwgTCYjMzk7QW1vcmVhdXggV2VzdCBDbHVzdGVyIDIuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8wY2ZjODdmNzhhN2I0Njc2YmEyZGUzMGJiOGE4ZmVhYy5zZXRDb250ZW50KGh0bWxfNDQ3YTEzZmIzMGUyNGM2MWFhMTE2YWE3MGU4YmUxNTUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzk1ZmUyZmEyZmU5ZDRiZGFiYjgyN2E4MWNkMzgxYzU2LmJpbmRQb3B1cChwb3B1cF8wY2ZjODdmNzhhN2I0Njc2YmEyZGUzMGJiOGE4ZmVhYykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2I5NWEwZTE3Zjk5NDk2NWI5NmMyMTMwYmZmYmNhNWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NjI2LCAtNzkuMzc3NTI5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNWZjYjdmNGUwZjBkNDAxZGFiMzdkMjdiMGM2NjIzZWUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzA5ZWMyMmVlM2ZhZTQyNTE4NDJjYjY3ZmQyNzZhODJlID0gJChgPGRpdiBpZD0iaHRtbF8wOWVjMjJlZTNmYWU0MjUxODQyY2I2N2ZkMjc2YTgyZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWRhbGUgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNWZjYjdmNGUwZjBkNDAxZGFiMzdkMjdiMGM2NjIzZWUuc2V0Q29udGVudChodG1sXzA5ZWMyMmVlM2ZhZTQyNTE4NDJjYjY3ZmQyNzZhODJlKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8zYjk1YTBlMTdmOTk0OTY1Yjk2YzIxMzBiZmZiY2E1Zi5iaW5kUG9wdXAocG9wdXBfNWZjYjdmNGUwZjBkNDAxZGFiMzdkMjdiMGM2NjIzZWUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzEyYTVmODFkNTc1MjRjMjI4N2NlNDc3ODAyNjI1ZjlmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwgLTc5LjM3NDg0NTk5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzQ2ODdhZWRlZmY4MjQ2MTFiMTViYWU1NmRjM2YxZDk2ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9iMjIzNTA1MzQ5NGQ0ZDY4YjI2N2FkYWVlNTBlZjM0MiA9ICQoYDxkaXYgaWQ9Imh0bWxfYjIyMzUwNTM0OTRkNGQ2OGIyNjdhZGFlZTUwZWYzNDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0biBBIFBPIEJveGVzIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzQ2ODdhZWRlZmY4MjQ2MTFiMTViYWU1NmRjM2YxZDk2LnNldENvbnRlbnQoaHRtbF9iMjIzNTA1MzQ5NGQ0ZDY4YjI2N2FkYWVlNTBlZjM0Mik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMTJhNWY4MWQ1NzUyNGMyMjg3Y2U0Nzc4MDI2MjVmOWYuYmluZFBvcHVwKHBvcHVwXzQ2ODdhZWRlZmY4MjQ2MTFiMTViYWU1NmRjM2YxZDk2KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNTYwYjY1MDJiMDA0NmM0YmFhOTE1NzQ5MWMwOTA0NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYwMjQxMzcwMDAwMDAxLCAtNzkuNTQzNDg0MDk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MGZmYjQiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMmEyN2FlMGExMzk0NDFlYmEzMmE3NTNhZmI4Njk3ZTIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2Q0NzgyY2ZmMTBhNzRmMGE4NWQzMjk2NGYxODhhYjZkID0gJChgPGRpdiBpZD0iaHRtbF9kNDc4MmNmZjEwYTc0ZjBhODVkMzI5NjRmMTg4YWI2ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QWxkZXJ3b29kLCBMb25nIEJyYW5jaCBDbHVzdGVyIDIuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8yYTI3YWUwYTEzOTQ0MWViYTMyYTc1M2FmYjg2OTdlMi5zZXRDb250ZW50KGh0bWxfZDQ3ODJjZmYxMGE3NGYwYTg1ZDMyOTY0ZjE4OGFiNmQpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2Y1NjBiNjUwMmIwMDQ2YzRiYWE5MTU3NDkxYzA5MDQ3LmJpbmRQb3B1cChwb3B1cF8yYTI3YWUwYTEzOTQ0MWViYTMyYTc1M2FmYjg2OTdlMikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGZkNzU2NDk1YzU5NDRlMWI1M2FmMDk4NjVlMGUyMzkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDY3NDgyOTk5OTk5OTQsIC03OS41OTQwNTQ0XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2M5Y2VjOWFmOWY1NjQ3ZDBiODIxMjg0ZjVkNjE1ZTZmID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9lZjg5MWVkMzc4MTc0MWVjYmFhNjg4NWNhOWU5NDU5YiA9ICQoYDxkaXYgaWQ9Imh0bWxfZWY4OTFlZDM3ODE3NDFlY2JhYTY4ODVjYTllOTQ1OWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRod2VzdCwgV2VzdCBIdW1iZXIgLSBDbGFpcnZpbGxlIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2M5Y2VjOWFmOWY1NjQ3ZDBiODIxMjg0ZjVkNjE1ZTZmLnNldENvbnRlbnQoaHRtbF9lZjg5MWVkMzc4MTc0MWVjYmFhNjg4NWNhOWU5NDU5Yik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfOGZkNzU2NDk1YzU5NDRlMWI1M2FmMDk4NjVlMGUyMzkuYmluZFBvcHVwKHBvcHVwX2M5Y2VjOWFmOWY1NjQ3ZDBiODIxMjg0ZjVkNjE1ZTZmKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMTQ4YjQ2MjRkODU0MWY3YWNkMWI5M2I5ZjQ2MjExNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywgLTc5LjM2NzY3NTNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZmFhZjM2NTllZWQ1NGNkNWE4OTIwMTFiODE4OGE2YjEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzc4NWRjYmI0YzAxMTQwYmNiMDdmOGE5ODk4YTExN2IyID0gJChgPGRpdiBpZD0iaHRtbF83ODVkY2JiNGMwMTE0MGJjYjA3ZjhhOTg5OGExMTdiMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24sIENhYmJhZ2V0b3duIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2ZhYWYzNjU5ZWVkNTRjZDVhODkyMDExYjgxODhhNmIxLnNldENvbnRlbnQoaHRtbF83ODVkY2JiNGMwMTE0MGJjYjA3ZjhhOTg5OGExMTdiMik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYTE0OGI0NjI0ZDg1NDFmN2FjZDFiOTNiOWY0NjIxMTYuYmluZFBvcHVwKHBvcHVwX2ZhYWYzNjU5ZWVkNTRjZDVhODkyMDExYjgxODhhNmIxKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81M2RkMjA3YjcxZjI0YjUzYjE1YzU5ZjQxZWZmNTQ1YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODQyOTIsIC03OS4zODIyODAyXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzgwOTk1YWJmNmFkYzQyNjZiNWZjNmM5MTllY2EzZWExID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF85YjUwOGYwMTQ2Mjk0MTk2YmYzODExMzljNTJhNWI1ZiA9ICQoYDxkaXYgaWQ9Imh0bWxfOWI1MDhmMDE0NjI5NDE5NmJmMzgxMTM5YzUyYTViNWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZpcnN0IENhbmFkaWFuIFBsYWNlLCBVbmRlcmdyb3VuZCBjaXR5IENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzgwOTk1YWJmNmFkYzQyNjZiNWZjNmM5MTllY2EzZWExLnNldENvbnRlbnQoaHRtbF85YjUwOGYwMTQ2Mjk0MTk2YmYzODExMzljNTJhNWI1Zik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNTNkZDIwN2I3MWYyNGI1M2IxNWM1OWY0MWVmZjU0NWIuYmluZFBvcHVwKHBvcHVwXzgwOTk1YWJmNmFkYzQyNjZiNWZjNmM5MTllY2EzZWExKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lM2ExYTE4Zjc3Yjk0OTAxYjVmYjgyYmEwMzk4NmE0MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzY1MzYwMDAwMDAwNSwgLTc5LjUwNjk0MzZdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiNmZmIzNjAiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYTQ1ZjUxNzM1MDY1NDNiMTkxNWRlNDZiZGE1YjQ2ZjEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzc1NTc3YTc5NDg2NjQ2M2RhMjI4OTI0MTVmZGY2NjkyID0gJChgPGRpdiBpZD0iaHRtbF83NTU3N2E3OTQ4NjY0NjNkYTIyODkyNDE1ZmRmNjY5MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEtpbmdzd2F5LCBNb250Z29tZXJ5IFJvYWQsIE9sZCBNaWxsIE5vcnRoIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2E0NWY1MTczNTA2NTQzYjE5MTVkZTQ2YmRhNWI0NmYxLnNldENvbnRlbnQoaHRtbF83NTU3N2E3OTQ4NjY0NjNkYTIyODkyNDE1ZmRmNjY5Mik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfZTNhMWExOGY3N2I5NDkwMWI1ZmI4MmJhMDM5ODZhNDMuYmluZFBvcHVwKHBvcHVwX2E0NWY1MTczNTA2NTQzYjE5MTVkZTQ2YmRhNWI0NmYxKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xYmZjM2I5N2IwOGE0ZmNlOTUzMTFkYjI5ZTg4MGQyOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2NTg1OTksIC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9jMDgzZDliNzk3MmE0MmVlYjFlOTNjNTEyMWY0ZjIxNiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMmMzNTkwMzkxY2Y2NGQxM2FmNjFiY2NjZTdiOTE1MzEgPSAkKGA8ZGl2IGlkPSJodG1sXzJjMzU5MDM5MWNmNjRkMTNhZjYxYmNjY2U3YjkxNTMxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaHVyY2ggYW5kIFdlbGxlc2xleSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9jMDgzZDliNzk3MmE0MmVlYjFlOTNjNTEyMWY0ZjIxNi5zZXRDb250ZW50KGh0bWxfMmMzNTkwMzkxY2Y2NGQxM2FmNjFiY2NjZTdiOTE1MzEpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzFiZmMzYjk3YjA4YTRmY2U5NTMxMWRiMjllODgwZDI5LmJpbmRQb3B1cChwb3B1cF9jMDgzZDliNzk3MmE0MmVlYjFlOTNjNTEyMWY0ZjIxNikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjAyNzM2YjMzZjA0NDEzZWI4YmZhZjA0MGNlZDU2Y2QgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI3NDM5LCAtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjZmZiMzYwIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiNmZmIzNjAiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2M0NmMyZTZjN2M5NGI1NmFjZDk1ZWMyY2ZiNjlkNWMpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2FiMWMzZDNjYzUwMjQ5NDhhMThkYjJlMjNmMTdmYmMzID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF85NDQ3ZTk1NjBhYzc0ZWU2ODAyZjVjZDE1ODFlZTE4OSA9ICQoYDxkaXYgaWQ9Imh0bWxfOTQ0N2U5NTYwYWM3NGVlNjgwMmY1Y2QxNTgxZWUxODkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJ1c2luZXNzIHJlcGx5IG1haWwgUHJvY2Vzc2luZyBDZW50cmUsIFNvdXRoIENlbnRyYWwgTGV0dGVyIFByb2Nlc3NpbmcgUGxhbnQgVG9yb250byBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9hYjFjM2QzY2M1MDI0OTQ4YTE4ZGIyZTIzZjE3ZmJjMy5zZXRDb250ZW50KGh0bWxfOTQ0N2U5NTYwYWM3NGVlNjgwMmY1Y2QxNTgxZWUxODkpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2IwMjczNmIzM2YwNDQxM2ViOGJmYWYwNDBjZWQ1NmNkLmJpbmRQb3B1cChwb3B1cF9hYjFjM2QzY2M1MDI0OTQ4YTE4ZGIyZTIzZjE3ZmJjMykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzVjMWEzNTA3ZTc0NDcxZjkxMjE1MTlmYzA1NDg4MmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzYyNTc5LCAtNzkuNDk4NTA5MDk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MDAwZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zYzQ2YzJlNmM3Yzk0YjU2YWNkOTVlYzJjZmI2OWQ1Yyk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNGZmNWY3Y2ViNDNmNDQ1MzlhNTYyNzQ4YTY2MTYzYTggPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzIwNWUxMTVhZTBhMDQwYjlhMTEyYzYxOTVhN2FjYmNmID0gJChgPGRpdiBpZD0iaHRtbF8yMDVlMTE1YWUwYTA0MGI5YTExMmM2MTk1YTdhY2JjZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+T2xkIE1pbGwgU291dGgsIEtpbmcmIzM5O3MgTWlsbCBQYXJrLCBTdW5ueWxlYSwgSHVtYmVyIEJheSwgTWltaWNvIE5FLCBUaGUgUXVlZW5zd2F5IEVhc3QsIFJveWFsIFlvcmsgU291dGggRWFzdCwgS2luZ3N3YXkgUGFyayBTb3V0aCBFYXN0IENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzRmZjVmN2NlYjQzZjQ0NTM5YTU2Mjc0OGE2NjE2M2E4LnNldENvbnRlbnQoaHRtbF8yMDVlMTE1YWUwYTA0MGI5YTExMmM2MTk1YTdhY2JjZik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMzVjMWEzNTA3ZTc0NDcxZjkxMjE1MTlmYzA1NDg4MmMuYmluZFBvcHVwKHBvcHVwXzRmZjVmN2NlYjQzZjQ0NTM5YTU2Mjc0OGE2NjE2M2E4KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hYThlODUyZDZhODQ0YmMzYjdkZjhjNzcxZWIyZGE0YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODg0MDgsIC03OS41MjA5OTk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmYjM2MCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzNjNDZjMmU2YzdjOTRiNTZhY2Q5NWVjMmNmYjY5ZDVjKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9jYWNjMmRhZmFmMzk0ZjdkYWM0Njg4MjRkYTg3ZDZiOSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfYmU4MjVkMjMxMzc5NGY4NmJkNTU5OWQxZTA5YjE4NWQgPSAkKGA8ZGl2IGlkPSJodG1sX2JlODI1ZDIzMTM3OTRmODZiZDU1OTlkMWUwOWIxODVkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NaW1pY28gTlcsIFRoZSBRdWVlbnN3YXkgV2VzdCwgU291dGggb2YgQmxvb3IsIEtpbmdzd2F5IFBhcmsgU291dGggV2VzdCwgUm95YWwgWW9yayBTb3V0aCBXZXN0IENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2NhY2MyZGFmYWYzOTRmN2RhYzQ2ODgyNGRhODdkNmI5LnNldENvbnRlbnQoaHRtbF9iZTgyNWQyMzEzNzk0Zjg2YmQ1NTk5ZDFlMDliMTg1ZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYWE4ZTg1MmQ2YTg0NGJjM2I3ZGY4Yzc3MWViMmRhNGIuYmluZFBvcHVwKHBvcHVwX2NhY2MyZGFmYWYzOTRmN2RhYzQ2ODgyNGRhODdkNmI5KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKPC9zY3JpcHQ+ onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### __*Cluster 1*__


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
      <th>11th Most Common Venue</th>
      <th>12th Most Common Venue</th>
      <th>13th Most Common Venue</th>
      <th>14th Most Common Venue</th>
      <th>15th Most Common Venue</th>
      <th>16th Most Common Venue</th>
      <th>17th Most Common Venue</th>
      <th>18th Most Common Venue</th>
      <th>19th Most Common Venue</th>
      <th>20th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Etobicoke</td>
      <td>0.0</td>
      <td>Pharmacy</td>
      <td>Convenience Store</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Playground</td>
      <td>Golf Course</td>
      <td>Shopping Mall</td>
      <td>Park</td>
      <td>Café</td>
      <td>Grocery Store</td>
      <td>Electronics Store</td>
      <td>Drugstore</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Doner Restaurant</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Scarborough</td>
      <td>0.0</td>
      <td>Italian Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Playground</td>
      <td>Burger Joint</td>
      <td>Park</td>
      <td>Zoo</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Field</td>
      <td>Fast Food Restaurant</td>
      <td>Drugstore</td>
      <td>Filipino Restaurant</td>
      <td>Fireworks Store</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fish Market</td>
    </tr>
    <tr>
      <th>50</th>
      <td>North York</td>
      <td>0.0</td>
      <td>Italian Restaurant</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Medical Center</td>
      <td>Pizza Place</td>
      <td>Shopping Mall</td>
      <td>Electronics Store</td>
      <td>Pharmacy</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Park</td>
      <td>Zoo</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Farm</td>
      <td>Falafel Restaurant</td>
      <td>Donut Shop</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>57</th>
      <td>North York</td>
      <td>0.0</td>
      <td>Golf Course</td>
      <td>Storage Facility</td>
      <td>Gas Station</td>
      <td>Park</td>
      <td>Discount Store</td>
      <td>Intersection</td>
      <td>Bakery</td>
      <td>Convenience Store</td>
      <td>Filipino Restaurant</td>
      <td>Fish Market</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Flea Market</td>
      <td>Event Space</td>
      <td>Fireworks Store</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Drugstore</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Scarborough</td>
      <td>0.0</td>
      <td>Park</td>
      <td>Thai Restaurant</td>
      <td>Farm</td>
      <td>Diner</td>
      <td>Skating Rink</td>
      <td>Gym</td>
      <td>Café</td>
      <td>Gym Pool</td>
      <td>Restaurant</td>
      <td>General Entertainment</td>
      <td>Ice Cream Shop</td>
      <td>College Stadium</td>
      <td>Convenience Store</td>
      <td>Farmers Market</td>
      <td>Fireworks Store</td>
      <td>Donut Shop</td>
      <td>Drugstore</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>66</th>
      <td>North York</td>
      <td>0.0</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Gym</td>
      <td>Grocery Store</td>
      <td>French Restaurant</td>
      <td>Gas Station</td>
      <td>Intersection</td>
      <td>Dentist's Office</td>
      <td>Golf Course</td>
      <td>Business Service</td>
      <td>Tennis Court</td>
      <td>Convenience Store</td>
      <td>Playground</td>
      <td>Dog Run</td>
      <td>Dumpling Restaurant</td>
      <td>Electronics Store</td>
      <td>Drugstore</td>
      <td>Eastern European Restaurant</td>
      <td>Zoo</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Etobicoke</td>
      <td>0.0</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Skating Rink</td>
      <td>Italian Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Grocery Store</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Pharmacy</td>
      <td>Fast Food Restaurant</td>
      <td>Pub</td>
      <td>Liquor Store</td>
      <td>American Restaurant</td>
      <td>Pizza Place</td>
      <td>Mexican Restaurant</td>
      <td>Dessert Shop</td>
      <td>Event Space</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Etobicoke</td>
      <td>0.0</td>
      <td>Italian Restaurant</td>
      <td>Park</td>
      <td>Shopping Mall</td>
      <td>Eastern European Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Bus Stop</td>
      <td>Zoo</td>
      <td>Dumpling Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Donut Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Field</td>
      <td>Filipino Restaurant</td>
      <td>Fireworks Store</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
  </tbody>
</table>
</div>



### __*Cluster 2*__


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
      <th>11th Most Common Venue</th>
      <th>12th Most Common Venue</th>
      <th>13th Most Common Venue</th>
      <th>14th Most Common Venue</th>
      <th>15th Most Common Venue</th>
      <th>16th Most Common Venue</th>
      <th>17th Most Common Venue</th>
      <th>18th Most Common Venue</th>
      <th>19th Most Common Venue</th>
      <th>20th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Park</td>
      <td>Pool</td>
      <td>Zoo</td>
      <td>Farm</td>
      <td>Drugstore</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Farmers Market</td>
      <td>Doner Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Field</td>
      <td>Filipino Restaurant</td>
      <td>Fireworks Store</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fish Market</td>
      <td>Flea Market</td>
    </tr>
  </tbody>
</table>
</div>



### _**Cluster 3**_


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
      <th>11th Most Common Venue</th>
      <th>12th Most Common Venue</th>
      <th>13th Most Common Venue</th>
      <th>14th Most Common Venue</th>
      <th>15th Most Common Venue</th>
      <th>16th Most Common Venue</th>
      <th>17th Most Common Venue</th>
      <th>18th Most Common Venue</th>
      <th>19th Most Common Venue</th>
      <th>20th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>North York</td>
      <td>2.0</td>
      <td>Park</td>
      <td>Convenience Store</td>
      <td>Bus Stop</td>
      <td>Pharmacy</td>
      <td>Shopping Mall</td>
      <td>Food &amp; Drink Shop</td>
      <td>Pizza Place</td>
      <td>Café</td>
      <td>Road</td>
      <td>Supermarket</td>
      <td>Shop &amp; Service</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Caribbean Restaurant</td>
      <td>Laundry Service</td>
      <td>Chinese Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Discount Store</td>
      <td>Tennis Court</td>
      <td>Cosmetics Shop</td>
      <td>Train Station</td>
    </tr>
    <tr>
      <th>8</th>
      <td>East York</td>
      <td>2.0</td>
      <td>Brewery</td>
      <td>Pizza Place</td>
      <td>Gym / Fitness Center</td>
      <td>Fast Food Restaurant</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Rock Climbing Spot</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Intersection</td>
      <td>Athletics &amp; Sports</td>
      <td>Gastropub</td>
      <td>Home Service</td>
      <td>Soccer Stadium</td>
      <td>Construction &amp; Landscaping</td>
      <td>Pharmacy</td>
      <td>Event Space</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>10</th>
      <td>North York</td>
      <td>2.0</td>
      <td>Grocery Store</td>
      <td>Fast Food Restaurant</td>
      <td>Pizza Place</td>
      <td>Metro Station</td>
      <td>Furniture / Home Store</td>
      <td>Discount Store</td>
      <td>Gas Station</td>
      <td>Coffee Shop</td>
      <td>Playground</td>
      <td>Restaurant</td>
      <td>Mediterranean Restaurant</td>
      <td>Latin American Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Department Store</td>
      <td>Park</td>
      <td>Bank</td>
      <td>Pub</td>
      <td>Pet Store</td>
      <td>Asian Restaurant</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Etobicoke</td>
      <td>2.0</td>
      <td>Park</td>
      <td>Pizza Place</td>
      <td>Mexican Restaurant</td>
      <td>Gym</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Clothing Store</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Bank</td>
      <td>Grocery Store</td>
      <td>Convenience Store</td>
      <td>Hotel</td>
      <td>Hardware Store</td>
      <td>Electronics Store</td>
      <td>Dive Bar</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Drugstore</td>
      <td>Dumpling Restaurant</td>
    </tr>
    <tr>
      <th>16</th>
      <td>York</td>
      <td>2.0</td>
      <td>Pizza Place</td>
      <td>Coffee Shop</td>
      <td>Convenience Store</td>
      <td>Grocery Store</td>
      <td>Mexican Restaurant</td>
      <td>Trail</td>
      <td>Sushi Restaurant</td>
      <td>Sandwich Place</td>
      <td>Korean Restaurant</td>
      <td>Gastropub</td>
      <td>Middle Eastern Restaurant</td>
      <td>Bank</td>
      <td>Optical Shop</td>
      <td>Bagel Shop</td>
      <td>Gift Shop</td>
      <td>Italian Restaurant</td>
      <td>Frozen Yogurt Shop</td>
      <td>Dance Studio</td>
      <td>Hockey Arena</td>
      <td>Soccer Stadium</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Pizza Place</td>
      <td>Coffee Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Restaurant</td>
      <td>Bank</td>
      <td>Filipino Restaurant</td>
      <td>Pharmacy</td>
      <td>Smoothie Shop</td>
      <td>Food &amp; Drink Shop</td>
      <td>Sports Bar</td>
      <td>Beer Store</td>
      <td>Liquor Store</td>
      <td>Sandwich Place</td>
      <td>Greek Restaurant</td>
      <td>Moving Target</td>
      <td>Chinese Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Discount Store</td>
      <td>Supermarket</td>
      <td>Ethiopian Restaurant</td>
    </tr>
    <tr>
      <th>21</th>
      <td>York</td>
      <td>2.0</td>
      <td>Pharmacy</td>
      <td>Park</td>
      <td>Bus Stop</td>
      <td>Fast Food Restaurant</td>
      <td>Beer Store</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Mexican Restaurant</td>
      <td>Falafel Restaurant</td>
      <td>Discount Store</td>
      <td>Grocery Store</td>
      <td>Japanese Restaurant</td>
      <td>Coffee Shop</td>
      <td>Hostel</td>
      <td>Bus Line</td>
      <td>Pizza Place</td>
      <td>Cosmetics Shop</td>
      <td>Women's Store</td>
      <td>Electronics Store</td>
      <td>Field</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Park</td>
      <td>Coffee Shop</td>
      <td>Chinese Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Pharmacy</td>
      <td>Mobile Phone Shop</td>
      <td>Dance Studio</td>
      <td>Farmers Market</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Creperie</td>
      <td>Drugstore</td>
      <td>Field</td>
      <td>Filipino Restaurant</td>
      <td>Fireworks Store</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Bakery</td>
      <td>Coffee Shop</td>
      <td>Bank</td>
      <td>Indian Restaurant</td>
      <td>Gas Station</td>
      <td>Grocery Store</td>
      <td>Gym / Fitness Center</td>
      <td>Burger Joint</td>
      <td>Fast Food Restaurant</td>
      <td>Pizza Place</td>
      <td>Sporting Goods Shop</td>
      <td>Fried Chicken Joint</td>
      <td>Caribbean Restaurant</td>
      <td>Music Store</td>
      <td>Chinese Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Bus Line</td>
      <td>Wings Joint</td>
      <td>Athletics &amp; Sports</td>
      <td>Pharmacy</td>
    </tr>
    <tr>
      <th>27</th>
      <td>North York</td>
      <td>2.0</td>
      <td>Pharmacy</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Convenience Store</td>
      <td>Grocery Store</td>
      <td>Fast Food Restaurant</td>
      <td>Shopping Mall</td>
      <td>Chinese Restaurant</td>
      <td>Sandwich Place</td>
      <td>Korean Restaurant</td>
      <td>Bank</td>
      <td>Restaurant</td>
      <td>Residential Building (Apartment / Condo)</td>
      <td>Bakery</td>
      <td>Recreation Center</td>
      <td>Pizza Place</td>
      <td>Ice Cream Shop</td>
      <td>Eastern European Restaurant</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>28</th>
      <td>North York</td>
      <td>2.0</td>
      <td>Bank</td>
      <td>Park</td>
      <td>Pizza Place</td>
      <td>Coffee Shop</td>
      <td>Gas Station</td>
      <td>Dog Run</td>
      <td>Sandwich Place</td>
      <td>Fried Chicken Joint</td>
      <td>Sushi Restaurant</td>
      <td>Supermarket</td>
      <td>Restaurant</td>
      <td>Shopping Mall</td>
      <td>Mobile Phone Shop</td>
      <td>Ski Area</td>
      <td>Convenience Store</td>
      <td>Ski Chalet</td>
      <td>Mediterranean Restaurant</td>
      <td>Diner</td>
      <td>Bridal Shop</td>
      <td>Middle Eastern Restaurant</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Ice Cream Shop</td>
      <td>Coffee Shop</td>
      <td>Train Station</td>
      <td>Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Bowling Alley</td>
      <td>Convenience Store</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Women's Store</td>
      <td>Grocery Store</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Drugstore</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Zoo</td>
      <td>Falafel Restaurant</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Chinese Restaurant</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Convenience Store</td>
      <td>Discount Store</td>
      <td>Grocery Store</td>
      <td>Fast Food Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Light Rail Station</td>
      <td>Sandwich Place</td>
      <td>Bank</td>
      <td>Burger Joint</td>
      <td>Auto Garage</td>
      <td>Metro Station</td>
      <td>Rental Car Location</td>
      <td>Department Store</td>
      <td>Pharmacy</td>
      <td>Bus Station</td>
      <td>Field</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>39</th>
      <td>North York</td>
      <td>2.0</td>
      <td>Bank</td>
      <td>Gas Station</td>
      <td>Grocery Store</td>
      <td>Japanese Restaurant</td>
      <td>Café</td>
      <td>Chinese Restaurant</td>
      <td>Shopping Mall</td>
      <td>Park</td>
      <td>Trail</td>
      <td>Restaurant</td>
      <td>Dog Run</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Donut Shop</td>
      <td>Electronics Store</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Field</td>
      <td>Filipino Restaurant</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Intersection</td>
      <td>Bus Line</td>
      <td>Bakery</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Soccer Field</td>
      <td>Mexican Restaurant</td>
      <td>Metro Station</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
      <td>Beer Store</td>
      <td>Bank</td>
      <td>Restaurant</td>
      <td>Park</td>
      <td>Trail</td>
      <td>General Entertainment</td>
      <td>Grocery Store</td>
      <td>Pub</td>
      <td>Convenience Store</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>46</th>
      <td>North York</td>
      <td>2.0</td>
      <td>Park</td>
      <td>Coffee Shop</td>
      <td>Shopping Mall</td>
      <td>Pizza Place</td>
      <td>Vietnamese Restaurant</td>
      <td>Grocery Store</td>
      <td>Bank</td>
      <td>Event Space</td>
      <td>Drugstore</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Zoo</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Field</td>
      <td>Filipino Restaurant</td>
    </tr>
    <tr>
      <th>49</th>
      <td>North York</td>
      <td>2.0</td>
      <td>Coffee Shop</td>
      <td>Dim Sum Restaurant</td>
      <td>Convenience Store</td>
      <td>Chinese Restaurant</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Mediterranean Restaurant</td>
      <td>Gas Station</td>
      <td>Park</td>
      <td>Athletics &amp; Sports</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Zoo</td>
      <td>Dumpling Restaurant</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Field</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Beach</td>
      <td>Pizza Place</td>
      <td>Ice Cream Shop</td>
      <td>Sports Bar</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Hardware Store</td>
      <td>Farm</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Donut Shop</td>
      <td>Field</td>
      <td>Filipino Restaurant</td>
      <td>Fireworks Store</td>
    </tr>
    <tr>
      <th>56</th>
      <td>York</td>
      <td>2.0</td>
      <td>Furniture / Home Store</td>
      <td>Grocery Store</td>
      <td>Dessert Shop</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Discount Store</td>
      <td>Sandwich Place</td>
      <td>Fast Food Restaurant</td>
      <td>Gas Station</td>
      <td>Wine Shop</td>
      <td>Playground</td>
      <td>Bar</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
      <td>Electronics Store</td>
      <td>Zoo</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>60</th>
      <td>North York</td>
      <td>2.0</td>
      <td>Hotel</td>
      <td>Gas Station</td>
      <td>Coffee Shop</td>
      <td>Grocery Store</td>
      <td>Pizza Place</td>
      <td>Pharmacy</td>
      <td>Discount Store</td>
      <td>Fast Food Restaurant</td>
      <td>American Restaurant</td>
      <td>Athletics &amp; Sports</td>
      <td>Road</td>
      <td>Liquor Store</td>
      <td>Fried Chicken Joint</td>
      <td>Kitchen Supply Store</td>
      <td>Vietnamese Restaurant</td>
      <td>Beer Store</td>
      <td>Gym / Fitness Center</td>
      <td>Shopping Mall</td>
      <td>Snack Place</td>
      <td>Falafel Restaurant</td>
    </tr>
    <tr>
      <th>63</th>
      <td>York</td>
      <td>2.0</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Brewery</td>
      <td>Furniture / Home Store</td>
      <td>Pharmacy</td>
      <td>Bus Line</td>
      <td>Convenience Store</td>
      <td>Gas Station</td>
      <td>Park</td>
      <td>Beer Store</td>
      <td>Sandwich Place</td>
      <td>Home Service</td>
      <td>Dive Bar</td>
      <td>Supermarket</td>
      <td>Fried Chicken Joint</td>
      <td>Liquor Store</td>
      <td>Discount Store</td>
      <td>Fast Food Restaurant</td>
      <td>Burger Joint</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Etobicoke</td>
      <td>2.0</td>
      <td>Pizza Place</td>
      <td>Gas Station</td>
      <td>Sandwich Place</td>
      <td>Intersection</td>
      <td>Chinese Restaurant</td>
      <td>Flea Market</td>
      <td>Golf Course</td>
      <td>Golf Driving Range</td>
      <td>Discount Store</td>
      <td>Supermarket</td>
      <td>Coffee Shop</td>
      <td>Ice Cream Shop</td>
      <td>Middle Eastern Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Farmers Market</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Electronics Store</td>
      <td>Fireworks Store</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Grocery Store</td>
      <td>Pizza Place</td>
      <td>Middle Eastern Restaurant</td>
      <td>Intersection</td>
      <td>Burger Joint</td>
      <td>Supermarket</td>
      <td>Furniture / Home Store</td>
      <td>Gas Station</td>
      <td>Korean Restaurant</td>
      <td>Restaurant</td>
      <td>Bakery</td>
      <td>Coffee Shop</td>
      <td>Badminton Court</td>
      <td>Rental Car Location</td>
      <td>Seafood Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Fish Market</td>
      <td>Asian Restaurant</td>
      <td>Indian Restaurant</td>
      <td>African Restaurant</td>
    </tr>
    <tr>
      <th>72</th>
      <td>North York</td>
      <td>2.0</td>
      <td>Pharmacy</td>
      <td>Eastern European Restaurant</td>
      <td>Park</td>
      <td>Convenience Store</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Bus Line</td>
      <td>Butcher</td>
      <td>Discount Store</td>
      <td>Coffee Shop</td>
      <td>Grocery Store</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Zoo</td>
      <td>Farm</td>
      <td>Dumpling Restaurant</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Etobicoke</td>
      <td>2.0</td>
      <td>Pharmacy</td>
      <td>Pizza Place</td>
      <td>Intersection</td>
      <td>Bus Line</td>
      <td>Business Service</td>
      <td>Shopping Mall</td>
      <td>Supermarket</td>
      <td>Supplement Shop</td>
      <td>Beer Store</td>
      <td>Gas Station</td>
      <td>Sandwich Place</td>
      <td>Chinese Restaurant</td>
      <td>Coffee Shop</td>
      <td>Bank</td>
      <td>American Restaurant</td>
      <td>Dessert Shop</td>
      <td>Farm</td>
      <td>Ethiopian Restaurant</td>
      <td>Curling Ice</td>
      <td>Event Space</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Chinese Restaurant</td>
      <td>Shopping Mall</td>
      <td>Sandwich Place</td>
      <td>Coffee Shop</td>
      <td>Caribbean Restaurant</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Print Shop</td>
      <td>Pharmacy</td>
      <td>Sushi Restaurant</td>
      <td>Supermarket</td>
      <td>Latin American Restaurant</td>
      <td>Sri Lankan Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Lounge</td>
      <td>Mediterranean Restaurant</td>
      <td>Seafood Restaurant</td>
      <td>Discount Store</td>
      <td>Cantonese Restaurant</td>
      <td>Bank</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Coffee Shop</td>
      <td>Bank</td>
      <td>Intersection</td>
      <td>Sandwich Place</td>
      <td>Fast Food Restaurant</td>
      <td>Convenience Store</td>
      <td>Gas Station</td>
      <td>Italian Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Shopping Mall</td>
      <td>Market</td>
      <td>Seafood Restaurant</td>
      <td>Cantonese Restaurant</td>
      <td>Caribbean Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Chinese Restaurant</td>
      <td>Taiwanese Restaurant</td>
      <td>Noodle House</td>
      <td>Bakery</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Chinese Restaurant</td>
      <td>Pizza Place</td>
      <td>Korean Restaurant</td>
      <td>Park</td>
      <td>Hobby Shop</td>
      <td>Caribbean Restaurant</td>
      <td>Noodle House</td>
      <td>Bakery</td>
      <td>Shop &amp; Service</td>
      <td>Shopping Mall</td>
      <td>Malay Restaurant</td>
      <td>Dessert Shop</td>
      <td>Coffee Shop</td>
      <td>Grocery Store</td>
      <td>Gym</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Martial Arts Dojo</td>
      <td>Event Space</td>
      <td>Pharmacy</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Etobicoke</td>
      <td>2.0</td>
      <td>Pizza Place</td>
      <td>Grocery Store</td>
      <td>Sandwich Place</td>
      <td>Park</td>
      <td>Caribbean Restaurant</td>
      <td>Bus Line</td>
      <td>Fast Food Restaurant</td>
      <td>Video Store</td>
      <td>Pharmacy</td>
      <td>Hardware Store</td>
      <td>Fried Chicken Joint</td>
      <td>Beer Store</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Farm</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Scarborough</td>
      <td>2.0</td>
      <td>Chinese Restaurant</td>
      <td>Coffee Shop</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Intersection</td>
      <td>Supermarket</td>
      <td>Caribbean Restaurant</td>
      <td>Tennis Court</td>
      <td>Other Great Outdoors</td>
      <td>Electronics Store</td>
      <td>Shopping Mall</td>
      <td>Grocery Store</td>
      <td>Auto Garage</td>
      <td>Sandwich Place</td>
      <td>Breakfast Spot</td>
      <td>Gym Pool</td>
      <td>Indian Restaurant</td>
      <td>Pharmacy</td>
      <td>Hotpot Restaurant</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Etobicoke</td>
      <td>2.0</td>
      <td>Discount Store</td>
      <td>Pizza Place</td>
      <td>Convenience Store</td>
      <td>Pharmacy</td>
      <td>Intersection</td>
      <td>Grocery Store</td>
      <td>Park</td>
      <td>Shopping Mall</td>
      <td>Coffee Shop</td>
      <td>Trail</td>
      <td>Donut Shop</td>
      <td>Sandwich Place</td>
      <td>Garden Center</td>
      <td>Gym</td>
      <td>Gas Station</td>
      <td>Moroccan Restaurant</td>
      <td>Pub</td>
      <td>Print Shop</td>
      <td>Liquor Store</td>
      <td>Dumpling Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



### _**Cluster 4**_


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
      <th>11th Most Common Venue</th>
      <th>12th Most Common Venue</th>
      <th>13th Most Common Venue</th>
      <th>14th Most Common Venue</th>
      <th>15th Most Common Venue</th>
      <th>16th Most Common Venue</th>
      <th>17th Most Common Venue</th>
      <th>18th Most Common Venue</th>
      <th>19th Most Common Venue</th>
      <th>20th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>North York</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Hockey Arena</td>
      <td>Sporting Goods Shop</td>
      <td>Portuguese Restaurant</td>
      <td>Men's Store</td>
      <td>Lounge</td>
      <td>Park</td>
      <td>Golf Course</td>
      <td>Intersection</td>
      <td>Gym / Fitness Center</td>
      <td>Zoo</td>
      <td>Ethiopian Restaurant</td>
      <td>Drugstore</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Doner Restaurant</td>
      <td>Farm</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Pub</td>
      <td>Park</td>
      <td>Diner</td>
      <td>Theater</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Breakfast Spot</td>
      <td>Italian Restaurant</td>
      <td>Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Performing Arts Venue</td>
      <td>Gastropub</td>
      <td>Indian Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Dessert Shop</td>
      <td>Mediterranean Restaurant</td>
      <td>French Restaurant</td>
      <td>Karaoke Bar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>North York</td>
      <td>3.0</td>
      <td>Fast Food Restaurant</td>
      <td>Furniture / Home Store</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Vietnamese Restaurant</td>
      <td>Dessert Shop</td>
      <td>Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Sushi Restaurant</td>
      <td>Café</td>
      <td>Men's Store</td>
      <td>Fireworks Store</td>
      <td>Bank</td>
      <td>Miscellaneous Shop</td>
      <td>Korean Restaurant</td>
      <td>Cheese Shop</td>
      <td>Event Space</td>
      <td>Boutique</td>
      <td>Bowling Alley</td>
      <td>Seafood Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Café</td>
      <td>Ramen Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Burger Joint</td>
      <td>Gay Bar</td>
      <td>Thai Restaurant</td>
      <td>Bookstore</td>
      <td>Ice Cream Shop</td>
      <td>Cosmetics Shop</td>
      <td>Gastropub</td>
      <td>Dance Studio</td>
      <td>Yoga Studio</td>
      <td>Bubble Tea Shop</td>
      <td>Clothing Store</td>
      <td>Pizza Place</td>
      <td>Middle Eastern Restaurant</td>
    </tr>
    <tr>
      <th>7</th>
      <td>North York</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Pizza Place</td>
      <td>Burger Joint</td>
      <td>Bank</td>
      <td>Bar</td>
      <td>Shop &amp; Service</td>
      <td>Supermarket</td>
      <td>Liquor Store</td>
      <td>Thai Restaurant</td>
      <td>Caribbean Restaurant</td>
      <td>Restaurant</td>
      <td>Diner</td>
      <td>Salad Place</td>
      <td>Greek Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Gym</td>
      <td>Breakfast Spot</td>
      <td>Spa</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Gastropub</td>
      <td>Italian Restaurant</td>
      <td>Ramen Restaurant</td>
      <td>Restaurant</td>
      <td>Clothing Store</td>
      <td>Theater</td>
      <td>Hotel</td>
      <td>Poke Place</td>
      <td>Plaza</td>
      <td>Pizza Place</td>
      <td>Diner</td>
      <td>Café</td>
      <td>Cosmetics Shop</td>
      <td>Burrito Place</td>
      <td>Creperie</td>
      <td>Gym</td>
      <td>Middle Eastern Restaurant</td>
      <td>Shopping Mall</td>
    </tr>
    <tr>
      <th>13</th>
      <td>North York</td>
      <td>3.0</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Gym</td>
      <td>Japanese Restaurant</td>
      <td>Supermarket</td>
      <td>Beer Store</td>
      <td>Salon / Barbershop</td>
      <td>Asian Restaurant</td>
      <td>Intersection</td>
      <td>Italian Restaurant</td>
      <td>Bank</td>
      <td>History Museum</td>
      <td>Clothing Store</td>
      <td>New American Restaurant</td>
      <td>Sporting Goods Shop</td>
      <td>Sushi Restaurant</td>
      <td>Bike Shop</td>
      <td>Sandwich Place</td>
      <td>Chinese Restaurant</td>
      <td>Burger Joint</td>
    </tr>
    <tr>
      <th>14</th>
      <td>East York</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Park</td>
      <td>Thai Restaurant</td>
      <td>Athletics &amp; Sports</td>
      <td>Sandwich Place</td>
      <td>Greek Restaurant</td>
      <td>Beer Store</td>
      <td>Liquor Store</td>
      <td>Convenience Store</td>
      <td>Restaurant</td>
      <td>Farmers Market</td>
      <td>Pharmacy</td>
      <td>Skating Rink</td>
      <td>Café</td>
      <td>Bus Stop</td>
      <td>Pastry Shop</td>
      <td>Pub</td>
      <td>Ice Cream Shop</td>
      <td>Video Store</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Gastropub</td>
      <td>Plaza</td>
      <td>Hotel</td>
      <td>Bookstore</td>
      <td>Seafood Restaurant</td>
      <td>Furniture / Home Store</td>
      <td>Bakery</td>
      <td>Tea Room</td>
      <td>Clothing Store</td>
      <td>Theater</td>
      <td>Creperie</td>
      <td>Gym</td>
      <td>Cosmetics Shop</td>
      <td>Poke Place</td>
      <td>Middle Eastern Restaurant</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Etobicoke</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Pharmacy</td>
      <td>Liquor Store</td>
      <td>Beer Store</td>
      <td>Shopping Mall</td>
      <td>Grocery Store</td>
      <td>Café</td>
      <td>College Rec Center</td>
      <td>Pet Store</td>
      <td>Intersection</td>
      <td>Farmers Market</td>
      <td>IT Services</td>
      <td>Pizza Place</td>
      <td>Gas Station</td>
      <td>Farm</td>
      <td>Drugstore</td>
      <td>Fast Food Restaurant</td>
      <td>Falafel Restaurant</td>
      <td>Field</td>
    </tr>
    <tr>
      <th>19</th>
      <td>East Toronto</td>
      <td>3.0</td>
      <td>Pub</td>
      <td>Coffee Shop</td>
      <td>Beach</td>
      <td>Breakfast Spot</td>
      <td>Japanese Restaurant</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Caribbean Restaurant</td>
      <td>Bar</td>
      <td>Indian Restaurant</td>
      <td>Tea Room</td>
      <td>Ice Cream Shop</td>
      <td>Burger Joint</td>
      <td>Park</td>
      <td>Sandwich Place</td>
      <td>Asian Restaurant</td>
      <td>Chocolate Shop</td>
      <td>Frozen Yogurt Shop</td>
      <td>Beer Store</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Japanese Restaurant</td>
      <td>Hotel</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Bakery</td>
      <td>Creperie</td>
      <td>Grocery Store</td>
      <td>Beer Bar</td>
      <td>Breakfast Spot</td>
      <td>Gym</td>
      <td>Gastropub</td>
      <td>Liquor Store</td>
      <td>Sandwich Place</td>
      <td>Farmers Market</td>
      <td>Seafood Restaurant</td>
      <td>Plaza</td>
      <td>Lounge</td>
      <td>Fish Market</td>
    </tr>
    <tr>
      <th>23</th>
      <td>East York</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Sporting Goods Shop</td>
      <td>Furniture / Home Store</td>
      <td>Grocery Store</td>
      <td>Electronics Store</td>
      <td>Pet Store</td>
      <td>Brewery</td>
      <td>Department Store</td>
      <td>Sports Bar</td>
      <td>Burger Joint</td>
      <td>Restaurant</td>
      <td>Bank</td>
      <td>Sandwich Place</td>
      <td>Breakfast Spot</td>
      <td>Pool</td>
      <td>Bike Shop</td>
      <td>Supermarket</td>
      <td>Performing Arts Venue</td>
      <td>Shopping Mall</td>
      <td>Smoothie Shop</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Café</td>
      <td>Ramen Restaurant</td>
      <td>Park</td>
      <td>Art Gallery</td>
      <td>Japanese Restaurant</td>
      <td>Pizza Place</td>
      <td>Mexican Restaurant</td>
      <td>Burrito Place</td>
      <td>Clothing Store</td>
      <td>Plaza</td>
      <td>Cosmetics Shop</td>
      <td>Theater</td>
      <td>Diner</td>
      <td>Furniture / Home Store</td>
      <td>Juice Bar</td>
      <td>Gastropub</td>
      <td>Yoga Studio</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Korean Restaurant</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Grocery Store</td>
      <td>Cocktail Bar</td>
      <td>Mexican Restaurant</td>
      <td>Ethiopian Restaurant</td>
      <td>Comedy Club</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Park</td>
      <td>Indian Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Bar</td>
      <td>Pizza Place</td>
      <td>Diner</td>
      <td>Pub</td>
      <td>Ramen Restaurant</td>
      <td>Record Shop</td>
      <td>Playground</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>29</th>
      <td>East York</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Grocery Store</td>
      <td>Indian Restaurant</td>
      <td>Gym</td>
      <td>Supermarket</td>
      <td>Brewery</td>
      <td>Burger Joint</td>
      <td>Middle Eastern Restaurant</td>
      <td>Turkish Restaurant</td>
      <td>Bank</td>
      <td>Afghan Restaurant</td>
      <td>Restaurant</td>
      <td>Discount Store</td>
      <td>Furniture / Home Store</td>
      <td>Performing Arts Venue</td>
      <td>Beer Store</td>
      <td>Fried Chicken Joint</td>
      <td>Park</td>
      <td>Smoothie Shop</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Theater</td>
      <td>Gym</td>
      <td>Japanese Restaurant</td>
      <td>Art Gallery</td>
      <td>Restaurant</td>
      <td>Tea Room</td>
      <td>Furniture / Home Store</td>
      <td>Italian Restaurant</td>
      <td>Movie Theater</td>
      <td>Concert Hall</td>
      <td>Plaza</td>
      <td>Sushi Restaurant</td>
      <td>Clothing Store</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Comedy Club</td>
      <td>Fast Food Restaurant</td>
      <td>Brazilian Restaurant</td>
    </tr>
    <tr>
      <th>31</th>
      <td>West Toronto</td>
      <td>3.0</td>
      <td>Café</td>
      <td>Park</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Sushi Restaurant</td>
      <td>Bar</td>
      <td>Pharmacy</td>
      <td>Italian Restaurant</td>
      <td>Convenience Store</td>
      <td>Brewery</td>
      <td>Gourmet Shop</td>
      <td>Portuguese Restaurant</td>
      <td>Beer Store</td>
      <td>Music Venue</td>
      <td>Restaurant</td>
      <td>Supermarket</td>
      <td>Dog Run</td>
      <td>Discount Store</td>
      <td>Brazilian Restaurant</td>
      <td>Breakfast Spot</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North York</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Clothing Store</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Sandwich Place</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Theater</td>
      <td>Tea Room</td>
      <td>Fast Food Restaurant</td>
      <td>Caribbean Restaurant</td>
      <td>Liquor Store</td>
      <td>Juice Bar</td>
      <td>Salon / Barbershop</td>
      <td>Fried Chicken Joint</td>
      <td>Chocolate Shop</td>
      <td>Electronics Store</td>
      <td>Toy / Game Store</td>
      <td>Beer Store</td>
      <td>Movie Theater</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North York</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Furniture / Home Store</td>
      <td>Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Bar</td>
      <td>Bank</td>
      <td>Middle Eastern Restaurant</td>
      <td>Sandwich Place</td>
      <td>Caribbean Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Miscellaneous Shop</td>
      <td>Massage Studio</td>
      <td>Sports Bar</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Dumpling Restaurant</td>
      <td>Falafel Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Event Space</td>
    </tr>
    <tr>
      <th>35</th>
      <td>East York</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Greek Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Beer Bar</td>
      <td>Ethiopian Restaurant</td>
      <td>Pizza Place</td>
      <td>Sandwich Place</td>
      <td>Breakfast Spot</td>
      <td>American Restaurant</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Pharmacy</td>
      <td>Gastropub</td>
      <td>Convenience Store</td>
      <td>Park</td>
      <td>Karaoke Bar</td>
      <td>Frame Store</td>
      <td>Bookstore</td>
      <td>Ramen Restaurant</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Park</td>
      <td>Gym</td>
      <td>Japanese Restaurant</td>
      <td>Theater</td>
      <td>Brewery</td>
      <td>Scenic Lookout</td>
      <td>Deli / Bodega</td>
      <td>Italian Restaurant</td>
      <td>Plaza</td>
      <td>Pizza Place</td>
      <td>Concert Hall</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Baseball Stadium</td>
      <td>Restaurant</td>
      <td>Mediterranean Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>37</th>
      <td>West Toronto</td>
      <td>3.0</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Restaurant</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Bakery</td>
      <td>Italian Restaurant</td>
      <td>Cocktail Bar</td>
      <td>Asian Restaurant</td>
      <td>Pizza Place</td>
      <td>Theater</td>
      <td>Coffee Shop</td>
      <td>Gift Shop</td>
      <td>Seafood Restaurant</td>
      <td>Park</td>
      <td>Vietnamese Restaurant</td>
      <td>Bookstore</td>
      <td>Wine Bar</td>
      <td>Japanese Restaurant</td>
      <td>Men's Store</td>
      <td>Brewery</td>
    </tr>
    <tr>
      <th>40</th>
      <td>North York</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Turkish Restaurant</td>
      <td>Middle Eastern Restaurant</td>
      <td>Café</td>
      <td>Gas Station</td>
      <td>Sandwich Place</td>
      <td>Electronics Store</td>
      <td>Liquor Store</td>
      <td>Other Repair Shop</td>
      <td>Italian Restaurant</td>
      <td>Athletics &amp; Sports</td>
      <td>Park</td>
      <td>Gym</td>
      <td>Latin American Restaurant</td>
      <td>Soccer Field</td>
      <td>Food Court</td>
      <td>Vietnamese Restaurant</td>
      <td>Business Service</td>
      <td>Airport</td>
      <td>Filipino Restaurant</td>
    </tr>
    <tr>
      <th>41</th>
      <td>East Toronto</td>
      <td>3.0</td>
      <td>Greek Restaurant</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Pub</td>
      <td>Fast Food Restaurant</td>
      <td>Bank</td>
      <td>Italian Restaurant</td>
      <td>Ramen Restaurant</td>
      <td>Bakery</td>
      <td>Spa</td>
      <td>Ice Cream Shop</td>
      <td>Discount Store</td>
      <td>Yoga Studio</td>
      <td>Pizza Place</td>
      <td>Furniture / Home Store</td>
      <td>Grocery Store</td>
      <td>Bookstore</td>
      <td>Restaurant</td>
      <td>Falafel Restaurant</td>
      <td>Caribbean Restaurant</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Concert Hall</td>
      <td>Italian Restaurant</td>
      <td>Theater</td>
      <td>Plaza</td>
      <td>Seafood Restaurant</td>
      <td>Gym</td>
      <td>Thai Restaurant</td>
      <td>Park</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Monument / Landmark</td>
      <td>Sandwich Place</td>
      <td>Speakeasy</td>
      <td>Spa</td>
      <td>Bistro</td>
      <td>Deli / Bodega</td>
    </tr>
    <tr>
      <th>43</th>
      <td>West Toronto</td>
      <td>3.0</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Bar</td>
      <td>Furniture / Home Store</td>
      <td>Bakery</td>
      <td>Tibetan Restaurant</td>
      <td>Gift Shop</td>
      <td>Supermarket</td>
      <td>Indian Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Thrift / Vintage Store</td>
      <td>Sandwich Place</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Park</td>
      <td>Performing Arts Venue</td>
      <td>Soccer Stadium</td>
      <td>New American Restaurant</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>47</th>
      <td>East Toronto</td>
      <td>3.0</td>
      <td>Indian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Grocery Store</td>
      <td>Beach</td>
      <td>Café</td>
      <td>Fast Food Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Restaurant</td>
      <td>Spa</td>
      <td>Brewery</td>
      <td>Light Rail Station</td>
      <td>Burrito Place</td>
      <td>Bakery</td>
      <td>Sandwich Place</td>
      <td>Park</td>
      <td>Gym</td>
      <td>Pub</td>
      <td>Middle Eastern Restaurant</td>
      <td>Pakistani Restaurant</td>
      <td>Food</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Hotel</td>
      <td>Japanese Restaurant</td>
      <td>Concert Hall</td>
      <td>Gym</td>
      <td>Seafood Restaurant</td>
      <td>Plaza</td>
      <td>Cosmetics Shop</td>
      <td>Bakery</td>
      <td>Theater</td>
      <td>Thai Restaurant</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Art Gallery</td>
      <td>Gastropub</td>
      <td>Church</td>
      <td>Park</td>
      <td>IT Services</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>52</th>
      <td>North York</td>
      <td>3.0</td>
      <td>Korean Restaurant</td>
      <td>Café</td>
      <td>Park</td>
      <td>Middle Eastern Restaurant</td>
      <td>Pizza Place</td>
      <td>Coffee Shop</td>
      <td>Bus Line</td>
      <td>Supermarket</td>
      <td>Fried Chicken Joint</td>
      <td>Bank</td>
      <td>Diner</td>
      <td>Dessert Shop</td>
      <td>Japanese Restaurant</td>
      <td>Shopping Mall</td>
      <td>Trail</td>
      <td>Grocery Store</td>
      <td>Ski Chalet</td>
      <td>Hookah Bar</td>
      <td>Hot Dog Joint</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>54</th>
      <td>East Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Bar</td>
      <td>Café</td>
      <td>Vietnamese Restaurant</td>
      <td>Diner</td>
      <td>Brewery</td>
      <td>Bakery</td>
      <td>American Restaurant</td>
      <td>French Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Pet Store</td>
      <td>Sandwich Place</td>
      <td>Thai Restaurant</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Park</td>
      <td>Gastropub</td>
      <td>Sushi Restaurant</td>
      <td>Latin American Restaurant</td>
      <td>Seafood Restaurant</td>
    </tr>
    <tr>
      <th>55</th>
      <td>North York</td>
      <td>3.0</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Sandwich Place</td>
      <td>Bank</td>
      <td>Juice Bar</td>
      <td>Thai Restaurant</td>
      <td>Bagel Shop</td>
      <td>Bakery</td>
      <td>Butcher</td>
      <td>Sushi Restaurant</td>
      <td>Pet Store</td>
      <td>Intersection</td>
      <td>Sports Club</td>
      <td>Liquor Store</td>
      <td>Breakfast Spot</td>
      <td>Bridal Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Café</td>
      <td>Skating Rink</td>
    </tr>
    <tr>
      <th>59</th>
      <td>North York</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Bubble Tea Shop</td>
      <td>Ramen Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Korean Restaurant</td>
      <td>Pizza Place</td>
      <td>Sandwich Place</td>
      <td>Fast Food Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Bank</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Dessert Shop</td>
      <td>Fried Chicken Joint</td>
      <td>Discount Store</td>
      <td>Gym</td>
      <td>Grocery Store</td>
      <td>Pharmacy</td>
      <td>Middle Eastern Restaurant</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Central Toronto</td>
      <td>3.0</td>
      <td>College Gym</td>
      <td>Pharmacy</td>
      <td>Coffee Shop</td>
      <td>Gym / Fitness Center</td>
      <td>Café</td>
      <td>College Quad</td>
      <td>Park</td>
      <td>Bookstore</td>
      <td>Bus Line</td>
      <td>Pool</td>
      <td>Trail</td>
      <td>Falafel Restaurant</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
      <td>Zoo</td>
      <td>Donut Shop</td>
      <td>Farm</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Central Toronto</td>
      <td>3.0</td>
      <td>Italian Restaurant</td>
      <td>Bank</td>
      <td>Asian Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Pharmacy</td>
      <td>Gym</td>
      <td>Lingerie Store</td>
      <td>Bakery</td>
      <td>Bagel Shop</td>
      <td>Japanese Restaurant</td>
      <td>Gastropub</td>
      <td>Gym Pool</td>
      <td>Skating Rink</td>
      <td>Deli / Bodega</td>
      <td>Dumpling Restaurant</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Falafel Restaurant</td>
    </tr>
    <tr>
      <th>64</th>
      <td>York</td>
      <td>3.0</td>
      <td>Train Station</td>
      <td>Coffee Shop</td>
      <td>Sandwich Place</td>
      <td>Café</td>
      <td>Fried Chicken Joint</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Gift Shop</td>
      <td>Pharmacy</td>
      <td>Soccer Field</td>
      <td>Middle Eastern Restaurant</td>
      <td>Pizza Place</td>
      <td>Bank</td>
      <td>Skating Rink</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Event Space</td>
      <td>Drugstore</td>
      <td>Falafel Restaurant</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Scarborough</td>
      <td>3.0</td>
      <td>Furniture / Home Store</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Restaurant</td>
      <td>Electronics Store</td>
      <td>Fast Food Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Pharmacy</td>
      <td>American Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Pizza Place</td>
      <td>Beer Store</td>
      <td>Gas Station</td>
      <td>Sandwich Place</td>
      <td>Bowling Alley</td>
      <td>Fried Chicken Joint</td>
      <td>Bank</td>
      <td>Pet Store</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Central Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Gym</td>
      <td>Dessert Shop</td>
      <td>Pizza Place</td>
      <td>Sushi Restaurant</td>
      <td>Movie Theater</td>
      <td>Food &amp; Drink Shop</td>
      <td>Supermarket</td>
      <td>Bookstore</td>
      <td>Sandwich Place</td>
      <td>Salad Place</td>
      <td>Pharmacy</td>
      <td>Yoga Studio</td>
      <td>Pub</td>
      <td>Diner</td>
      <td>Mexican Restaurant</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Central Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Gym / Fitness Center</td>
      <td>Park</td>
      <td>Bank</td>
      <td>Italian Restaurant</td>
      <td>Trail</td>
      <td>Skating Rink</td>
      <td>Pharmacy</td>
      <td>Burger Joint</td>
      <td>Liquor Store</td>
      <td>Japanese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Bagel Shop</td>
      <td>Bakery</td>
      <td>Gastropub</td>
      <td>Gas Station</td>
      <td>Garden</td>
      <td>Salon / Barbershop</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>69</th>
      <td>West Toronto</td>
      <td>3.0</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Bar</td>
      <td>Convenience Store</td>
      <td>Italian Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Park</td>
      <td>Sushi Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Indian Restaurant</td>
      <td>Sandwich Place</td>
      <td>Mexican Restaurant</td>
      <td>Cajun / Creole Restaurant</td>
      <td>Gastropub</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Flea Market</td>
      <td>Nail Salon</td>
      <td>Antique Shop</td>
      <td>Breakfast Spot</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Central Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Park</td>
      <td>Skating Rink</td>
      <td>Diner</td>
      <td>Restaurant</td>
      <td>Sporting Goods Shop</td>
      <td>Mexican Restaurant</td>
      <td>Café</td>
      <td>Spa</td>
      <td>Food &amp; Drink Shop</td>
      <td>Clothing Store</td>
      <td>Furniture / Home Store</td>
      <td>Chinese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Garden</td>
      <td>Flower Shop</td>
      <td>Electronics Store</td>
      <td>Tea Room</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Central Toronto</td>
      <td>3.0</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Museum</td>
      <td>Bakery</td>
      <td>Restaurant</td>
      <td>Grocery Store</td>
      <td>Gym</td>
      <td>Pub</td>
      <td>History Museum</td>
      <td>Tea Room</td>
      <td>Sandwich Place</td>
      <td>Thai Restaurant</td>
      <td>Bookstore</td>
      <td>Mexican Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Pizza Place</td>
      <td>Burger Joint</td>
      <td>Deli / Bodega</td>
    </tr>
    <tr>
      <th>75</th>
      <td>West Toronto</td>
      <td>3.0</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Bar</td>
      <td>Pizza Place</td>
      <td>Breakfast Spot</td>
      <td>Pub</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Thai Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Grocery Store</td>
      <td>Gift Shop</td>
      <td>Gas Station</td>
      <td>Ice Cream Shop</td>
      <td>Restaurant</td>
      <td>Sandwich Place</td>
      <td>Dog Run</td>
      <td>Gourmet Shop</td>
      <td>Bookstore</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Mississauga</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Chinese Restaurant</td>
      <td>Middle Eastern Restaurant</td>
      <td>Hotel</td>
      <td>Bakery</td>
      <td>Mexican Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Gym</td>
      <td>Indian Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Nightclub</td>
      <td>Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Steakhouse</td>
      <td>Juice Bar</td>
      <td>Sporting Goods Shop</td>
      <td>Japanese Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Sandwich Place</td>
      <td>Caribbean Restaurant</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Central Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Café</td>
      <td>Indian Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Dessert Shop</td>
      <td>Pizza Place</td>
      <td>Restaurant</td>
      <td>Gym</td>
      <td>Movie Theater</td>
      <td>Pub</td>
      <td>Pharmacy</td>
      <td>Bookstore</td>
      <td>Middle Eastern Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Bar</td>
      <td>Sandwich Place</td>
      <td>Ramen Restaurant</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Coffee Shop</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Bar</td>
      <td>Beer Bar</td>
      <td>Bookstore</td>
      <td>Mexican Restaurant</td>
      <td>Restaurant</td>
      <td>Comfort Food Restaurant</td>
      <td>Grocery Store</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Dessert Shop</td>
      <td>Park</td>
      <td>Doner Restaurant</td>
      <td>Belgian Restaurant</td>
      <td>Jazz Club</td>
      <td>Cheese Shop</td>
    </tr>
    <tr>
      <th>81</th>
      <td>West Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
      <td>Pub</td>
      <td>Italian Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Bank</td>
      <td>Falafel Restaurant</td>
      <td>Latin American Restaurant</td>
      <td>Frozen Yogurt Shop</td>
      <td>Sandwich Place</td>
      <td>French Restaurant</td>
      <td>Bookstore</td>
      <td>Diner</td>
      <td>River</td>
      <td>Spa</td>
      <td>Flower Shop</td>
    </tr>
    <tr>
      <th>83</th>
      <td>Central Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Grocery Store</td>
      <td>Park</td>
      <td>Thai Restaurant</td>
      <td>Gym</td>
      <td>Restaurant</td>
      <td>Bagel Shop</td>
      <td>Sandwich Place</td>
      <td>Bank</td>
      <td>Café</td>
      <td>Sushi Restaurant</td>
      <td>Pizza Place</td>
      <td>Pharmacy</td>
      <td>Pub</td>
      <td>Tennis Court</td>
      <td>Neighborhood</td>
      <td>Chiropractor</td>
      <td>Japanese Restaurant</td>
      <td>Fried Chicken Joint</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Coffee Shop</td>
      <td>Art Gallery</td>
      <td>Park</td>
      <td>Bakery</td>
      <td>Yoga Studio</td>
      <td>Mexican Restaurant</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Dessert Shop</td>
      <td>Taco Place</td>
      <td>Pizza Place</td>
      <td>Tea Room</td>
      <td>Caribbean Restaurant</td>
      <td>French Restaurant</td>
      <td>Gaming Cafe</td>
      <td>Record Shop</td>
      <td>Vietnamese Restaurant</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Central Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Sushi Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Grocery Store</td>
      <td>Café</td>
      <td>Liquor Store</td>
      <td>Bagel Shop</td>
      <td>Sandwich Place</td>
      <td>Gym / Fitness Center</td>
      <td>Pizza Place</td>
      <td>Spa</td>
      <td>Pub</td>
      <td>Restaurant</td>
      <td>Gym</td>
      <td>Bank</td>
      <td>Chiropractor</td>
      <td>Breakfast Spot</td>
      <td>Neighborhood</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Harbor / Marina</td>
      <td>Sculpture Garden</td>
      <td>Dog Run</td>
      <td>Airport</td>
      <td>Scenic Lookout</td>
      <td>Dance Studio</td>
      <td>Garden</td>
      <td>Park</td>
      <td>Track</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Zoo</td>
      <td>Event Space</td>
      <td>Donut Shop</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Grocery Store</td>
      <td>Convenience Store</td>
      <td>Japanese Restaurant</td>
      <td>Pie Shop</td>
      <td>Office</td>
      <td>Sandwich Place</td>
      <td>Candy Store</td>
      <td>Filipino Restaurant</td>
      <td>Metro Station</td>
      <td>Breakfast Spot</td>
      <td>Bistro</td>
      <td>Korean Restaurant</td>
      <td>Bank</td>
      <td>BBQ Joint</td>
      <td>Mattress Store</td>
      <td>Playground</td>
      <td>Athletics &amp; Sports</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Hotel</td>
      <td>Gastropub</td>
      <td>Seafood Restaurant</td>
      <td>Bakery</td>
      <td>Breakfast Spot</td>
      <td>Park</td>
      <td>Beer Bar</td>
      <td>Gym</td>
      <td>Creperie</td>
      <td>American Restaurant</td>
      <td>Art Gallery</td>
      <td>Church</td>
      <td>Food Truck</td>
      <td>French Restaurant</td>
      <td>Irish Pub</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Etobicoke</td>
      <td>3.0</td>
      <td>Hotel</td>
      <td>Coffee Shop</td>
      <td>Drugstore</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Doner Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Field</td>
      <td>Filipino Restaurant</td>
      <td>Fireworks Store</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fish Market</td>
      <td>Flea Market</td>
      <td>Flower Shop</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Café</td>
      <td>Diner</td>
      <td>Japanese Restaurant</td>
      <td>Gastropub</td>
      <td>Park</td>
      <td>Jewelry Store</td>
      <td>Restaurant</td>
      <td>Theater</td>
      <td>Thai Restaurant</td>
      <td>Bakery</td>
      <td>Taiwanese Restaurant</td>
      <td>Gift Shop</td>
      <td>Caribbean Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Garden</td>
      <td>Botanical Garden</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Steakhouse</td>
      <td>American Restaurant</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>Theater</td>
      <td>Concert Hall</td>
      <td>Park</td>
      <td>Monument / Landmark</td>
      <td>Japanese Restaurant</td>
      <td>Plaza</td>
      <td>Italian Restaurant</td>
      <td>Seafood Restaurant</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Gym</td>
      <td>Brazilian Restaurant</td>
      <td>Cocktail Bar</td>
      <td>Bookstore</td>
      <td>Bistro</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Etobicoke</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Burger Joint</td>
      <td>French Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Sushi Restaurant</td>
      <td>Bank</td>
      <td>Italian Restaurant</td>
      <td>Pub</td>
      <td>Dessert Shop</td>
      <td>Greek Restaurant</td>
      <td>Gastropub</td>
      <td>Lighting Store</td>
      <td>Business Service</td>
      <td>River</td>
      <td>Cupcake Shop</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Gas Station</td>
      <td>Tapas Restaurant</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Downtown Toronto</td>
      <td>3.0</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Diner</td>
      <td>Sushi Restaurant</td>
      <td>Men's Store</td>
      <td>Japanese Restaurant</td>
      <td>Caribbean Restaurant</td>
      <td>Dance Studio</td>
      <td>Bubble Tea Shop</td>
      <td>Thai Restaurant</td>
      <td>Clothing Store</td>
      <td>Ramen Restaurant</td>
      <td>Restaurant</td>
      <td>Pizza Place</td>
      <td>Ice Cream Shop</td>
      <td>Bookstore</td>
      <td>Burger Joint</td>
      <td>Café</td>
      <td>Gay Bar</td>
      <td>Yoga Studio</td>
    </tr>
    <tr>
      <th>100</th>
      <td>East Toronto</td>
      <td>3.0</td>
      <td>Park</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Brewery</td>
      <td>Italian Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Bakery</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>French Restaurant</td>
      <td>Burrito Place</td>
      <td>Snack Place</td>
      <td>Thai Restaurant</td>
      <td>Liquor Store</td>
      <td>Electronics Store</td>
      <td>Steakhouse</td>
      <td>Bistro</td>
      <td>Beach</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Etobicoke</td>
      <td>3.0</td>
      <td>Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Burrito Place</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Convenience Store</td>
      <td>Burger Joint</td>
      <td>Grocery Store</td>
      <td>Yoga Studio</td>
      <td>Italian Restaurant</td>
      <td>Movie Theater</td>
      <td>Social Club</td>
      <td>Supplement Shop</td>
      <td>Liquor Store</td>
      <td>Cheese Shop</td>
      <td>Middle Eastern Restaurant</td>
      <td>Miscellaneous Shop</td>
      <td>Flower Shop</td>
    </tr>
  </tbody>
</table>
</div>



### _**Cluster 5**_


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
      <th>11th Most Common Venue</th>
      <th>12th Most Common Venue</th>
      <th>13th Most Common Venue</th>
      <th>14th Most Common Venue</th>
      <th>15th Most Common Venue</th>
      <th>16th Most Common Venue</th>
      <th>17th Most Common Venue</th>
      <th>18th Most Common Venue</th>
      <th>19th Most Common Venue</th>
      <th>20th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53</th>
      <td>North York</td>
      <td>4.0</td>
      <td>Vietnamese Restaurant</td>
      <td>Food Truck</td>
      <td>Baseball Field</td>
      <td>Farmers Market</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Fast Food Restaurant</td>
      <td>Donut Shop</td>
      <td>Field</td>
      <td>Filipino Restaurant</td>
      <td>Fireworks Store</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fish Market</td>
      <td>Flea Market</td>
      <td>Flower Shop</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
