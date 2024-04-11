#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
import pandas as pd
import matplotlib.pyplot as plt


# The next paragraph of code is the code to request the data from the API.

# In[26]:


url = "https://api.ned.nl/v1/utilizations"

headers = {
 'X-AUTH-TOKEN': '048cbb84c5f228247869ce29efa0e784580e61ee69117e164de1af273943f84f', # personal key
 'accept': 'text/csv'}
params = {'activity': 1, 'point': 8, 'type': 2, 'granularity': 5, 'granularitytimezone': 1,
          'classification': 2, 'activity': 1,
          'validfrom[strictly_before]': '2019-04-01',
          'validfrom[after]': '2018-08-26'}
response = requests.get(url, headers=headers, params=params, allow_redirects=False)

import pandas as pd
from io import StringIO

# Assuming response_text contains the CSV data as a string
response_text = response.text

# Read the CSV data into a DataFrame
df= pd.read_csv(StringIO(response_text))

# Assuming df is your DataFrame
#df.to_csv('18.csv', index=False)

# Display the top 10 rows of the DataFrame
print(df)


# I had to request the data manually for 90 times, as the API would not provide me with more than 144 rows (hourly data for 5 days) at once. Therefore, I changed the date manually and saved every 5 days to a csv file. Later I comebine all the csv files in one data frame. I tried a loop, but then the API would not read the request correctly. As this process is very time consuming, I only take data for one year and a few months to check if the model works.
# 
# For the API to work, you need an account at https://ned.nl and need to be logged in to request data. For the request you also need a personal key, which you have to include in the code.

# In[27]:


# Initialize an empty list to store DataFrames
dfs = []

# Loop through each CSV file
for i in range(1, 19):
    # Read the CSV file into a DataFrame
    df_power = pd.read_csv(f"{i}.csv")
    
    # Append the DataFrame to the list
    dfs.append(df_power)

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Write the combined DataFrame to a new CSV file
combined_df.to_csv('solarpower_summer_2018.csv', index=False)


# In[28]:


combined_df


# # Cleaning dataframe
# 
# In this section the difference between the volume and capacity is visualized and all the useless columns are dropped. 

# In[29]:


df_hourly = pd.read_csv('solarpower_summer_2018.csv')

df_hourly = df_hourly.drop(columns=['id', 'point', 'type', 'granularity','granularitytimezone',
                                        'activity', 'classification', 'emission', 'emissionfactor',
                                        'validfrom', 'lastupdate', 'percentage'])
df_hourly.rename(columns={'capacity': 'solar_power_kW', 'volume': 'volume_kWh', 'validto': 'Date'}, inplace=True)

# Assuming df_hourly is your DataFrame
df_hourly['Date'] = pd.to_datetime(df_hourly['Date'])
df_hourly['Date'] = df_hourly['Date'].dt.strftime('%d-%m-%Y %H:%M')

df_hourly = df_hourly.set_index('Date', drop = True)
#df_hourly = df_hourly.iloc[ : 12937,:]

difference = df_hourly['solar_power_kW']-df_hourly['volume_kWh']
df_hourly['difference']= difference


# In[30]:


df_hourly


# In[31]:


# volume is the capacity times the hour, which is 1. Therefore, the same values and can be dropped.
df_power = df_hourly.drop(columns=['volume_kWh', 'difference'])
df_power


# # Solar power distribution
# 
# In this assignment I will do a forecast on the solar power production for the Dutch island Texel. This island is part of the province Noord-Holland in the Netherlands. The API I used to gather data for the solar powerproduction is the data for the complete provence. To have a more accurate forecast. The total amount of solar power in the provence is multiplied by the percentage of the contribution of the island Texel. The registered solar power in 2017 on Texel was 4391 kW.
# ![image-2.png](attachment:image-2.png)(https://texel.incijfers.nl/dashboard/hernieuwbare-energie). 
# 
# The total registered solar power of the province Noord-Holland in 2017 was 312986 kW. Therefore, the contribution of Texel to the province was 4391/312986.
# For 2018 this was 500042 kW in Noord Holland and for Texel it was 6161 kW.
# ![image-3.png](attachment:image-3.png) https://energietransitie-nh.incijfers.nl/dashboard/energietransitie/zon

# In[11]:


contribution = 4391/312986
df_power_Texel = df_power * contribution
df_power_Texel


# In[32]:


contribution_2018 = 6161 / 500042
df_power_Texel_2018 = df_power * contribution_2018
df_power_Texel_2018


# In[12]:


df_power_Texel.to_csv('df_power_hourly.csv', index=True)


# In[35]:


df_power_Texel_2018.to_csv('df_power_hourly_summer_2018.csv', index=True)


# In[33]:


# Assuming df is your DataFrame with the index representing the date
# and 'column_name' is the name of the column you want to plot
plt.plot(df_hourly.index, df_power['solar_power_kW'])
plt.xlabel('Date')
plt.ylabel('Column Value')
plt.show()


# In[ ]:




