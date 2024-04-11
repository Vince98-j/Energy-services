#!/usr/bin/env python
# coding: utf-8

# # Running an API
# The API from open-meteo.com is used to import hourly weather data from the Dutch island Texel.

# In[2]:


import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
import matplotlib.pyplot as plt


# In[3]:


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 53.0833,
	"longitude": 4.8333,
	"start_date": "2017-01-01",
	"end_date": "2019-04-01",
	"hourly": ["temperature_2m", "precipitation", "rain", "snowfall", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance", "is_day", "sunshine_duration"],
	"daily": "precipitation_sum"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
hourly_rain = hourly.Variables(2).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
hourly_cloud_cover_low = hourly.Variables(5).ValuesAsNumpy()
hourly_cloud_cover_mid = hourly.Variables(6).ValuesAsNumpy()
hourly_cloud_cover_high = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()
hourly_shortwave_radiation = hourly.Variables(9).ValuesAsNumpy()
hourly_diffuse_radiation = hourly.Variables(10).ValuesAsNumpy()
hourly_direct_normal_irradiance = hourly.Variables(11).ValuesAsNumpy()
hourly_is_day = hourly.Variables(12).ValuesAsNumpy()
hourly_sunshine_duration = hourly.Variables(13).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
hourly_data["diffuse_radiation"] = hourly_diffuse_radiation
hourly_data["direct_normal_irradiance"] = hourly_direct_normal_irradiance
hourly_data["is_day"] = hourly_is_day
hourly_data["sunshine_duration"] = hourly_sunshine_duration

hourly_dataframe = pd.DataFrame(data = hourly_data)
print(hourly_dataframe)

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_precipitation_sum = daily.Variables(0).ValuesAsNumpy()

daily_data = {"date": pd.date_range(
	start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
	end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = daily.Interval()),
	inclusive = "left"
)}
daily_data["precipitation_sum"] = daily_precipitation_sum

daily_dataframe = pd.DataFrame(data = daily_data)
#print(daily_dataframe)


# In[4]:


hourly_dataframe.to_csv('raw_weather.csv', index=False)


# # Importing the csv file and do some checking

# In[5]:


df_weather = pd.read_csv('raw_weather.csv')


# In[6]:


df_weather['date'] = pd.to_datetime(df_weather['date'])
df_weather['date'] = df_weather['date'].dt.strftime('%d-%m-%Y %H:%M')
df_weather = df_weather.set_index('date', drop = True)

df_weather.rename(columns={'temperature_2m' : 'Temp [C]', 'precipitation':'precipitation [mm]','rain': 'rain [mm]', 'snowfall': 'snowfall [cm]', 'wind_speed_10m' : 'wind [km/h]', 'shortwave_radiation' : 'GHI [W/m²]', 'direct_normal_irradiance' : 'DNI [W/m²]', 'diffuse_radiation' : 'DR [W/m²]', 'cloud_cover': 'cloud [%]', 'cloud_cover_low' : 'cloud_l [%]', 'cloud_cover_mid':'cloud_m [%]', 'cloud_cover_high':'cloud_h [%]'}, inplace=True)

df_weather_2018_2019=df_weather

#Drop all rows after 31-03-2018 23:00 for the training data
df_weather_2018_2019=df_weather.iloc[10920 : ,:]
df_weather = df_weather.iloc[ : 10920,:]


# In[7]:


df_weather_2018_2019.to_csv('df_weather_2018_2019.csv', index=True)


# In[8]:


df_weather_2018_2019


# # Checking if there is data in every column

# In[9]:


plt.title('Sample Plot')
plt.xlabel('Time')
plt.ylabel('Temp [C]')
#plt.plot(df_weather.index, df_weather['Temp [C]'])
plt.show()


# In every column for the dataframe is some data which might be usefull. Therefore, it can be saved and used for the data preperation.

# In[10]:


df_weather.to_csv('df_weather.csv', index=True)


# In[ ]:




