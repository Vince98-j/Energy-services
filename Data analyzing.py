#!/usr/bin/env python
# coding: utf-8

# In[615]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb 
from scipy import stats
import numpy as np


# # Goal assignment
# 
# In the last assignment for energy services a model is trained and used to forecast the generated solar power for the Dutch island Texel. Noord Holland is a province in the Netherlands and Texel is the only island belonging to Noord Holland. The province is coloured red in the map below. The black arrow is pointing at Texel. As the amount of installed solar power is increasing every year should a factor implied for future estimations. This will be explained and investigated later.
# ![image-2.png](attachment:image-2.png)
# 
# # Data preperation
# First the data is prepared. The dataframes are imported, merged and NaN are deleted.

# In[616]:


df_solar = pd.read_csv('df_power_hourly.csv')
df_weather = pd.read_csv('df_weather.csv')

print(df_weather)
print(df_solar)


# check for any NaN

# In[617]:


df_solar[df_solar.isnull().any(axis = 'columns')]


# In[618]:


df_weather[df_weather.isnull().any(axis = 'columns')]


# No NaN so the dataframes can be merged

# In[619]:


df_hourly = pd.concat([df_solar, df_weather], axis=1) 
df_hourly = df_hourly.drop (columns = ['date']) 


# Check for empty columns again. A lot of weather data is missing in 2018. This is important data for testing the model later in the assignment.

# In[620]:


df_hourly[df_hourly.isnull().any(axis = 'columns')]
df_hourly=df_hourly.dropna()
df_hourly


# In[621]:


#set index to date
df_hourly['Date'] = pd.to_datetime(df_hourly['Date'], format='%d-%m-%Y %H:%M')
df_hourly = df_hourly.set_index('Date', drop = True)


# In[622]:


fig, axes = plt.subplots(nrows=15, ncols=1, figsize=(15, 30))

# Flatten the axes array to iterate over each subplot
axes = axes.flatten()

# List of column names to plot
columns_to_plot = ['solar_power_kW', 'Temp [C]', 'precipitation [mm]', 'rain [mm]', 'snowfall [cm]', 'cloud [%]', 'cloud_l [%]', 'cloud_m [%]', 'cloud_h [%]', 'wind [km/h]', 'GHI [W/m²]', 'DR [W/m²]', 'DNI [W/m²]','is_day', 'sunshine_duration']

# Iterate over each column and plot on its corresponding subplot
for i, column in enumerate(columns_to_plot):
    ax = axes[i]  # Select the current subplot
    ax.plot(df_hourly[column])  # Plot the graphs
    ax.set_title(column, fontsize=15)  # Set subplot title

# Adjust layout
plt.tight_layout()


# # Data analysis
# The data is analyzed to see if there are strange values and to decided what kind of cleaning should be done.

# In[623]:


df_hourly.describe()


# In[624]:


df_hourly_check = df_hourly.sort_values(by = 'solar_power_kW', ascending = True)
df_hourly_check [:9]


# The data description does not show any strange values, the heighest measured solar power is lower than the registered solar power, which makes sense. As it is almost impossible that every solar panel is generating the max possible power at the same time for an hour long.
# 
# Cloud cover does not go beyond 100% which is correct. Living in the Netherlands, I know 100% cloud cover happens all the time :').
# 
# The maximum windspeed is heigh, but it is not uncommen at it is an island next to the North sea, which is one of the roughest seas in the world.
# 
# The sunshine duration is also normal, as this is not going above the 3600 s per hour.
# 
# ### Boxplot analysis

# In[625]:


fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 10))

# Flatten the axes array to iterate over each subplot
axes = axes.flatten()

# List of column names to plot
columns_to_plot = ['solar_power_kW', 'Temp [C]', 'precipitation [mm]', 'rain [mm]', 'snowfall [cm]', 'cloud [%]', 'cloud_l [%]', 'cloud_m [%]', 'cloud_h [%]', 'wind [km/h]', 'GHI [W/m²]', 'DR [W/m²]', 'DNI [W/m²]','is_day', 'sunshine_duration']

# Iterate over each column and plot on its corresponding subplot
for i, column in enumerate(columns_to_plot):
    ax = axes[i]  # Select the current subplot
    ax.boxplot(df_hourly[column])  # Plot the boxplot
    ax.set_title(column, fontsize=16)  # Set subplot title

# Adjust layout
plt.tight_layout()


# # Outlier removal by z-score

# In[626]:


columns_to_calculate_zscore = ['solar_power_kW', 'Temp [C]', 'precipitation [mm]', 'rain [mm]', 'snowfall [cm]', 'cloud [%]', 'cloud_l [%]', 'cloud_m [%]', 'cloud_h [%]', 'wind [km/h]', 'GHI [W/m²]', 'DR [W/m²]', 'DNI [W/m²]','is_day', 'sunshine_duration']

# Select the columns from the DataFrame
data_to_calculate_zscore = df_hourly[columns_to_calculate_zscore]

# Calculate the z-score
z = np.abs(stats.zscore(data_to_calculate_zscore))

threshold = 3 # 3 sigma...Includes 99.7% of the data
z>threshold
np.where(z > threshold)

df_hourly_deleted_z_score = df_hourly[(z < threshold)]
df_hourly_deleted_z_score = df_hourly_deleted_z_score.dropna()


# In[627]:


fig, axes = plt.subplots(nrows=14, ncols=1, figsize=(15, 30))

# Flatten the axes array to iterate over each subplot
axes = axes.flatten()

# List of column names to plot
columns_to_plot = ['solar_power_kW', 'Temp [C]', 'precipitation [mm]', 'rain [mm]', 'snowfall [cm]', 'cloud [%]', 'cloud_l [%]', 'cloud_m [%]', 'cloud_h [%]', 'wind [km/h]', 'GHI [W/m²]', 'DR [W/m²]', 'DNI [W/m²]', 'sunshine_duration']


# Iterate over each column and plot on its corresponding subplot
for i, column in enumerate(columns_to_plot):
    ax = axes[i]  # Select the current subplot
    ax.plot(df_hourly[column])  # Plot the graphs
    ax.plot(df_hourly_deleted_z_score[column])
    ax.set_title(column, fontsize=15)  # Set subplot title

# Adjust layout
plt.tight_layout()


# Blue is the original data and orange is after the z-score removal.
#  
# According to the graphs, the z-score has the most effect on the solarpower, radiance columns and all kinds of precipitation. 
# 
# At first, I thought it was not a good idea to delete the z-score, as the measured data is not wrong and results in a decrease of the peaks for the mentioned columns. Therefore, I choose to not remove any outliers. Later, in chosing a regression model, I realized that deleting the z-score improved the models. Therefore, I do delete it.

# # Feature selection
# In this part of the code, certain features are chosen for the training of the model. First, the columns are reorganized.

# In[628]:


df_hourly_fs=df_hourly_deleted_z_score.iloc[:, [0,10,11,12,5,6,7,8,1,2,3,4,9,13,14]] #change position of the columns
df_hourly_fs


# First, I did not delete the columns. However, the regression model, was not performing well and read that it is good to delete columns with similar data. Eliminating these columns before the feature selection improved the outcome of the regression models.

# In[629]:


df_hourly_fs_1 = df_hourly_fs.drop(columns=['GHI [W/m²]', 'DNI [W/m²]', 'cloud_l [%]', 'cloud_m [%]', 'cloud_h [%]', 'is_day'])


# Add some extra features as an hour column and solar_power-1.

# In[630]:


df_hourly_fs_1['Hour'] = df_hourly_fs_1.index.hour
time_diff = (df_hourly_fs_1.index.to_series().diff().dt.total_seconds() / 3600)

# Set the shifted power value to NaN where the time difference is greater than 2 hours
df_hourly_fs_1['solar_power-1'] = df_hourly_fs_1['solar_power_kW'].shift(1)
df_hourly_fs_1.loc[time_diff > 2, 'solar_power-1'] = pd.NA
df_hourly_fs_1 = df_hourly_fs_1.dropna()


# In[631]:


#Z=df_hourly_fs.values
Z=df_hourly_fs_1.values

Y=Z[:,0]
#X=Z[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
X=Z[:, [1,2,3,4,5,6,7,8,9,10]]
print(Y)
print(X)


# # Filter method

# In[632]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression,f_regression


# In[633]:


features=SelectKBest(k=5,score_func=f_regression) # Test different k number of features, uses f-test ANOVA

fit=features.fit(X,Y) #calculates the scores using the score_function f_regression of the features

# This paragraph of code tells the colomns which are important
selected_features_indices = fit.get_support(indices=True)
X_df = pd.DataFrame(X) # Convert the NumPy array to a pandas DataFrame
selected_feature_names = X_df.columns[selected_features_indices]

print(fit.scores_)
print("Selected Features:", selected_feature_names) 
features_results=fit.transform(X)
print(features_results) 


# In[634]:


plt.bar([i for i in range(len(fit.scores_))], fit.scores_)


# # Wrapper method
# 
# I did the wrapper method, but it selects not good features at all. I tried it for the regression, but it selects all the precipitation columns and the outcome for every regression model is more or less a horizontal line. Therefore, this is not chosen.

# In[635]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[636]:


model=LinearRegression() # LinearRegression Model as Estimator
rfe1=RFE(model,n_features_to_select=1)# using 1 features
rfe2=RFE(model,n_features_to_select=2)# using 2 features
rfe3=RFE(model,n_features_to_select=3)# using 3 features
rfe4=RFE(model,n_features_to_select=4)# using 4 features
fit1=rfe1.fit(X,Y)
fit2=rfe2.fit(X,Y)
fit3=rfe3.fit(X,Y)
fit4=rfe4.fit(X,Y)

print( "Feature Ranking (Linear Model, 1 features): %s" % (fit1.ranking_)) # windspeed
print( "Feature Ranking (Linear Model, 2 features): %s" % (fit2.ranking_)) # windspeed, day_of_week
print( "Feature Ranking (Linear Model, 3 features): %s" % (fit3.ranking_)) # windspeed, windGust, day_of_week 
print( "Feature Ranking (Linear Model, 4 features): %s" % (fit4.ranking_)) # windspeed, windGust, day_of_week 

#The lower the ranking, the more important it is.


# In[637]:


#df_regression=df_hourly_fs.drop(columns=['cloud [%]', 'cloud_l [%]', 'cloud_m [%]',  'cloud_h [%]', 'Temp [C]',
 #                                        'precipitation [mm]', 'rain [mm]', 'snowfall [cm]', 'sunshine_duration'])

#df_regression = df_hourly_fs_1.drop(columns=['cloud [%]','precipitation [mm]', 'rain [mm]', 'snowfall [cm]', 'wind [km/h]',
 #                                            'sunshine_duration'])

# BEST SO FAR df_regression = df_hourly_fs_1.drop(columns= ['cloud [%]','precipitation [mm]', 'rain [mm]', 'snowfall [cm]', 'wind [km/h]'])

df_regression = df_hourly_fs_1.drop(columns= ['cloud [%]','precipitation [mm]', 'rain [mm]', 'snowfall [cm]', 'wind [km/h]', 'Hour'])


# The chosen features are presented in the following column.

# In[638]:


df_regression=df_regression.iloc[:,[0,4,1,3,2]]
df_regression


# # Regression
# 
# In this part of the code are regression models tested. At the end of this section a summary of all the regression outcomes are presented in a table and compared. From this table a model will be chosen and optimized.

# In[639]:


from sklearn.model_selection import train_test_split
from sklearn import  metrics
import statsmodels.api as sm


# In[640]:


from statsmodels.tsa.ar_model import AutoReg
#from statsmodels.tsa.arima.model import ARIMA

#Identify output Y
Y=df_regression.values[:,0]

split_point = len(Y) - 1000
train, test = Y[0:split_point], Y[split_point:]
# train autoregression
window = 1
model = AutoReg(train, lags=window)
#model=ARIMA(train, order=(1, 1,1))
model_fit = model.fit()

print(model_fit.summary())


# In[641]:


#model_fit.plot_predict()
plt.plot(train,label='Real')
plt.plot(model_fit.fittedvalues,label='AR')
plt.legend()


# # Testing the AR model
# 

# In[642]:


from sklearn.metrics import mean_squared_error
from math import sqrt

coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

plt.scatter(test,predictions)
plt.show()


# In[643]:


MAE_AR=metrics.mean_absolute_error(test,predictions) # Mean Absolute Error 
MBE_AR=np.mean(test-predictions) # Mean Bias Error
MSE_AR=metrics.mean_squared_error(test,predictions)  # Mean Squared Error
RMSE_AR= np.sqrt(metrics.mean_squared_error(test,predictions))# Root Mean Squared Error
cvRMSE_AR=RMSE_AR/np.mean(test) # Coefficient of Variation of RMSE
NMBE_AR=MBE_AR/np.mean(test) # Normalized Mean Bias Error
print(MAE_AR, MBE_AR,MSE_AR, RMSE_AR,cvRMSE_AR,NMBE_AR)


# # Split Data into training and test data¶

# In[644]:


#Create matrix from data frame
Z=df_regression.values
#Identify output Y
Y=Z[:,0]
#Identify input Y
#X=Z[:,[1,2,3,4,5]]
X=Z[:,[1,2,3,4]]

#X2=Z[:,[1,4]]
#X2=Z[:,[1,2,3]]


# In[645]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets with a 90-10 ratio
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
# Adjust the test_size parameter to 0.1 for a 90-10 split

print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))


# # Linear Regression

# In[646]:


from sklearn import  linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred_LR = regr.predict(X_test)


# In[647]:


plt.plot(y_test[1:200])
plt.plot(y_pred_LR[1:200])
plt.show()
plt.scatter(y_test,y_pred_LR)


# In[648]:


#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MBE_LR=np.mean(y_test- y_pred_LR) #here we calculate MBE
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
NMBE_LR=MBE_LR/np.mean(y_test)
print(MAE_LR, MBE_LR,MSE_LR, RMSE_LR,cvRMSE_LR,NMBE_LR)


# # Decision Tree Regressor

# In[649]:


from sklearn.tree import DecisionTreeRegressor

# Create Regression Decision Tree object
DT_regr_model = DecisionTreeRegressor(min_samples_leaf=5)

# Train the model using the training sets
DT_regr_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_test)


# In[650]:


plt.plot(y_test[1000:2000])
plt.plot(y_pred_DT[1000:2000])
plt.show()
plt.scatter(y_test,y_pred_DT)


# In[651]:


#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT) 
MBE_DT=np.mean(y_test-y_pred_DT) #here we calculate MBE
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)
NMBE_DT=MBE_DT/np.mean(y_test)
print(MAE_DT, MBE_DT,MSE_DT, RMSE_DT,cvRMSE_DT,NMBE_DT)


# # Random forest

# In[652]:


from sklearn.ensemble import RandomForestRegressor


# In[653]:


parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 10,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)


# In[654]:


plt.plot(y_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()
plt.scatter(y_test,y_pred_RF)


# In[655]:


#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MBE_RF=np.mean(y_test-y_pred_DT) #here we calculate MBE
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
NMBE_RF=MBE_RF/np.mean(y_test)
print(MAE_RF,MBE_RF,MSE_RF,RMSE_RF,cvRMSE_RF,NMBE_RF)


# # Uniformized data

# In[656]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[657]:


parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 10,
              'max_leaf_nodes': None}

RF_model2 = RandomForestRegressor(**parameters)
RF_model2.fit(X_train_scaled, y_train.reshape(-1,1))
y_pred_RF2 = RF_model2.predict(X_test_scaled)


# In[658]:


plt.plot(y_test[1:200])
plt.plot(y_pred_RF2[1:200])
plt.show()
plt.scatter(y_test,y_pred_RF2)


# In[659]:


#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF2) 
MBE_RF=np.mean(y_test-y_pred_RF2) #here we calculate MBE
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF2)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF2))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
NMBE_RF=MBE_RF/np.mean(y_test)
print(MAE_RF,MBE_RF,MSE_RF,RMSE_RF,cvRMSE_RF,NMBE_RF)


# # Gradient Boosting

# In[660]:


from sklearn.ensemble import GradientBoostingRegressor

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GB_model = GradientBoostingRegressor(**params)

GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB = GB_model.predict(X_test)


# In[661]:


plt.plot(y_test[1:200])
plt.plot(y_pred_GB[1:200])
plt.show()
plt.scatter(y_test,y_pred_GB)


# In[662]:


MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MBE_GB=np.mean(y_test-y_pred_GB)
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)
NMBE_GB=MBE_GB/np.mean(y_test)
print(MAE_GB,MBE_GB,MSE_GB,RMSE_GB,cvRMSE_GB,NMBE_GB)


# # Neural Networks

# In[663]:


from sklearn.neural_network import MLPRegressor

NN_model = MLPRegressor(hidden_layer_sizes=(5,5,5))
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)


# In[664]:


plt.plot(y_test[1:200])
plt.plot(y_pred_NN[1:200])
plt.show()
plt.scatter(y_test,y_pred_NN)


# In[665]:


MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN)
MBE_NN=np.mean(y_test-y_pred_NN)
MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)
NMBE_NN=MBE_NN/np.mean(y_test)
print(MAE_NN,MBE_NN,MSE_NN,RMSE_NN,cvRMSE_NN,NMBE_NN)


# # Summarizing and evaluating the different regression models

# In[666]:


data = [
    ['Auto regression', MAE_AR, MBE_AR, MSE_AR, RMSE_AR, cvRMSE_AR, NMBE_AR],
    ['linear regression', MAE_LR, MBE_LR, MSE_LR, RMSE_LR, cvRMSE_LR, NMBE_LR],
    ['Decision tree', MAE_DT, MBE_DT, MSE_DT, RMSE_DT, cvRMSE_DT, NMBE_DT],
    ['Random forest', MAE_RF, MBE_RF, MSE_RF, RMSE_RF, cvRMSE_RF, NMBE_RF],
    ['Gradient Boosting', MAE_GB, MBE_GB, MSE_GB, RMSE_GB, cvRMSE_GB, NMBE_GB],
    ['Neural Network', MAE_NN, MBE_NN, MSE_NN, RMSE_NN, cvRMSE_NN, NMBE_NN]
]

columns = ['Type Model', 'MAE', 'MBE', 'MSE', 'RMSE', 'cvRMSE', 'NMBE']

# Create a DataFrame
df_evaluating_regression = pd.DataFrame(data, columns=columns)

# Set the index to 'Reg_model' column
df_evaluating_regression = df_evaluating_regression.set_index('Type Model', drop=True)


# In[667]:


df_evaluating_regression


# # Random forest and Gradient Boosting combined
# 
# The combination of these two regressors show an improvement for the MAE, but all other errors are increasing unfortunately.

# In[668]:


import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Initialize Random Forest and Gradient Boosting models
rf_model = RandomForestRegressor()
gb_model = GradientBoostingRegressor()

# Train the models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Make predictions on the validation set
rf_preds = rf_model.predict(X)
gb_preds = gb_model.predict(X)

# Blend the predictions
blended_preds = (rf_preds + gb_preds) / 2

# Apply bottom limit of zero to the blended predictions
blended_preds = np.maximum(blended_preds, 0)

# Calculate metrics
blended_mae = mean_absolute_error(Y, blended_preds)

# Print the metrics
print("Blended MAE:", blended_mae)


# In[669]:


import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(Y, blended_preds, color='blue', label='Predictions')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2, color='red', label='Ideal Prediction')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Scatter Plot)')
plt.legend()
plt.grid(True)
plt.show()

# Normal plot
plt.figure(figsize=(10, 6))
plt.plot(Y, label='Actual', color='blue')
plt.plot(blended_preds, label='Predicted', color='red')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Actual vs Predicted (Normal Plot)')
plt.legend()
plt.grid(True)
plt.show()


# # Improving the random forest regressor

# ============================================================================================================
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# 
# # Define hyperparameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }
# 
# # Instantiate the model
# rf_model = RandomForestRegressor()
# 
# # Grid search
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)
# 
# # Get the best hyperparameters
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)
# 
# # Evaluate performance on the validation set
# best_model = grid_search.best_estimator_
# mse = mean_squared_error(y_test, best_model.predict(X_test))
# print("Validation MSE:", mse)
# 
# ============================================================================================================

# # Save models

# In[670]:


import pickle


# In[671]:


with open ('RF_model.pkl','wb') as file:
    pickle.dump(RF_model, file)


# In[672]:


with open ('GB_model.pkl','wb') as file:
    pickle.dump(GB_model, file)


# In[ ]:




