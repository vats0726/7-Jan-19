import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm


df = pd.read_csv('accdata.csv')                          # Importing the dataset
df = df[['Close']]                                       # Choosing Required Data
forecast_out = int(1)                                    # Predicting 10 days into future
df['Prediction'] = df[['Close']].shift(-forecast_out)    # Label column with data shifted 10 units up
X = np.array(df.drop(['Prediction'], 1))                 # Dropping a column and creating a numpy array
 X = preprocessing.scale(X)                              # Standardisation (Feature Scaling)
X_forecast = X[-forecast_out:]                           # Set X_forecast equal to last 10
X = X[:-forecast_out]                                    # Remove last 10 from X
y = np.array(df['Prediction'])                           # Creating a numpy Array of Prediction Label df
y = y[:-forecast_out]                                    # Removing Last 10 NaN values               
regressor = LinearRegression()                           # regressor is the object of class LinearRegression
regressor.fit(X,y)                                       # Training the object with the dataset
forecast_prediction = regressor.predict(X_forecast)      # Predicting using regression
print(forecast_prediction)                               # Printing the Prediction of closing price for next 10 days
