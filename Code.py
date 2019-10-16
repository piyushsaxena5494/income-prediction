#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Importing training dataset in "training_data" variable
training_data = pd.read_csv("C:\\Users\\john\\Desktop\\MSc Data Science coursework\\Semester 1\\Machine Learning\\Assignments\\Kaggle Competition - 1\\tcdml1920-income-ind\\tcd ml 2019-20 income prediction training (with labels).csv");
#Importing test dataset in "test_data" variable
test_data = pd.read_csv("C:\\Users\\john\\Desktop\\MSc Data Science coursework\\Semester 1\\Machine Learning\\Assignments\\Kaggle Competition - 1\\tcdml1920-income-ind\\tcd ml 2019-20 income prediction test (without labels).csv")

#Preprocessing training and test dataset

#Replacing 0 with NaN values in training data
training_data = training_data.replace('0',np.nan)
#Replacing NaN values with most frequent value in that column in training data
training_data = training_data.fillna(method = 'ffill')
training_data.iat[0,2] = 'other'
#Replacing 0 with NaN values in test dataset
test_data = test_data.replace('0',np.nan)
#Replacing NaN values with most frequent value in that column in test data
test_data = test_data.fillna(method = 'ffill')

#Using Target Encoding to encode categorical data
mean_encode1 = training_data.groupby('Gender')['Income in EUR'].mean()
training_data.loc[:,'Gender Mean Encoded'] = training_data['Gender'].map(mean_encode1)
test_data.loc[:,'Gender Mean Encoded'] = test_data['Gender'].map(mean_encode1)

mean_encode2 = training_data.groupby('Country')['Income in EUR'].mean()
training_data.loc[:,'Country Mean Encoded'] = training_data['Country'].map(mean_encode2)
test_data.loc[:,'Country Mean Encoded'] = test_data['Country'].map(mean_encode2)

mean_encode3 = training_data.groupby('University Degree')['Income in EUR'].mean()
training_data.loc[:,'University Degree Mean Encoded'] = training_data['University Degree'].map(mean_encode3)
test_data.loc[:,'University Degree Mean Encoded'] = test_data['University Degree'].map(mean_encode3)

mean_encode4 = training_data.groupby('Profession')['Income in EUR'].mean()
training_data.loc[:,'Profession Mean encoded'] = training_data['Profession'].map(mean_encode4)
test_data.loc[:,'Profession Mean Encoded'] = test_data['Profession'].map(mean_encode4)

#Replacing NaN values with most frequent value in that new column in training data
training_data = training_data.fillna(method = 'ffill')
#Replacing NaN values with most frequent value in that new column in test data
test_data = test_data.fillna(method='ffill')

#Extracting the Independent and Dependent variables
X = training_data.iloc[:, [5,12,13,14,15]].values
y = training_data.iloc[:, 11].values
X1 = test_data.iloc[:, [5,12,13,14,15]].values

#Loading scikit's train_test_split library
from sklearn.model_selection import train_test_split
#Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

#Loading scikit's random forest regressor library
from sklearn.ensemble import RandomForestRegressor
#Creating a Random Forest Regressor
regressor = RandomForestRegressor(n_estimators = 200, random_state = 20)
#Training the Regressor
regressor.fit(X_train, y_train)
#Applying the trained Regressor to the test set and predicting the results
y_pred = regressor.predict(X_test)
#Applying the trained Regressor to the actual test data
y_pred1 = regressor.predict(X1)

#Loading scikit's mean squared error library
from sklearn.metrics import mean_squared_error
#Loading math's square root library
from math import sqrt
#Calulating root mean square error and printing it
rmse = sqrt(mean_squared_error(y_test, y_pred))
print (rmse)

#Creating a dataframe and storing it in "result" variable
result = pd.DataFrame(y_pred1)
#Converting dataframe stored in "result" variable to csv format and naming it as "Prediction_Results.csv"
result.to_csv("Prediction_Results.csv")