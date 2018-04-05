#SVR

#importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

#retrieve dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#featureScale X
X_Scaler = StandardScaler()
y_Scaler = StandardScaler()
X = X_Scaler.fit_transform(X)
y= y.reshape(-1,1)
y = y_Scaler.fit_transform(y)

#svr regressor
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

#visualizing SVR results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('SVR Regression')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
plt.show()

print(y_Scaler.inverse_transform( regressor.predict(X_Scaler.transform(np.array([[6.5]])) )))