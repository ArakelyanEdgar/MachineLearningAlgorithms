#polynomial regression

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

#getting dataset
dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#one hot encoding categorical data to dummy variables
#oneHotEncoder_X = OneHotEncoder(categorical_features=[0])
#X = oneHotEncoder_X.fit_transform(X).toarray()

#removing dummy variable to avoid dummy variable trap for both position and level data
#X = X[:, 1:]

#splitting data into test and training set, we actually choose not to do so because of small sample size
#X_train, x_test, y_train, y_test = train_test_split(X,y,  test_size=.2, random_state=0)

#linear non_polynomial
linReg = LinearRegression()
linReg.fit(X, y)

#visualizing linear regression
plt.scatter(X, y, color='red')
plt.plot(X, linReg.predict(X), color='blue')
plt.title('LEVEL TO SALARY (linear regression)')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
plt.show()

#polynomial regression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
linReg_2 = LinearRegression()
linReg_2.fit(X_poly, y)


#visualizing polynomial regression
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, linReg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('LEVEL TO SALARY (polynomial regression)')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
plt.show()


#compare results for 6.5
print(linReg.predict(13))
print(linReg_2.predict(poly_reg.fit_transform(13)))