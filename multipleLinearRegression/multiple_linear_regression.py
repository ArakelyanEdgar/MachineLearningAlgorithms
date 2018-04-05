# Multiple Linear Regression

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.set_printoptions(suppress=True, linewidth=10)

#read dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

#encoding categorical data
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
X_onehotencoder = OneHotEncoder(categorical_features=[3])
X = X_onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap i.e # of dummy variables = N - 1 where N is the number of categorical data values
X = X[:, 1:]

#splitting datasets into training sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#feature scaling
standardScaler_X = StandardScaler()
X_train = standardScaler_X.fit_transform(X_train)
X_test = standardScaler_X.transform(X_test)

#fitting to a linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#predicting test set results
y_pred = regressor.predict(X_test)


#building optimal model using BACKWARDS ELIMINATION----------------
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
#initialize with all independent variables
X_opt = X[:, [0,1,2,3,4,5]]

#fitting full model with all independent variables
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()


#removing x2
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()


#removing x1
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()


#removing new x2 which is column 4
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()


print(regressor_OLS.summary())

#removing new x2 which is column 5
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

#automated backwards selection
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)