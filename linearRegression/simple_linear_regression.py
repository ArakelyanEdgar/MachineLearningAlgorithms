import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.set_printoptions(precision=3, linewidth=200,suppress=True)

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

#Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#fitting simple linear regression to training set
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the Test set Results
y_pred = regressor.predict(X_test)

#visualizing the  results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()