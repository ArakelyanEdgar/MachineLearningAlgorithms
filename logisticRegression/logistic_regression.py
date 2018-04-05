#logisitc regression   

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

np.set_printoptions(suppress=True)

#getting datasets
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

#splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#normalizing age and estimated salary
StandardScaler_X = StandardScaler()
X_train = StandardScaler_X.fit_transform(X_train)
X_test = StandardScaler_X.transform(X_test)

#logistic regression
classifier = LogisticRegression(random_state=0, solver='liblinear')
classifier.fit(X_train, y_train)


#predicting test_set results
y_pred = classifier.predict(X_test)
print(y_pred)

#Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#visualizing classifier model
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid( np.arange(start=X_set[:, 0].min()-1, stop=X_set[:,0].max()+1, step=0.01),  
                        np.arange(start=X_set[:, 1].min()-1, stop=X_set[:,1].max()+1, step=0.01) )
plt.contourf(X1, X2, classifier.predict( np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap= ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set==j, 1], c = ListedColormap(('red', 'green'))(i), label=j )

plt.title('LOGISTIC REGRESSION (TRAINING SET')
plt.xlabel('AGE')
plt.ylabel('Est Salary')
plt.legend()

wished_xticks = np.arange(min(np.array(X)[:, 0]-6), max(np.array(X)[:, 0])+10, 2)
temp_x = np.c_[ wished_xticks, [0]*len(wished_xticks) ]
transformed_x = StandardScaler_X.transform(temp_x)[:,0]
plt.xticks(transformed_x, wished_xticks)
 
wished_yticks = np.arange(min(np.array(X)[:, 1]-10000), max(np.array(X)[:, 1])+15000, 10000)
temp_y = np.c_[ [0]*len(wished_yticks), wished_yticks ]
transformed_y = StandardScaler_X.transform(temp_y)[:,1]
plt.yticks(transformed_y, wished_yticks)

plt.show() 

#visualize test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid( np.arange(start=X_set[:, 0].min()-1, stop=X_set[:,0].max()+1, step=0.01),  
                        np.arange(start=X_set[:, 1].min()-1, stop=X_set[:,1].max()+1, step=0.01) )
plt.contourf(X1, X2, classifier.predict( np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap= ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set==j, 1], c = ListedColormap(('red', 'green'))(i), label=j )

plt.title('LOGISTIC REGRESSION (TEST SET)')
plt.xlabel('AGE')
plt.ylabel('Est Salary')
plt.legend()

wished_xticks = np.arange(min(np.array(X)[:, 0]-6), max(np.array(X)[:, 0])+10, 2)
temp_x = np.c_[ wished_xticks, [0]*len(wished_xticks) ]
transformed_x = StandardScaler_X.transform(temp_x)[:,0]
plt.xticks(transformed_x, wished_xticks)
 
wished_yticks = np.arange(min(np.array(X)[:, 1]-10000), max(np.array(X)[:, 1])+15000, 10000)
temp_y = np.c_[ [0]*len(wished_yticks), wished_yticks ]
transformed_y = StandardScaler_X.transform(temp_y)[:,1]
plt.yticks(transformed_y, wished_yticks)

plt.show() 
