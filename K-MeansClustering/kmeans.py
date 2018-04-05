#Kmeans clustering

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#read dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
y = dataset.iloc[:, 4].values

#splitting datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

wcss = []
for i in range(1,11):
    KCluster = KMeans(n_clusters=i, random_state=0, init='k-means++', max_iter=300, n_init=10)
    KCluster.fit(X)
    wcss.append(KCluster.inertia_)

#visualizing dataset
#plt.plot(range(1,11), wcss)
#plt.title('ELBOW METHOD')
#plt.xlabel('# of clusters')
#plt.ylabel('# of Sum Squared Distance')
#plt.show()

#applying kmeans to mall dataset
KCluster = KMeans(n_clusters=5, init='k-means++', random_state=0, max_iter=300, n_init=10)
y_kmeans = KCluster.fit_predict(X)

#visualizing clusters
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label='CLUSTER 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label='CLUSTER 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label='CLUSTER 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label='CLUSTER 4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label='CLUSTER 5')
plt.scatter(KCluster.cluster_centers_[:,0], KCluster.cluster_centers_[:, 1], s=300, c='yellow', label='centroid')
plt.title('Clusters of clients')
plt.xlabel('Annual income')
plt.ylabel('Spending income')
plt.legend()
plt.show()