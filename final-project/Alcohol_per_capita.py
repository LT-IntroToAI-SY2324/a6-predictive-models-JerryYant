import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#import data
data = pd.read_csv("final-project/alcohol_per_capita.csv")
x = data[["TIME", "LITRES/CAPITA"]]

x_std = StandardScaler().fit_transform(x)

#K means
k = 100 #Unsure about what value to set k at
km = KMeans(n_clusters=k).fit(x_std)

#Centroid and lable
centroids = km.cluster_centers_
labels = km.labels_

#Plotting data into clusters
for i in range(k):
    cluster = x_std[labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1])

#plotting centroids
plt.scatter(cluster[:, 0], cluster[:, 1], marker = "X", s = 100, c = "r", label = 'centroid')

#graph
plt.xlabel("TIME")
plt.ylabel("LITRES/CAPITA")
plt.show()