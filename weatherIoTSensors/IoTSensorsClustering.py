import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pylab as pl

# read csv file
df = pd.read_csv('SensorsDataSet/28_1800.csv')
table=df.pivot_table(index=["Name"])
table=table.reset_index()

# features order:
# 0.Name, 1.AirTemp., 2.Alt, 3.DewPoint, 4.Lat, 5.Longt, 6.Pres., 7.R.Humidity, 8.Visib., 9.WindDir., 10.WindGust, 11.WindSpeed

# change the order of features
table2=table.iloc[:,[1,2,3,6,7,8,9,10,11,0,4,5]]
print(table2.tail())

# specific features - new table is created
# enter the fourth column to address feature e.g: 1. AirTemp, 7. Relative Humidity
# e.g: table3=table.iloc[:,[0,4,5,1]]
table3=table.iloc[:,[0,4,5,7]]
table3=table3.dropna()

# for debug print tail of table3
print(table3.tail())
title="RelativeHumidity"


# k=clustering number
k=4
cluster=KMeans(n_clusters=k)
table3["Cluster"]=cluster.fit_predict(table3[table3.columns[3:]])
# centroids = clusters' center points
centroids = cluster.cluster_centers_

weather_clusters=table3[["Name","Lat","Longt","Cluster",title]]
print(weather_clusters)
weather_clusters.to_csv('ClusteredData.csv', index=False)


# informative
print("centroids")
print(table3.columns)
print(centroids)

# plotting clusters
plt.figure(num=None, figsize=(8, 6), dpi=80)

if k==2:
    x_0=weather_clusters[weather_clusters.Cluster==0]["Longt"]
    y_0=weather_clusters[weather_clusters.Cluster == 0]["Lat"]
    c1=pl.scatter(x_0,y_0,c='r',marker='o',alpha=0.4)
    x_1=weather_clusters[weather_clusters.Cluster==1]["Longt"]
    y_1=weather_clusters[weather_clusters.Cluster == 1]["Lat"]
    c2=pl.scatter(x_1,y_1,c='g',marker='o',alpha=0.4)
    # Numbers of Elements in Clusters
    print("Cluster0 Size:",len(x_0), ", Cluster1 Size:",len(x_1))

    # Print Cluster Max, Min Points to determine Cluster Seperation Point
    max_c0 = max(weather_clusters[weather_clusters.Cluster == 0][title])
    min_c0 = min(weather_clusters[weather_clusters.Cluster == 0][title])
    print("max_c0:", max_c0, " min_c0:", min_c0, "Color:R")
    max_c1 = max(weather_clusters[weather_clusters.Cluster == 1][title])
    min_c1 = min(weather_clusters[weather_clusters.Cluster == 1][title])
    print("max_c1:", max_c1, " min_c1:", min_c1, "Color:G")

elif k==3:
    x_0 = weather_clusters[weather_clusters.Cluster == 0]["Longt"]
    y_0 = weather_clusters[weather_clusters.Cluster == 0]["Lat"]
    c1 = pl.scatter(x_0, y_0, c='r', marker='o', alpha=0.4)
    x_1 = weather_clusters[weather_clusters.Cluster == 1]["Longt"]
    y_1 = weather_clusters[weather_clusters.Cluster == 1]["Lat"]
    c2 = pl.scatter(x_1, y_1, c='g', marker='o', alpha=0.4)
    # for sensor fault visibility in figure
    # c2 = pl.scatter(x_1, y_1, c='b', marker='x', alpha=1, s=300, linewidths=4, zorder=10)
    x_2=weather_clusters[weather_clusters.Cluster==2]["Longt"]
    y_2=weather_clusters[weather_clusters.Cluster == 2]["Lat"]
    #c3 = pl.scatter(x_2, y_2, c='b', marker='x', alpha=1, s=300, linewidths=4, zorder=10)
    c3=pl.scatter(x_2,y_2,c='b',marker='o', alpha=0.4)
    # Numbers of Elements in Clusters
    print("Cluster0 Size:", len(x_0), ", Cluster1 Size:", len(x_1), ", Cluster2 Size:", len(x_2))

    # Print Cluster Max, Min Points to determine Cluster Seperation Point
    max_c0 = max(weather_clusters[weather_clusters.Cluster == 0][title])
    min_c0 = min(weather_clusters[weather_clusters.Cluster == 0][title])
    print("max_c0:", max_c0, " min_c0:", min_c0, "Color:R")
    max_c1 = max(weather_clusters[weather_clusters.Cluster == 1][title])
    min_c1 = min(weather_clusters[weather_clusters.Cluster == 1][title])
    print("max_c1:", max_c1, " min_c1:", min_c1, "Color:G")
    max_c2 = max(weather_clusters[weather_clusters.Cluster == 2][title])
    min_c2 = min(weather_clusters[weather_clusters.Cluster == 2][title])
    print("max_c2:", max_c2, " min_c2:", min_c2, "Color:B")
elif k==4:
    x_0 = weather_clusters[weather_clusters.Cluster == 0]["Longt"]
    y_0 = weather_clusters[weather_clusters.Cluster == 0]["Lat"]
    c1 = pl.scatter(x_0, y_0, c='r', marker='o', alpha=0.4)
    x_1 = weather_clusters[weather_clusters.Cluster == 1]["Longt"]
    y_1 = weather_clusters[weather_clusters.Cluster == 1]["Lat"]
    #c2 = pl.scatter(x_1, y_1, c='g', marker='x', alpha=0.8, s=169, linewidths=3, zorder=10)
    c2 = pl.scatter(x_1, y_1, c='g', marker='o', alpha=0.4)
    x_2 = weather_clusters[weather_clusters.Cluster == 2]["Longt"]
    y_2 = weather_clusters[weather_clusters.Cluster == 2]["Lat"]
    c3 = pl.scatter(x_2, y_2, c='b', marker='o', alpha=0.4)
    x_3=weather_clusters[weather_clusters.Cluster==3]["Longt"]
    y_3=weather_clusters[weather_clusters.Cluster == 3]["Lat"]
    c3=pl.scatter(x_3,y_3,c='y',marker='o', alpha=0.4)
    # Numbers of Elements in Clusters
    print("Cluster0 Size:", len(x_0), ", Cluster1 Size:", len(x_1), ", Cluster2 Size:", len(x_2), ", Cluster3 Size:", len(x_3))

    # Print Cluster Max, Min Points to determine Cluster Seperation Point
    max_c0 = max(weather_clusters[weather_clusters.Cluster == 0][title])
    min_c0 = min(weather_clusters[weather_clusters.Cluster == 0][title])
    print("max_c0:", max_c0, " min_c0:", min_c0, "Color:R")
    max_c1 = max(weather_clusters[weather_clusters.Cluster == 1][title])
    min_c1 = min(weather_clusters[weather_clusters.Cluster == 1][title])
    print("max_c1:", max_c1, " min_c1:", min_c1, "Color:G")
    max_c2 = max(weather_clusters[weather_clusters.Cluster == 2][title])
    min_c2 = min(weather_clusters[weather_clusters.Cluster == 2][title])
    print("max_c2:", max_c2, " min_c2:", min_c2, "Color:B")
    max_c3 = max(weather_clusters[weather_clusters.Cluster == 3][title])
    min_c3 = min(weather_clusters[weather_clusters.Cluster == 3][title])
    print("max_c3:", max_c3, " min_c3:", min_c3, "Color:Y")

pl.xlabel('Longitude')
pl.ylabel('Latitude')
pl.title(title)
pl.savefig("plot_output.png")
pl.show()





