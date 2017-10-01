import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pylab as pl

# read csv file
df = pd.read_csv('SensorsDataSet/28_1800.csv')

table=df.pivot_table(index=["Name"])
table=table.reset_index()


#features order:
#0.Name, 1.AirTemp., 2.Alt, 3.DewPoint, 4.Lat, 5.Longt, 6.Pres., 7.R.Humidity, 8.Visib., 9.WindDir., 10.WindGust, 11.WindSpeed


# normalize data normalized_data=(x-min)/max-min

def normalize_2values(input_table):
    min_col3 = input_table.iloc[:, [3]].min()
    max_col3 =input_table.iloc[:,[3]].max()

    for i in range(len(input_table)):
        input_table.iloc[i,[3]]=((input_table.iloc[i,[3]]-min_col3)/(max_col3 - min_col3))

    min_col4 = input_table.iloc[:, [4]].min()
    max_col4 = input_table.iloc[:, [4]].max()

    for i in range(len(input_table)):
        input_table.iloc[i, [4]] = ((input_table.iloc[i, [4]] - min_col4) / (max_col4 - min_col4))
    return input_table

# change the order of features
table2=table.iloc[:,[1,2,3,6,7,8,9,10,11,0,4,5]]
#table2=table.iloc[:,[0]]
#print("table2.tail()")
print(table2.tail())

# specific features
table3=table.iloc[:,[0,4,5,1,7]]
table3=table3.dropna()
min_feature1=table3.iloc[:, [3]].min()
max_feature1=table3.iloc[:, [3]].max()
min_feature2=table3.iloc[:, [4]].min()
max_feature2=table3.iloc[:, [4]].max()
table3_normalized=normalize_2values(table3)

print(table3_normalized.tail())
feature1="RelativeHumidity"
feature2="AirTemp"

# k=clustering number
k=4
cluster=KMeans(n_clusters=k)
table3["Cluster"]=cluster.fit_predict(table3_normalized[table3_normalized.columns[3:]])
centroids = cluster.cluster_centers_


weather_clusters=table3[["Name","Lat","Longt","Cluster",feature1,feature2]]
print(weather_clusters)
weather_clusters.to_csv('ClusteredData.csv', index=False)

print("centroids")
print(table3.columns)

# before denormalizing, centroids:
denormalized_centroids=centroids
print("normalized centroids:")
print(centroids)


for r in range(len(denormalized_centroids)):
    denormalized_centroids[r][0] = (denormalized_centroids[r][0] * (max_feature1 - min_feature1)) + min_feature1
    denormalized_centroids[r][1] = (denormalized_centroids[r][1] * (max_feature2 - min_feature2))+ min_feature2


# after denormalizing, centroids:
print("denormalized centroids:", feature1,feature2 )
print(denormalized_centroids)


#plotting clusters
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
    print("Cluster0 Color: Red, Cluster1 Color: Green")

elif k==3:
    x_0 = weather_clusters[weather_clusters.Cluster == 0]["Longt"]
    y_0 = weather_clusters[weather_clusters.Cluster == 0]["Lat"]
    c1 = pl.scatter(x_0, y_0, c='r', marker='o', alpha=0.4)
    x_1 = weather_clusters[weather_clusters.Cluster == 1]["Longt"]
    y_1 = weather_clusters[weather_clusters.Cluster == 1]["Lat"]
    c3 = pl.scatter(x_1, y_1, c='g', marker='o', alpha=0.4)
    #c2 = pl.scatter(x_1, y_1, c='g', marker='x', alpha=0.8, s=169, linewidths=3, zorder=10)
    x_2=weather_clusters[weather_clusters.Cluster==2]["Longt"]
    y_2=weather_clusters[weather_clusters.Cluster == 2]["Lat"]
    c3=pl.scatter(x_2,y_2,c='b',marker='o', alpha=0.4)
    # Numbers of Elements in Clusters
    print("Cluster0 Size:", len(x_0), ", Cluster1 Size:", len(x_1), ", Cluster2 Size:", len(x_2))
    print("Cluster0 Color: Red, Cluster1 Color: Green, Cluster2 Color: Blue")

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
    print("Cluster0 Color: Red, Cluster1 Color: Green, Cluster2 Color: Blue, Cluster3 Color: Yellow")


pl.xlabel('Longitude')
pl.ylabel('Latitude')
pl.title('AirTemp, RelativeHumidity')
pl.savefig("plot_output.png")
pl.show()





