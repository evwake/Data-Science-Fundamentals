import pandas as pd
import numpy as np

# Silhouette coefficient must be computed using this equation:
# si = |1 - (ai/bi)|
# You will not get the same values if you use a different version of that equation!
# How you structure your code is up to you

def dist(a, b):
    # Compute Euclidean distance between a and b
    return np.sum((np.array(a)-np.array(b))**2)**(0.5)

def calculate_distance_for_cluster(x, cluster_data):
    distances = []
    for j in cluster_data.iterrows():
        y = j[1]
        if x['Instance'] != y['Instance']:
            x_vals = x[1:5]
            y_vals = y[1:5]
            distances.append(dist(x_vals, y_vals))
    
    return sum(distances) / len(distances)



for k in [2, 3, 5, 7]:
    df = pd.read_csv("../data/k" + str(k) + ".csv" )
    total_sc = [0 for j in range(k)]
    cluster_names = df['Cluster'].unique()
    clusters = [df[df['Cluster'] == cluster_name] for cluster_name in cluster_names]
    total_si = 0
    for cluster in clusters:
        
        for i in range(len(cluster)):
            ai = calculate_distance_for_cluster(cluster.iloc[i], cluster)
            bi = None
            for cluster2 in clusters:
                if cluster.iloc[-1, -1] != cluster2.iloc[-1, -1]:
                    temp = calculate_distance_for_cluster(cluster.iloc[i], cluster2)
                    if bi == None or temp < bi:
                        bi = temp
            total_si += abs(1- (ai / bi))
    print("Avg silhouette coefficient for k of", k, "=",  round(total_si / len(df), 3))
