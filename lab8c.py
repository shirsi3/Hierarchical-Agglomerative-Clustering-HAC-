import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('lab8official.csv')
top_features = ['dsport', 'sport', 'dbytes', 'dtcpb', 'stcpb']

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[top_features])
data_scaled = pd.DataFrame(data_scaled, columns=top_features)


sample_size = 1000  
np.random.seed(0) 


for combo in combinations(top_features, 2):
    plt.figure(figsize=(8, 6))
    indices = np.random.choice(data_scaled.shape[0], sample_size, replace=False)
    plt.scatter(data_scaled.iloc[indices][combo[0]], data_scaled.iloc[indices][combo[1]], alpha=0.5, s=10, color='pink')
    plt.xlabel(combo[0])
    plt.ylabel(combo[1])
    plt.title(f'Scatter Plot of {combo[0]} vs {combo[1]}')
    plt.grid(True)
    plt.show()


cluster_numbers = [2, 5, 10]

for k in cluster_numbers:
    clustering = AgglomerativeClustering(n_clusters=k)
    cluster_labels = clustering.fit_predict(data_scaled)
    plt.figure(figsize=(10, 7))
    plt.title(f'Dendrogram for Agglomerative Clustering with k={k}')
    Z = linkage(data_scaled, method='ward')
    dendrogram(Z)
    plt.show()
