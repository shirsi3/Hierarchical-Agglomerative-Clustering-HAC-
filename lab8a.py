import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


data = pd.read_csv("lab8official.csv")

def perform_kmeans(data, K_values):
    results = []

    for k in K_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0) 
        clusters = kmeans.fit_predict(data)

        # select 10-20 samples
        samples_per_cluster = 10
        selected_samples = []
        for cluster_idx in range(k):
            cluster_samples = np.where(clusters == cluster_idx)[0]
            selected_samples.extend(np.random.choice(cluster_samples, samples_per_cluster, replace=False))

        
        selected_data = data.iloc[selected_samples]

        # top 5 features for each cluster centroids
        centroids = kmeans.cluster_centers_
        top_features = []
        for cluster_idx in range(k):
            centroid = centroids[cluster_idx]
            top_feature_indices = np.argsort(centroid)[::-1][:5]
            top_features.append(top_feature_indices)

        results.append((k, clusters, top_features, selected_data))

    return results

def present_results(results):
    for k, clusters, top_features, selected_data in results:
        print(f"K={k}")
        print("Top 5 Features in Each Cluster:")
        for cluster_idx, feature_indices in enumerate(top_features):
            print(f"Cluster {cluster_idx + 1}:")
            for idx in feature_indices:
                print(f"Feature {idx}: {data.columns[idx]}")
            print()

        print("Selected Samples from Each Cluster:")
        print(selected_data)
        print("=" * 40)

def plot_clusters_and_features(results, data):
    for k, clusters, top_features, selected_data in results:
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(data)
        
        plt.figure(figsize=(12, 6))

        
        plt.subplot(1, 2, 1)
        for cluster_idx in range(k):
            cluster_data = reduced_features[clusters == cluster_idx]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_idx + 1}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.title(f'Clusters (K={k})')

        
        plt.subplot(1, 2, 2)
        for cluster_idx, feature_indices in enumerate(top_features):
            feature_names = [data.columns[idx] for idx in feature_indices]
            feature_means = selected_data.iloc[:, feature_indices].mean()
            plt.barh(feature_names, feature_means, label=f'Cluster {cluster_idx + 1}')
        plt.xlabel('Mean Feature Value')
        plt.title(f'Top 5 Features (K={k})')

        plt.tight_layout()  
        plt.show()

if __name__ == "__main__":
    K_values = [2, 5, 10]
    clustering_results = perform_kmeans(data, K_values)
    present_results(clustering_results)
    plot_clusters_and_features(clustering_results, data)
