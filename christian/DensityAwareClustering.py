import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
class DensityAwareClustering:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Density-aware clustering method for merchant and buyer coordinates
        
        Parameters:
        - eps: Maximum distance between points in a cluster
        - min_samples: Minimum number of points to form a dense region
        """
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
    
    def find_natural_clusters(self, df):
        """
        Find clusters based on spatial density of both merchant and buyer coordinates
        
        Parameters:
        - df: DataFrame containing latitude and longitude for merchants and buyers
        
        Returns:
        - Clustered DataFrame with merchant and buyer cluster IDs
        - Cluster centroids
        """
        # Prepare coordinates for both merchants and buyers
        merchant_coords = df[['merch_lat', 'merch_long']].copy()
        merchant_coords['type'] = 'merchant'
        merchant_coords['coordinates'] = list(zip(merchant_coords['merch_lat'], merchant_coords['merch_long']))
        
        buyer_coords = df[['lat', 'long']].copy()
        buyer_coords['type'] = 'buyer'
        buyer_coords['coordinates'] = list(zip(buyer_coords['lat'], buyer_coords['long']))
        
        # Combine coordinates
        all_coords = pd.concat([
            merchant_coords[['coordinates', 'type']], 
            buyer_coords[['coordinates', 'type']]
        ])
        
        # Prepare coordinates for clustering
        X = np.array(all_coords['coordinates'].tolist())
        X_scaled = self.scaler.fit_transform(X)
        
        # Adaptive initial clustering
        initial_clusters = min(50, len(X) // self.min_samples)
        
        kmeans = MiniBatchKMeans(
            n_clusters=initial_clusters,
            random_state=42,
            batch_size=1024,
            max_iter=100,
            compute_labels=False  # Reduces memory usage
        )
        kmeans.fit(X_scaled)
        labels = kmeans.predict(X_scaled)
        
        # Memory-optimized distance calculation
        def compute_intra_cluster_distances(X, labels, cluster_id, batch_size=1000):
            cluster_mask = labels == cluster_id
            cluster_points = X[cluster_mask]
            if len(cluster_points) < 2:
                return 0
            
            total_distance = 0
            count = 0
            
            # Process in memory-friendly batches
            for i in range(0, len(cluster_points), batch_size):
                batch = cluster_points[i:i+batch_size]
                
                # Intra-batch distances
                if len(batch) > 1:
                    intra_dists = cdist(batch, batch, metric='euclidean')
                    mask = np.triu_indices(len(batch), k=1)
                    total_distance += intra_dists[mask].sum()
                    count += len(mask[0])
                
                # Inter-batch distances
                if i > 0:
                    inter_dists = cdist(batch, cluster_points[:i], metric='euclidean')
                    total_distance += inter_dists.sum()
                    count += inter_dists.size
                    
            return total_distance / count if count > 0 else 0

        # Cluster validation
        valid_clusters = []
        print('Filtering clusters based on density and size...')
        for cluster_id in range(initial_clusters):
            cluster_size = np.sum(labels == cluster_id)
            if cluster_size < self.min_samples:
                continue
                
            intra_dist = compute_intra_cluster_distances(X_scaled, labels, cluster_id)
            
            if intra_dist <= self.eps:
                valid_clusters.append(cluster_id)
        
        # Label remapping
        valid_mask = np.isin(labels, valid_clusters)
        new_labels = np.full(len(labels), -1, dtype=np.int16)
        for new_id, old_id in enumerate(valid_clusters):
            new_labels[labels == old_id] = new_id
        
        # Add cluster labels to original dataframe
        df_clustered = df.copy()
        
        # Separate merchant and buyer labels
        merchant_mask = all_coords['type'] == 'merchant'
        buyer_mask = all_coords['type'] == 'buyer'
        
        print("adding distances")
        df_clustered['merchant_buyer_distance'] = df_clustered.apply(
            lambda row: euclidean(
                (row['merch_lat'], row['merch_long']),
                (row['lat'], row['long'])
            ),
            axis=1
        )



        df_clustered['merchant_cluster_id'] = new_labels[merchant_mask]
        df_clustered['buyer_cluster_id'] = new_labels[buyer_mask]
        
        # Calculate centroids for valid clusters
        valid_centroids = self.scaler.inverse_transform(
            kmeans.cluster_centers_[valid_clusters]
        )
        
        return df_clustered, valid_centroids

    def visualize_clusters(self, df_clustered, centroids):
        """
        Visualize clustering results for merchants and buyers
        
        Parameters:
        - df_clustered: DataFrame with cluster assignments
        - centroids: Cluster centroids
        """
        plt.figure(figsize=(15, 8))
        
        # Merchants
        merchant_clusters = df_clustered[df_clustered['merchant_cluster_id'] != -1]
        plt.scatter(
            merchant_clusters['merch_long'], 
            merchant_clusters['merch_lat'],
            c=merchant_clusters['merchant_cluster_id'], 
            cmap='viridis', 
            alpha=0.7,
            label='Merchants',
            marker='s'
        )
        
        # Buyers
        buyer_clusters = df_clustered[df_clustered['buyer_cluster_id'] != -1]
        plt.scatter(
            buyer_clusters['long'], 
            buyer_clusters['lat'],
            c=buyer_clusters['buyer_cluster_id'], 
            cmap='plasma', 
            alpha=0.7,
            label='Buyers',
            marker='o'
        )
        
        # Plot cluster centroids
        plt.scatter(
            centroids[:, 1], 
            centroids[:, 0], 
            color='red', 
            marker='x', 
            s=200, 
            linewidths=3,
            label='Centroids'
        )
        
        plt.colorbar(label='Cluster ID')
        plt.title('Density-Aware Geolocation Clusters')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def analyze_clusters(self, df_clustered):
        """
        Provide detailed cluster analysis for merchants and buyers
        
        Parameters:
        - df_clustered: DataFrame with cluster assignments
        
        Returns:
        - Merchant cluster statistics
        - Buyer cluster statistics
        """
        # Merchant cluster analysis
        merchant_cluster_summary = df_clustered[df_clustered['merchant_cluster_id'] != -1].groupby('merchant_cluster_id').agg({
            'merch_lat': ['count', 'mean'],
            'merch_long': ['mean']
        })
        
        # Buyer cluster analysis
        buyer_cluster_summary = df_clustered[df_clustered['buyer_cluster_id'] != -1].groupby('buyer_cluster_id').agg({
            'lat': ['count', 'mean'],
            'long': ['mean']
        })
        
        return merchant_cluster_summary, buyer_cluster_summary