#assgn 4
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

# Given dataset 
data = np.array([ [1.713, 1.586], [0.180, 1.786], [0.353, 1.240], [0.940, 1.566], [1.486, 0.759], [1.266, 1.106], [1.540, 0.419], [0.459, 1.799], [0.773, 0.186] ])  
labels = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1]) 

# Applying k-means clustering with 3 clusters 
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) 
kmeans.fit(data) 

# Predict cluster for new point
new_point = np.array([[0.906, 0.606]])
predicted_cluster = kmeans.predict(new_point) 

# Assign class based on the majority class in the predicted cluster 
cluster_labels = kmeans.labels_ 
cluster_classes = {i: np.bincount(labels[cluster_labels == i]).argmax() for i in range(3)} 
predicted_class = cluster_classes[predicted_cluster[0]] 

# Visualizing the clusters 
plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', marker='o', label='Data Points') 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids') 
plt.scatter(new_point[:, 0], new_point[:, 1], c='blue', marker='s', s=100, label='New Point') 
plt.xlabel('VAR1' ) 
plt.ylabel('VAR2' ) 
plt.legend() 
plt.title('K-Means Clustering with New Data Point') 
plt.show() 
print(f"Predicted cluster: {predicted_cluster[0]}") 
print(f"Predicted class for (0.906, 0.606): {predicted_class}")