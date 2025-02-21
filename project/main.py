X_b = df_b[keep]  # Make sure to use the same columns that you used for scaling

X_b = X_b.apply(lambda col: col.fillna(col.mean()), axis=0)

# Step 1: Standardize the features before PCA (you can also use your scaler if needed)
X_b_scaled = scaler_b.transform(X_b)

# Step 2: Apply PCA to reduce the features to 2D
pca = PCA(n_components=2)
X_b_pca = pca.fit_transform(X_b_scaled)

# Step 3: Add PCA components to the DataFrame for plotting
df_b_pca = pd.DataFrame(X_b_pca, columns=['PCA1', 'PCA2'])
df_b_pca['cluster'] = df_b['cluster']  # Assign the clusters to the reduced components

# Step 4: Create the scatter plot
plt.figure(figsize=(10, 6))

# Scatter plot, using different colors for each cluster
sns.scatterplot(data=df_b_pca, x='PCA1', y='PCA2', hue='cluster', palette='Set1', style='cluster', s=100)

# Step 5: Customize the plot
plt.title('2D Visualization of Clusters after PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()





# df_b_nmf = pd.DataFrame(X_b_nmf, columns=[f'NMF_{i+1}' for i in range(X_b_nmf.shape[1])])
# df_b_nmf['cluster'] = df_b['cluster']

# # Visualizing the first two NMF components
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df_b_nmf, x='NMF_1', y='NMF_2', hue='cluster', palette='Set1', style='cluster', s=100)
# plt.title('2D Visualization of Clusters after NMF')
# plt.xlabel('NMF Component 1')
# plt.ylabel('NMF Component 2')
# plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score
# import numpy as np
# from sklearn.model_selection import ParameterGrid

# # Define your dataset
# # Assuming X is the feature set (after scaling, if necessary)
# X1 = df_b[keep]  # Feature columns for clustering (replace 'keep' with actual column names)

# knn_imputer = KNNImputer(n_neighbors=3)

# X = knn_imputer.fit_transform(X1)

# # Scale the data if necessary (important for DBSCAN)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Define the range of hyperparameters to search over
# param_grid = {
#     'eps': np.arange(0.1, 2.0, 0.1),        # eps values from 0.1 to 2.0 with step size of 0.1
#     'min_samples': range(3, 15)             # min_samples from 3 to 15
# }

# # Initialize GridSearch using a custom scoring function (Silhouette Score)
# best_score = -1  # Store the best silhouette score
# best_params = None  # Store the best parameters
# best_model = None  # Store the best model

# # Iterate over all combinations of eps and min_samples
# for params in ParameterGrid(param_grid):
#     eps = params['eps']
#     min_samples = params['min_samples']
    
#     # Fit DBSCAN with current parameters
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     dbscan.fit(X_scaled)

#     # Check the number of clusters and ignore if the model fails to find meaningful clusters
#     if len(set(dbscan.labels_)) > 1:  # Ensure there is more than one cluster
#         score = silhouette_score(X_scaled, dbscan.labels_)
        
#         # Keep track of the best score and parameters
#         if score > best_score:
#             best_score = score
#             best_params = params
#             best_model = dbscan

# print(f"Best Parameters: {best_params}")
# print(f"Best Silhouette Score: {best_score}")


# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler

# X1 = df_b[keep]  # Features for all rows
# Y = df_b['cluster']  # Cluster labels for rows with assigned clusters

# knn_imputer = KNNImputer(n_neighbors=3)

# X = knn_imputer.fit_transform(X1)

# # Separate clustered and unclustered data
# X_clustered = X[df_b['cluster'].notna()]
# X_unclustered = X[df_b['cluster'].isna()]

# # Scale the features
# scaler = StandardScaler()
# X_clustered_scaled = scaler.fit_transform(X_clustered)
# X_unclustered_scaled = scaler.transform(X_unclustered)

# # Train DBSCAN on the clustered data
# dbscan = DBSCAN(eps=1.7, min_samples=4)  # Adjust parameters as needed
# dbscan.fit(X_clustered_scaled)

# # Use DBSCAN to predict clusters for unclustered data
# predicted_labels = dbscan.fit_predict(X_unclustered_scaled)

# # Assign predicted clusters to the unclustered data in df_b
# df_b.loc[df_b['cluster'].isna(), 'cluster'] = predicted_labels

# # Now df will have clusters assigned to the unclustered data as we


# from sklearn.semi_supervised import LabelPropagation

# X1 = df_b[keep]  # Features for all rows
# Y1 = df_b['cluster']  # Cluster labels for rows with assigned clusters

# knn_imputer = KNNImputer(n_neighbors=3)

# X = knn_imputer.fit_transform(X1)
# # Y = knn_imputer.fit_transform(Y1)

# # Initialize LabelPropagation model
# label_prop = LabelPropagation(kernel='knn', n_neighbors=5)

# # Fit the model on the features and clusters (with -1 for unlabeled data)
# label_prop.fit(X, Y1)

# # Propagate labels to the unclustered data
# df_b['cluster'] = label_prop.labels_







# # Apply NMF for dimensionality reduction
# n_components = 5  # Number of clusters or topics you want to extract
# nmf_model = NMF(n_components=n_components, random_state=42)
# X_b_nmf = nmf_model.fit_transform(X_b_scaled)

# # After applying NMF, X_b_nmf contains the matrix with reduced dimensions (latent features)
# # You can use this matrix to assign clusters based on similarity (e.g., cosine similarity) or directly.

# # Step 1: Find the closest clusters for unclustered records
# X_b_additional = df_b[df_b['cluster'].isna()][keep]  # Features for unclustered records
# X_b_clustered = df_b[df_b['cluster'].notna()][keep]  # Features for clustered records

# # Fill missing values
# X_b_additional = X_b_additional.apply(lambda col: col.fillna(col.mean()), axis=0)
# X_b_clustered = X_b_clustered.apply(lambda col: col.fillna(col.mean()), axis=0)

# # Standardize the features
# X_b_additional_scaled = scaler_b.transform(X_b_additional)
# X_b_clustered_scaled = scaler_b.transform(X_b_clustered)

# X_b_additional_scaled = np.abs(X_b_additional_scaled)
# X_b_clustered_scaled = np.abs(X_b_clustered_scaled)

# # Apply NMF transformation on unclustered and clustered records
# X_b_additional_nmf = nmf_model.transform(X_b_additional_scaled)
# X_b_clustered_nmf = nmf_model.transform(X_b_clustered_scaled)

# # Calculate the cosine similarity between unclustered and clustered records in NMF space
# similarity_matrix = cosine_similarity(X_b_additional_nmf, X_b_clustered_nmf)

# # Find the most similar clustered record for each unclustered record
# closest_matches = np.argmax(similarity_matrix, axis=1)

# # Get the cluster labels from the most similar clustered records
# closest_clusters = df_b.loc[df_b['cluster'].notna(), 'cluster'].iloc[closest_matches].values

# # Assign the closest cluster to the unclustered records in df_b
# df_b.loc[df_b['cluster'].isna(), 'cluster'] = closest_clusters

# # Print the updated df_b to check if all unclustered records got assigned a cluster
# print(df_b.head())



# X_b_additional = X_b_additional.apply(lambda col: col.fillna(col.mean()), axis=0)
# X_b_clustered = X_b_clustered.apply(lambda col: col.fillna(col.mean()), axis=0)

# # Standardize the features of unclustered and clustered records
# X_b_additional_scaled = scaler_b.transform(X_b_additional)
# X_b_clustered_scaled = scaler_b.transform(X_b_clustered)

# # Calculate the cosine similarity between unclustered records and clustered records
# similarity_matrix = cosine_similarity(X_b_additional_scaled, X_b_clustered_scaled)

# # Find the most similar clustered record for each unclustered record
# closest_matches = np.argmax(similarity_matrix, axis=1)

# # Ensure that closest_matches has the same length as the number of unclustered records
# print("Shape of closest matches:", closest_matches.shape)  # Check the shape
# print("Shape of unclustered records:", X_b_additional.shape)

# # Get the cluster labels of the most similar clustered records
# closest_clusters = df_b.loc[df_b['cluster'].notna(), 'cluster'].iloc[closest_matches].values

# # Assign the closest cluster to the unclustered records in df_b
# df_b.loc[df_b['cluster'].isna(), 'cluster'] = closest_clusters

# # Print the updated df_b to check if all unclustered records got assigned a cluster
# print(df_b)


# df_b.head()




# X_b_additional = df_b[df_b['cluster'].isna()][keep]  # Features for unclustered records
# X_b_clustered = df_b[df_b['cluster'].notna()][keep]  # Features for clustered records

# print(X_b_additional.shape)
# print(X_b_clustered.shape)

# X_b_additional = X_b_additional.apply(lambda col: col.fillna(col.mean()), axis=0)
# X_b_clustered = X_b_clustered.apply(lambda col: col.fillna(col.mean()), axis=0)

# # Standardize the features of unclustered and clustered records in df_b
# X_b_additional_scaled = scaler_b.transform(X_b_additional)
# X_b_clustered_scaled = scaler_b.transform(X_b_clustered)

# # Calculate the cosine similarity between unclustered records and clustered records
# similarity_matrix = cosine_similarity(X_b_additional_scaled, X_b_clustered_scaled)

# # print(similarity_matrix)

# # For each unclustered record, find the most similar clustered record
# closest_matches = np.argmax(similarity_matrix, axis=1)
# # print(closest_matches)

# # print(df_b)

# # Get the cluster labels from df_b for the closest matched clustered records
# # df_b.loc[df_b['cluster'].isna(), 'cluster'] = df_b['cluster'].iloc[closest_matches].values
# # Assuming closest_matches is a list of indices of rows to update
# df_b.loc[df_b['cluster'].isna(), 'cluster'] = df_b.loc[closest_matches, 'cluster'].values

# i = 0
# while i< df_b.shape[0] - 1:
#     print(df_b.iloc[i])
#     print(closest_matches[i])
    # print(df_b.iloc[closest_matches[i]])
    # i += 1



# for idx in closest_matches:
#     df.at
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

# Example of creating the dataframe and scaler, assuming df_b and scaler_b are already defined

# Extract the unclustered and clustered records
X_b_additional = df_b[df_b['cluster'].isna()][keep]  # Features for unclustered records
X_b_clustered = df_b[df_b['cluster'].notna()][keep]  # Features for clustered records

print(X_b_clustered.shape)

test = df_b[df_b['cluster'].notna()]
print(test)






# Fill missing values with column mean for both sets

X_b = df_b[keep]  # Make sure to use the same columns that you used for scaling

X_b = X_b.apply(lambda col: col.fillna(col.mean()), axis=0)

# Step 1: Standardize the features before PCA (you can also use your scaler if needed)
X_b_scaled = scaler_b.transform(X_b)

# Step 2: Apply PCA to reduce the features to 2D
pca = PCA(n_components=2)
X_b_pca = pca.fit_transform(X_b_scaled)

# Step 3: Add PCA components to the DataFrame for plotting
df_b_pca = pd.DataFrame(X_b_pca, columns=['PCA1', 'PCA2'])
df_b_pca['cluster'] = df_b['cluster']  # Assign the clusters to the reduced components

# Step 4: Create the scatter plot
plt.figure(figsize=(10, 6))

# Scatter plot, using different colors for each cluster
sns.scatterplot(data=df_b_pca, x='PCA1', y='PCA2', hue='cluster', palette='Set1', style='cluster', s=100)

# Step 5: Customize the plot
plt.title('2D Visualization of Clusters after PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
