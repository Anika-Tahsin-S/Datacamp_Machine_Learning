Creating a distance matrix using linkage

scipy.cluster.hierarchy.linkage(observations,
                                method = 'single',
                                metric = 'euclidean',
                                optimal_ordering = False
)

# method: how to calculate the proximity of clusters
# metric: distance metric
# optimal_ordering: order data points


# Create cluster labels with fcluster

scipy.cluster.hierarchy.fcluster(distance_matrix,
                               num_clusters,
                               criterion
)

# distance_matrix: output of linkage() method
# num_clusters: number of clusters
# criterion: how to decide thresholds to form clusters



##                   Hierarchical Clustering: Ward Method                  ##
comic_con = pd.read_csv("datasets/comic_con.csv")
comic_con.head()

# Import the fcluster and linkage functions
from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() function
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method = "ward", metric = 'euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion = 'maxclust')

# Plot clusters
sns.scatterplot(x = 'x_scaled', y = 'y_scaled', 
                hue = 'cluster_labels', data = comic_con)
plt.show()
# The two clusters correspond to the points of attractions in the figure towards the bottom (a stage) and the top right (an interesting stall).




##                   Hierarchical Clustering: Single Method                  ##
# Import the fcluster and linkage functions
from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() function
distance_matrix = linkage(comic_con[["x_scaled", "y_scaled"]], method = 'single', metric = 'euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion = "maxclust")

# Plot clusters
sns.scatterplot(x = 'x_scaled', y = 'y_scaled', 
                hue = 'cluster_labels', data = comic_con)
plt.show()
# The clusters formed are not different from the ones created using the ward method.



##                   Hierarchical Clustering: Complete Method                  ##
from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() function
distance_matrix = linkage(comic_con[["x_scaled", "y_scaled"]], method = 'complete', metric = 'euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion = "maxclust")

# Plot clusters
sns.scatterplot(x = 'x_scaled', y = 'y_scaled', 
                hue = 'cluster_labels', data = comic_con)
plt.show()
# Coincidentally, the clusters formed are not different from the ward or single methods. Next, let us learn how to visualize clusters.