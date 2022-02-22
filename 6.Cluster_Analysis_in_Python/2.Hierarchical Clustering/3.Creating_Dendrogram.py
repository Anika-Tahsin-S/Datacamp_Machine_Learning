from scipy.cluster.hierarchy import dendrogram

Z = linkage(df[['x_whiten', 'y_whiten']],
                method = 'ward',
                metric = 'euclidean')
dn = dendrogram(Z)
plt.show()


##                   Exercise                  ##
# The data is stored in a pandas DataFrame, comic_con. 
comic_con = pd.read_csv("datasets/comic_con.csv")
comic_con.head()

# x_scaled and y_scaled are the column names of the standardized X and Y coordinates of people at a given point in time. 
# cluster_labels has the cluster labels. 
# A linkage object is stored in the variable distance_matrix.

# Import the fcluster and linkage functions
from scipy.cluster.hierarchy import linkage, fcluster
# Use the linkage() function
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], 
                            method = "ward", metric = 'euclidean')


# Import the dendrogram function
from scipy.cluster.hierarchy import dendrogram

# Create a dendrogram
dn = dendrogram(distance_matrix)

# Display the dendogram
plt.show()

# The top two clusters are farthest away from each other.