# Step 1:Generate cluster centers 
kmeans(obs, k_or_guess, iter, thresh, check_finite)

# Step 2:Generate cluster labels
vq(obs, code_book, check_finite = True)


##                   Running KMeans                  ##
from scipy.cluster.vq import kmeans, vq

cluster_centers, _ = kmeans(df[['scaled_x', 'scaled_y']], 3)
df['cluster_labels'], _ = vq(df[['scaled_x', 'scaled_y']], cluster_centers)


sns.scatterplot(x = 'scaled_x', y = 'scaled_y', hue = 'cluster_labels', data = df)
plt.show()


##                   K-means Clustering: First Exercise                  ##
# Import the kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

# Generate cluster centers
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], 2)

# Assign cluster labels
comic_con['cluster_labels'], distortion_list = vq(comic_con[['x_scaled', 'y_scaled']], cluster_centers)

# Plot clusters
sns.scatterplot(x = 'x_scaled', y = 'y_scaled', 
                hue = 'cluster_labels', data = comic_con)
plt.show()


##                   Runtime of K-means Clustering                  ##
%timeit kmeans(fifa[['scaled_sliding_tackle', 'scaled_aggression']], 3)
# It took only about 5 seconds to run hierarchical clustering on this data, but only 50 milliseconds to run k-means clustering.