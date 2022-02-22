# The first principal component of the data is the direction in which the data varies the most.
# Use PCA to find the first principal component of the length and width measurements of the grain samples, and represent it as an arrow on the scatter plot.
# The array grains gives the length and width of the grain samples. PyPlot (plt) and PCA have already been imported for you.

# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()