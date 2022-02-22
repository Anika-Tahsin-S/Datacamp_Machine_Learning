import pandas as pd

# The features in the pre-loaded dataset sales_df are: storeID, product, quantity and revenue.

sales_df = pd.read_csv('grocery_sales.csv')
sales_df.head()

# Transform the test set with the pre-fitted scaler
X_test_std = scaler.transform(X_test)

# Calculate the coefficient of determination (R squared) on X_test_std
r_squared = la.score(X_test_std, y_test)
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))

# Create a list that has True values when coefficients equal 0
zero_coef = la.coef_ == 0

# Calculate how many features have a zero coefficient
n_ignored = sum(zero_coef)
print("The model has ignored {} out of {} features.".format(n_ignored, len(la.coef_)))


##                   Manual Feature Extraction I                  ##
# Calculate the price from the quantity sold and revenue
sales_df['price'] = sales_df['revenue'] / sales_df['quantity']

# Drop the quantity and revenue features
reduced_df = sales_df.drop(['revenue', 'quantity'], axis = 1)

print(reduced_df.head())



##                   Manual Feature Extraction II                  ##
# Calculate the mean height
height_df['height'] = height_df[['height_1', 'height_2', 'height_3']].mean(axis = 1)

# Drop the 3 original height features
reduced_df = height_df.drop(['height_1', 'height_2', 'height_3'], axis = 1)

print(reduced_df.head())



##                   Principal Component Intuition                  ##
# After standardizing the lower and upper arm lengths from the ANSUR dataset we've added two perpendicular vectors that are aligned with the main directions of variance. 
# We can describe each point in the dataset as a combination of these two vectors multiplied with a value each. 
# These values are then called principal components.
# Which of the following statements is true?

# People with a negative component for the yellow vector have long forearms relative to their upper arms.