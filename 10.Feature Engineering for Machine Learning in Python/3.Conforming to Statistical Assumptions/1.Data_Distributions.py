##                   What does your data look like? (I)                  ##
# Part 1
# Create a histogram
so_numeric_df.hist()
plt.show()

# Part 2
# Create a boxplot of two columns
so_numeric_df[['Age', 'Years Experience']].boxplot()
plt.show()

# Part 3
# Create a boxplot of ConvertedSalary
so_numeric_df[['ConvertedSalary']].boxplot()
plt.show()




##                   What does your data look like? (II)                  ##
# Part 1
# Import packages
import matplotlib.pyplot as plt
import seaborn as sns

# Plot pairwise relationships
sns.pairplot(so_numeric_df)

# Show plot
plt.show()

# Part 2
# Print summary statistics
print(so_numeric_df.describe())
# output:
#            ConvertedSalary      Age  Years Experience
#     count        9.990e+02  999.000           999.000
#     mean         6.162e+04   36.003             9.962
#     std          1.761e+05   13.255             4.878
#     min          0.000e+00   18.000             0.000
#     25%          0.000e+00   25.000             7.000
#     50%          2.712e+04   35.000            10.000
#     75%          7.000e+04   45.000            13.000
#     max          2.000e+06   83.000            27.000




##                   When don't you have to transform your data?                  ##
# While making sure that all of your data is on the same scale is advisable for most analyses, for which of the following machine learning models is normalizing data not always necessary?
# Answer: Decision Trees
# As decision trees split along a singular point, they do not require all the columns to be on the same scale.