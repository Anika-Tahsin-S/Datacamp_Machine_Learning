## Goal of EDA
# Size of the data
# Properties of the target variable
# Properties of the features
# Generate ideas for feature engineering



## EDA Part I
twosigma_train = pd.read_csv('twosigma_train.csv')
print('Train shape:', twosigma_train.shape)

twosigma_test = pd.read_csv('twosigma_test.csv')
print('Test shape:', twosigma_test.shape)

print(twosigma_train.columns.tolist())
twosigma_train.interest_level.value_counts()

twosigma_train.describe()


## EDA Part II
import matplotlib.pyplot as plt
plt.style.use('ggplot')

prices = twosigma_train.groupby('interest_level', as_index = False)['price'].median()

# Draw a barplot
fig = plt.figure(figsize = (7, 5))

# Plot the line plot
plt.bar(prices.interest_level, prices.price, width = 0.5, alpha = 0.8)
plt.xlabel('Interest Level')
plt.ylabel('Median Price')
plt.title('Median listing price across interest level')
plt.show()




## ====================================================================================================== ##