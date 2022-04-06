## Creating Features
# Concatenate the train and test data
data = pd.concat([[train, test]])

# Create new features for the data DataFrame....

# Get the train and test back
train = data[data.id.isin(train.id)]
test = data[data.id.isin(test.id)]



## Arithmetical Features
two_sigma.head(1)

two_sigma['price_per_bathroom'] = two_sigma.price / two_sigma.bedrooms
two_sigma['rooms_number'] = two_sigma.bedrooms + two_sigma.bathrooms


## Datetime Features
dem.head(1)

dem['date'] = pd.to_datetime(dem['date'])

dem['year'] = dem['date'].dt.year
dem['month'] = dem['date'].dt.month
dem['week'] = dem['date'].dt.weekofyear

# Day features
dem['dayofyear'] = dem['date'].dt.dayofyear
dem['dayofmonth'] = dem['date'].dt.day
dem['dayofweek'] = dem['date'].dt.dayofweek





## ====================================================================================================== ##