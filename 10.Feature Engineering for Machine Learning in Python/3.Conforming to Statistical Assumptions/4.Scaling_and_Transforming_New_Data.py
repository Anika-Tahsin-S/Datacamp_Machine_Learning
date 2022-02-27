# Reuse training scalers
scaler = StandaedScaler()

scaler.fit(train[['col']])

train['scaled_col'] = scaler.tranform(train[['col']])

# Fit some model
# ......

test = pd.read_csv('test_csv')

test['scaled_col'] = scaler.transform(test[['col']])



# Training transformations for reuse
train_std = train['col'].std()
train_mean = train['col'].mean()

# Calculate the cutoff
cut_off = train_std * 3
train_lower, train_upper = train_mean - cut_off, train_mean + cut_off

# Subset train data
test = pd.read_csv('test_csv')

# Subset test data
test = test[(test['col'] < train_upper) & (test['col'] > train_lower)]





# -------------------------------------------------------------------------- #
##                   Train and testing transformations (I)                  ##
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Apply a standard scaler to the data
SS_scaler = StandardScaler()

# Fit the standard scaler to the data
SS_scaler.fit(so_train_numeric[['Age']])

# Transform the test data using the fitted scaler
so_test_numeric['Age_ss'] = SS_scaler.transform(so_test_numeric[['Age']])
print(so_test_numeric[['Age', 'Age_ss']].head())
# output:
#          Age  Age_ss
#     700   35  -0.069
#     701   18  -1.343
#     702   47   0.830
#     703   57   1.579
#     704   41   0.380






##                   Train and testing transformations (II)                  ##
train_std = so_train_numeric['ConvertedSalary'].std()
train_mean = so_train_numeric['ConvertedSalary'].mean()

cut_off = train_std * 3
train_lower, train_upper = train_mean - cut_off, train_mean + cut_off

# Trim the test DataFrame
trimmed_df = so_test_numeric[(so_test_numeric['ConvertedSalary'] < train_upper) \
                             & (so_test_numeric['ConvertedSalary'] > train_lower)]