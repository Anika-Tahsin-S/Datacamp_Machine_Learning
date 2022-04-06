## K-fold cross-validation
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle = True, random_state = 123)

for train_index, test_index in kf.split(train):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

## Stratified K-fold
from sklearn.model_selection import StratifiedKFold

str_kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 123)

for train_index, test_index in str_kf.split(train, train['target']):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]




## ====================================================================================================== ##