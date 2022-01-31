import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

names = ["crime", "zone", "industry", "charles", "no", "rooms", "age", "distance", "radial", "tax", "pupil", "aam", "lower", "med_price"]

data = pd.read_csv("boston_housing.csv", names = names)

X, y = data.iloc[:,:-1], data.iloc[:,-1]

xgb_pipeline = Pipeline[("st_scaler", StandardScaler()),
                        ("xgb_model", xgb.XGBRegressor())]

scores = cross_val_score(xgb_pipeline, X, y, scoring = "neg_mean_squared_error", cv = 10)

final_avg_rmse = np.mean(np.sqrt(np.abs(scores)))
print("Final XGB RMSE:", final_avg_rmse)


##                   Cross-validating XGBoost Model                  ##
# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


X, y = data.iloc[:,:-1], data.iloc[:,-1]

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse = False)),
         ("xgb_model", xgb.XGBRegressor(max_depth = 2, objective = "reg:linear"))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict('records'), y, cv = 10, scoring = "neg_mean_squared_error")

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))


##                   Kidney Disease Case Study I: Categorical Imputer                  ##

kidney_data = pd.read_csv('datasets/chronic_kidney_disease.csv',header=None,na_values='?')

kidney_feature_names = ['age','bp','sg','al','su','bgr','bu','sc','sod',
                        'pot','hemo','pcv','wc','rc','rbc','pc','pcc',
                        'ba','htn','dm','cad','appet','pe','ane']
kidney_target_name = ['class']
df.columns = kidney_feature_names + kidney_target_name
X, y = kidney_data.iloc[:,:-1], kidney_data.iloc[:,-1]


# Import necessary modules
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
                                            [([numeric_feature], Imputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                            input_df=True,
                                            df_out=True
                                           )

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
                                                [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
                                                input_df = True,
                                                df_out = True
                                               )


 ##                   Kidney Disease Case Study II: Feature Union                  ##
# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
                                          ("num_mapper", numeric_imputation_mapper),
                                          ("cat_mapper", categorical_imputation_mapper)
                                         ])


 ##                   Kidney Disease Case Study III: Full pipeline                  ##
# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort = False)),
                     ("clf", xgb.XGBClassifier(max_depth = 3))
                    ])

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, X, y, scoring = "roc_auc", cv = 3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))             