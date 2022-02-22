# Tuning XGBoost hyperparameters in a Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

names = ["crime","zone","industry","charles","no","rooms","age", 
         "distance","radial","tax","pupil","aam","lower","med_price"]
data = pd.read_csv("datasets/boston.csv",names=names,skiprows=1)
X, y = data.iloc[:,:-1],data.iloc[:,-1]

xgb_pipeline = Pipeline([("st_scaler",StandardScaler()), 
                        ("xgb_model",xgb.XGBRegressor())])

gbm_param_grid = {'xgb_model__subsample': np.arange(.05, 1, .05),
                  'xgb_model__max_depth': np.arange(3,20,1),
                  'xgb_model__colsample_bytree': np.arange(.1,1.05,.05) }

randomized_neg_mse = \
RandomizedSearchCV (estimator=xgb_pipeline,n_iter=10,
                    param_distributions=gbm_param_grid,
                    scoring='neg_mean_squared_error', cv=4)

randomized_neg_mse.fit(X, y)

# Tuning XGBoost hyperparameters in a Pipeline II
print("Best rmse: ", np.sqrt(np.abs(randomized_neg_mse.best_score_)))

print("Best model: ", randomized_neg_mse.best_estimator_)



 ##                   Bringing It All Together                  ##
# Create the parameter grid
gbm_param_grid = {
    'clf__learning_rate': np.arange(0.05, 1, 0.05),
    'clf__max_depth': np.arange(3, 10, 1),
    'clf__n_estimators': np.arange(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(estimator = pipeline, param_distributions = gbm_param_grid, n_iter = 2, scoring = "roc_auc", cv = 2, verbose = 1)

# Fit the estimator
randomized_roc_auc.fit(X, y)

# Compute metrics
print(randomized_roc_auc.best_score_)
print(randomized_roc_auc.best_estimator_)