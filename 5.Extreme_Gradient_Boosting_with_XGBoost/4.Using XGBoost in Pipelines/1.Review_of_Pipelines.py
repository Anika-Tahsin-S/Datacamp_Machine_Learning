import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

names = ["crime", "zone", "industry", "charles", "no", "rooms", "age", "distance", "radial", "tax", "pupil", "aam", "lower", "med_price"]

data = pd.read_csv("boston_housing.csv", names = names)

X, y = data.iloc[:,:-1], data.iloc[:,-1]

rf_pipeline = Pipeline[("st_scaler", StandardScaler()),
                        ("rf_model", RandomForestRegressor())]

scores = cross_val_score(rf_pipeline, X, y, scoring = "neg_mean_squared_error", cv = 10)

final_avg_rmse = np.mean(np.sqrt(np.abs(scores)))
print("Final RMSE:", final_avg_rmse)