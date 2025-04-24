# Imports
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import Ridge

""" code starts here """

df = pd.read_csv("./assets/train.csv")

# Dealing with missing values: Removal
df = df.drop(columns = ["Average Temperature During Storage (celcius)"])
df = df.dropna()

# Decoding for dataset
df = pd.get_dummies(df, columns = ["color"], dtype = int)

mapping = {"Morning": 1, "Midday": 2, "Evening": 3}
df["Harvest Time"] = df["Harvest Time"].map(mapping)

# Dealing with outliers
numeric_cols = df.select_dtypes(include = "number")

# Calculate IQR
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df_filtered = df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis = 1)]

# Splitting Data
X = df_filtered.drop(columns = ["Scoville Heat Units (SHU)"])
y = df_filtered["Scoville Heat Units (SHU)"]

num_components = 0.85

# Pipeline A
pipeA = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    ("scalar", StandardScaler()),
    ("pca", PCA(n_components = num_components)),
    ("regressor", Ridge())
])

param_grid_A = {
    "regressor__alpha": np.arange(0, 20, 2)
}

gs_A = GridSearchCV(estimator = pipeA, 
                    param_grid = param_grid_A, 
                    scoring = "neg_mean_absolute_error", 
                    cv = KFold(n_splits = 10, shuffle = True, random_state = 42), 
                    n_jobs = -1)
    
gs_A.fit(X, y)

print("Best params:", gs_A.best_params_)
print("Best MAE:", gs_A.best_score_)

model = gs_A.best_estimator_

cv_scores = cross_val_score(model, X, y, cv = 10, scoring = "neg_mean_absolute_error")
print("Mean MAE:", -cv_scores.mean())