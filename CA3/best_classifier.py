from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

""" Code """
df = pd.read_csv("./assets/train.csv", index_col = 0)

# Peprocessing Part
useless_features = ["Banana Density", "Peel Thickness"]
df_clean = df.copy()
X = df_clean.drop(columns = ["Quality"] + useless_features)
y = df_clean["Quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Using transform to prevent data leakage

# Modeling
size = 1360
X_train_subset, y_train_subset = X_train_scaled[:size], y_train[:size]

best_model = SVC(C = 11, gamma = "scale", kernel = "rbf")
best_model.fit(X_train_subset, y_train_subset)