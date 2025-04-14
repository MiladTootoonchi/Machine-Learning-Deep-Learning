import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report

# Reading data
df = pd.read_csv("./assets/train.csv", index_col = 0)

# Cleaning
df = df.dropna()
X = df.drop(columns = ["class"])
y = df["class"]

# Preprocessing
svc = SVC(kernel="rbf")
SVM = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components = 0.995)),     # To keep the 99.5% variance
    ("svc", svc)
])

# Modelling
param_grid = {
    'svc__C': [1, 10, 100],
    'svc__gamma': ['scale', 0.01, 0.001],
}

gs_SVC = GridSearchCV(estimator = SVM, 
                    param_grid = param_grid, 
                    scoring = 'f1_macro', 
                    cv = StratifiedKFold(n_splits = 2), 
                    n_jobs = -1)

gs_SVC.fit(X, y)
print("CV: DONE")

y_pred = gs_SVC.predict(X)
score = f1_score(y, y_pred, average = 'macro')
print("F1 Macro:", score)


# Final evaluation
model = gs_SVC.best_estimator_
print(gs_SVC.best_params_)

cv_scores = cross_val_score(model, X, y, cv = 5, scoring = 'f1_macro')
print("Mean CV F1-macro:", np.mean(cv_scores))

X_train60, X_test40, y_train60, y_test40 = train_test_split(X, y, test_size = 0.4, random_state = 42)
y_pred = model.predict(X_test40)

cm = confusion_matrix(y_test40, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()

print(f"F1-macro: {f1_score(y_test40, y_pred, average = 'macro')} \n")
print(classification_report(y_test40, y_pred))