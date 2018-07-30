# --------------------------------------   IMPORT   -------------------------------------- #
# 1. Imports Libraries and Modules
import pandas as pd
import numpy as np

from sklearn.model_selection import test_train_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib


# ------------------------------------   LOAD DATA   -------------------------------------- #
# 2. Load Dataset
dataset_url = '..... Enter link to the dataset......'
data = pd.read_csv(dataset_url, sep = ';')


# ----------------------------------   DATA PROCESSING   ---------------------------------- #
''' Follow the Data Clean process and identify Target data '''

# 3. Copy your Target Variable in a seperate variable
y = data.target      # replace 'target' with your target column name 

# 4. Drop Target variable from the main data 
X = data.drop('target', axis=1)  # replace 'target' with your target column name

# 5. Split data in Test and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# 6. Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))


# -------------------------------   MACHINE LEARNING MODELS   ------------------------------- #
# 7. Declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5, 3, 1] }
 
# 8. Tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)

# 9. Check best parameter for hyper parameter tunning
print clf.best_params_
# {'randomforestregressor__max_depth': None, 'randomforestregressor__max_features': 'auto'}
 
# 10. Refit on the entire training set
# No additional code needed if clf.refit == True (default is True)
 
# 11. Evaluate model pipeline on test data
pred = clf.predict(X_test)
print r2_score(y_test, pred)
print mean_squared_error(y_test, pred)
 
# -----------------------------   REUSE MODEL FOR FUTURE USE    ----------------------------- #  
# 12. Save model for future use
joblib.dump(clf, 'rf_regressor.pkl')

# 13. Load model for .pkl file
clf2 = joblib.load('rf_regressor.pkl')
 
# 14. Predict data set using loaded model
clf2.predict(X_test)

