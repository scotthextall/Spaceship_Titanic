import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set pandas options
pd.set_option("display.max_columns", 50)

# Step 1: Load datasets and train/test split

# Load train and test datasets
X_full = pd.read_csv("Data/train.csv", index_col="PassengerId")
X_test_full = pd.read_csv("Data/test.csv", index_col="PassengerId")

# Remove rows with missing target data and separate target from features
X_full.dropna(axis=0, subset=["Transported"], inplace=True)
y = X_full["Transported"]
X_full.drop(["Transported"], axis=1, inplace=True)

# Train/test split
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)


# Step 2a: Data Preprocessing - handling missing values

# Train data exploration
print(X_train.shape)
print(X_train.head())
print(X_train.info())
print(X_train.describe())

# Check columns for missing values
missing_values = X_train.isnull().sum()
print(missing_values)
print(X_train.isnull().mean()*100)

# Identify categorical vs. numerical columns
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
numerical_cols = [col for col in X_train.columns if X_train[col].dtype != "object"]

# Create imputers for categorical and numerical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')

# Impute values in categorical and numerical columns
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_valid[categorical_cols] = cat_imputer.transform(X_valid[categorical_cols])
X_test_full[categorical_cols] = cat_imputer.transform(X_test_full[categorical_cols])
X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_valid[numerical_cols] = num_imputer.transform(X_valid[numerical_cols])
X_test_full[numerical_cols] = num_imputer.transform(X_test_full[numerical_cols])

# Check imputer statistics
print(cat_imputer.statistics_)
print(num_imputer.statistics_)

# Verify no more missing values
print(X_train.isnull().sum())
print(X_valid.isnull().sum())
print(X_test_full.isnull().sum())


# Step 2b: Data Preprocessing - encoding categorical variables

# Assess cardinality of categorical columns
cardinality = X_train[categorical_cols].nunique()
low_cardinality_cols = [col for col in X_train[categorical_cols] if X_train[col].nunique() < 10]
high_cardinality_cols = list(set(categorical_cols) - set(low_cardinality_cols))
print(cardinality)

# Drop Name and Cabin columns due to high cardinality - Cabin column may be potentially relevant here however
X_train.drop(columns=high_cardinality_cols, inplace=True)
X_valid.drop(columns=high_cardinality_cols, inplace=True)
X_test_full.drop(columns=high_cardinality_cols, inplace=True)

# One-hot encode remaining low-cardinality categorical columns
OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test_full[low_cardinality_cols]))

# One-hot encoding removed index, need to replace
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
OH_cols_test.index = X_test_full.index

# Drop categorical columns from original data
num_X_train = X_train.drop(low_cardinality_cols, axis=1)
num_X_valid = X_valid.drop(low_cardinality_cols, axis=1)
num_X_test = X_test_full.drop(low_cardinality_cols, axis=1)

# Concat new categorical columns with numerical columns
X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
X_test_full = pd.concat([num_X_test, OH_cols_test], axis=1)

# Ensure all columns have string type
X_train.columns = X_train.columns.astype(str)
X_valid.columns = X_valid.columns.astype(str)
X_test_full.columns = X_test_full.columns.astype(str)


# Step 3: Model selection

"""# Define grid search range
param_grid = {
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.1, 0.01, 0.001]
}

# Create XGBoost model
model = XGBClassifier(n_jobs=6)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring="accuracy", cv=5)
grid_search.fit(X_train, y_train)

# Get best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)"""

# Fit the model with best grid search parameters
best_model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=200, n_jobs=6)

# Step 4: Model training

best_model.fit(X_train, y_train,
             eval_set=[(X_train, y_train), (X_valid, y_valid)],
             verbose=False)


# Step 5: Model evaluation

# Make predictions on training data
y_pred = best_model.predict(X_train)

# Cross-validation scores
scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")

# Print cross-validation scores for each fold
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Create confusion matrix
cm = confusion_matrix(y_train, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Create classification report
report = classification_report(y_train, y_pred)
print("Classification Report:\n", report)


# Step 6: Predict using test data

# Make predictions on test data - convert to bool
y_test_pred = best_model.predict(X_test_full)
y_test_pred = y_test_pred.astype(bool)

# Export predictions to csv
submission_df = pd.DataFrame({"PassengerId": X_test_full.index.astype(str), "Transported": y_test_pred})
print(submission_df.info())
submission_df.to_csv("submission.csv", index=False)
