import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


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


# Step 2: Data Preprocessing

# Train data exploration
print(X_train.shape)
print(X_train.head())
print(X_train.info())
print(X_train.describe())

# Check columns for missing values
missing_values = X_train.isnull().sum()
print(missing_values)

# Identify categorical vs. numerical columns
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
numerical_cols = [col for col in X_train.columns if X_train[col].dtype != "object"]

print(X_train[categorical_cols])

# Imputation using mode for categorical columns, check imputation statistics
categorical_imputer = SimpleImputer(strategy='most_frequent')
imputed_X_train_cat = pd.DataFrame(categorical_imputer.fit_transform(X_train[categorical_cols]))
imputed_X_valid_cat = pd.DataFrame(categorical_imputer.transform(X_valid[categorical_cols]))
print(categorical_imputer.statistics_)

imputed_X_train_cat.columns = X_train[categorical_cols].columns
X_train[categorical_cols] = imputed_X_train_cat
print(imputed_X_train_cat)
print(X_train)

"""# Imputation removes column names, put them back
imputed_X_train_cat.columns = X_train[categorical_cols].columns
imputed_X_valid_cat.columns = X_valid[categorical_cols].columns

# Imputation using mean for numerical columns, check imputation statistics
numerical_imputer = SimpleImputer(strategy='mean')
imputed_X_train_num = pd.DataFrame(numerical_imputer.fit_transform(X_train[numerical_cols]))
imputed_X_valid_num = pd.DataFrame(numerical_imputer.transform(X_valid[numerical_cols]))

imputed_X_train_num.columns = X_train.columns
imputed_X_valid_num.columns = X_valid.columns

# Check there are no more missing values
X_train_missing = X_train.isnull().sum()
X_valid_missing = X_valid.isnull().sum()
print(X_train.head())"""

