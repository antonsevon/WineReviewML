import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor

# Reads the CSV file
df = pd.read_csv('winequality-red.csv')

# Prints the dataset dimensions
print(f"Dataset dimensions: {df.shape[0]} samples, {df.shape[1]} columns")

# Prints the number of missing values
print(f"Missing values: {df.isnull().sum().sum()}")

# Splits the data into features (X) and target label (y)
X = df.drop('quality', axis=1).to_numpy()
y = df['quality'].to_numpy()

# Splits the data into test data (15%) and combined training and validation data (85%)
X_comb, X_test, y_comb, y_test = train_test_split(
    X, y, test_size = 0.15, random_state = 69
)

# Splits the combined data into training data (70%) and validation data (15%)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_comb, y_comb, test_size = (0.15 / 0.85), random_state = 69
)

# Prints the sizes of the train, validation, and test sets
print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(df):.1%})")
print(f"Validation set size: {X_valid.shape[0]} samples ({X_valid.shape[0]/len(df):.1%})")
print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(df):.1%})")

# Displays the first five rows of the dataset
df.head(5)

## Finds the best parameter combinations using GridSearchCV
# AI was used in this part in the GridSearch implementation

# Random Forest Regression

# Grid of parameters to test
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None], # None allows trees to grow without limit
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 1.0]
}

rf_base = RandomForestRegressor(random_state=69)

# Searches for the best parameter combinations using 5-fold cross-validation
rf_grid_search = GridSearchCV(estimator=rf_base,
                              param_grid=rf_param_grid,
                              cv=5,
                              scoring='neg_root_mean_squared_error',
                              n_jobs=-1)

rf_grid_search.fit(X_train, y_train)

# Selects and saves the best model
best_rf_model = rf_grid_search.best_estimator_


# Gradient Boosting Regression

# Grid of parameters to test
gb_param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

gb_base = GradientBoostingRegressor(random_state=69)

# Searches for the best parameter combinations using 5-fold cross-validation
gb_grid_search = GridSearchCV(estimator=gb_base,
                              param_grid=gb_param_grid,
                              cv=5,
                              scoring='neg_root_mean_squared_error',
                              n_jobs=-1)

gb_grid_search.fit(X_train, y_train)

# Selects and saves the best model
best_gb_model = gb_grid_search.best_estimator_


# Prints the best hyperparameters of each model
rf_p = rf_grid_search.best_params_
gb_p = gb_grid_search.best_params_

print(f"""Best parameters for Random Forest Regression:
    Number of trees: {rf_p['n_estimators']}
    Max tree depth: {rf_p['max_depth']}
    Min samples to split a node: {rf_p['min_samples_split']}
    Max features: {rf_p['max_features']}
""")

print(f"""Best parameters for Gradient Boosting Regression:
    Number of trees: {gb_p['n_estimators']}
    Learning rate: {gb_p['learning_rate']}
    Max tree depth: {gb_p['max_depth']}
""")


# Predicts on training and validation sets
train_predF = best_rf_model.predict(X_train)
valid_predF = best_rf_model.predict(X_valid)

train_predG = best_gb_model.predict(X_train)
valid_predG = best_gb_model.predict(X_valid)

# Calculates the root mean squared errors (RMSE)
train_rmseF = np.sqrt(mean_squared_error(y_train, train_predF))
valid_rmseF = np.sqrt(mean_squared_error(y_valid, valid_predF))

train_rmseG = np.sqrt(mean_squared_error(y_train, train_predG))
valid_rmseG = np.sqrt(mean_squared_error(y_valid, valid_predG))

# Prints the RMSE on train and validation sets
print(f"Training set RMSE using Random Forest Regression: {train_rmseF:.4f}")
print(f"Validation set RMSE using Random Forest Regression: {valid_rmseF:.4f}\n")

print(f"Training set RMSE using Gradient Boosting Regression: {train_rmseG:.4f}")
print(f"Validation set RMSE using Gradient Boosting Regression: {valid_rmseG:.4f}")

# Calculates the root mean squared error (RMSE) for the test data
test_predF = best_rf_model.predict(X_test)

test_rmseF = np.sqrt(mean_squared_error(y_test, test_predF))

print(f"Test set RMSE using Random Forest Regression: {test_rmseF:.4f}")

# Plots a histogram of the quality scores in the dataset
plt.figure(figsize=(8, 5))

sns.countplot(x=df['quality'], color='steelblue')

plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality Score (0-10)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

features = df.drop('quality', axis=1).columns

# Calculates and plots feature importances for Random Forest
rf_importances = pd.Series(best_rf_model.feature_importances_, index=features)
rf_importances.sort_values().plot(kind='barh', ax=axes[0, 0], color='gold')
axes[0, 0].set_title("Random Forest: Feature Importances")
axes[0, 0].set_xlabel("Importance Score")

# Calculates and plots feature importances for Gradient Boosting
gb_importances = pd.Series(best_gb_model.feature_importances_, index=features)
gb_importances.sort_values().plot(kind='barh', ax=axes[0, 1], color='saddlebrown')
axes[0, 1].set_title("Gradient Boosting: Feature Importances")
axes[0, 1].set_xlabel("Importance Score")

# Plots actual vs. predicted scatter plot for Random Forest
axes[1, 0].scatter(y_valid, valid_predF, alpha=0.6, color='gold')
axes[1, 0].plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], "k--", lw=2, label="Ideal Fit")
axes[1, 0].set_title("Random Forest: Actual vs. Predicted")
axes[1, 0].set_xlabel("Actual Quality Score")
axes[1, 0].set_ylabel("Predicted Quality Score")
axes[1, 0].legend()

# Plots actual vs. predicted scatter plot for Gradient Boosting
axes[1, 1].scatter(y_valid, valid_predG, alpha=0.6, color='saddlebrown')
axes[1, 1].plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], "k--", lw=2, label="Ideal Fit")
axes[1, 1].set_title("Gradient Boosting: Actual vs. Predicted")
axes[1, 1].set_xlabel("Actual Quality Score")
axes[1, 1].set_ylabel("Predicted Quality Score")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Evaluates a baseline Dummy Regressor on the test set
baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

print(f"Test set RMSE using Dummy Regressor: {baseline_rmse:.4f}")

(1 - (test_rmseF / baseline_rmse)) * 100