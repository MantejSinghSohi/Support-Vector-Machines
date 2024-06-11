# NAME- MANTEJ SINGH SOHI
# ROLL NO- 22AG10024

# SUPPORT VECTOR CLASSIFIER
import tensorflow as tf

# Load MNIST dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flattening the data
x_train_flat= x_train.reshape(-1, 28 * 28)
x_test_flat= x_test.reshape(-1, 28 * 28)


# normalizing the data
x_train_norm= x_train_flat/255.0
x_test_norm= x_test_flat/255.0

# Fetch first 10,000 samples from training data
x_train_subset = x_train_norm[:10000]
y_train_subset = y_train[:10000]

# Fetch 2000 samples from testing data
x_test_subset = x_test_norm[:2000]
y_test_subset = y_test[:2000]

# Verify the shapes of the subsets
print("Shape of x_train_subset:", x_train_subset.shape)
print("Shape of y_train_subset:", y_train_subset.shape)
print("Shape of x_test_subset:", x_test_subset.shape)
print("Shape of y_test_subset:", y_test_subset.shape)

# PART 2

from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Define kernel types
kernels = ['linear', 'poly', 'rbf']

# Train SVC with different kernels
for kernel in kernels:
    print(f"Training SVC for {kernel} kernel...")
    svc = SVC(kernel=kernel)
    svc.fit(x_train_subset, y_train_subset)

    # Predict on test set
    y_pred = svc.predict(x_test_subset)

    # Generate classification report
    print(f"Classification Report for SVC with {kernel} kernel:")
    print(classification_report(y_test_subset, y_pred))

# PART 3

# Importing GridSearchCV and RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define parameter grid for GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}

# Perform GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(x_train_subset, y_train_subset)

# Identify the best hyperparameters for GridSearchCV
print("Best hyperparameters from GridSearchCV:")
print(grid_search.best_params_)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(SVC(), param_grid, cv=5, n_iter=10, random_state=42)
random_search.fit(x_train_subset, y_train_subset)

# Identify the best hyperparameters for RandomizedSearchCV
print("\nBest hyperparameters from RandomizedSearchCV:")
print(random_search.best_params_)

# PART 4

# For GridSearchCV
best_params_grid = grid_search.best_params_
best_C_grid = best_params_grid['C']
best_gamma_grid = best_params_grid['gamma']

# Training the SVC model with RBF kernel using the best hyperparameters obtained from the tuning step (GridSearchCV)
svc_model_1 = SVC(C= best_C_grid, kernel='rbf', gamma= best_gamma_grid)
svc_model_1.fit(x_train_subset, y_train_subset)

# Predict on test set
y_pred = svc.predict(x_test_subset)

# Generate classification report
print(f"Classification Report for SVC with rbf kernel for best hyperparameters tuned using GridSearchCV:")
print(classification_report(y_test_subset, y_pred))

# PART 5

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
conf_mat = confusion_matrix(y_test_subset, y_pred)

# Plotting confusion matrix for final model
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# SUPPORT VECTOR REGRESSION
# PART 1

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California housing dataset
california_housing = fetch_california_housing()

# Split features (X) and target variable (y)
X = california_housing.data
y = california_housing.target

# Divide the dataset into training and testing datasets using a test size of 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Print the shapes of the training and testing datasets
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# PART 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Initialize SVR model with default parameters and epsilon set to 0.5
svr_model = SVR(epsilon=0.5)

# Train the SVR model on the training data
svr_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svr_model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Create scatter plot visualization of predictions versus ground truth
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.3) # reduced the alpha value to make the graph look less
plt.title('SVR: Predictions vs. Ground Truth')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.show()

# PART 3

# Define a range of epsilon values from 0 to 2.5 with a step size of 0.1
epsilon_values = np.arange(0, 2.6, 0.1)

# Set up a parameter grid for GridSearchCV with the epsilon values
param_grid = {'epsilon': epsilon_values}

# Initialize SVR model
svr_model = SVR()

# Perform 10-fold cross-validated grid search using GridSearchCV
grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=10)
grid_search.fit(X_train, y_train)

# Print the best epsilon parameter obtained from GridSearchCV
best_epsilon = grid_search.best_params_['epsilon']
print("Best epsilon parameter obtained:", best_epsilon)

# PART 4

# Initialize SVR model with the best epsilon parameter
svr_model = SVR(epsilon=best_epsilon)

# Train the SVR model on the training data
svr_model.fit(X_train, y_train)

# Predict the target variable for the testing set
y_pred = svr_model.predict(X_test)

# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Create scatter plot visualization of predictions versus ground truth
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.3)  # Reduce alpha to make points more transparent
plt.title('SVR: Predictions vs. Ground Truth')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.show()