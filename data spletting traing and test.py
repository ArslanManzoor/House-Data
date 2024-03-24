import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Create a Complex Synthetic Dataset
# Read the CSV file (assuming the file is named "house_data.csv")
df = pd.read_csv("house_data.csv")

# Calculate the house price using the specified formula
df['house_price'] = (
    df['square_footage'] * 200 +
    df['num_bedrooms'] * 10000 +
    df['num_bathrooms'] * 5000 +
    df['age_of_home'] * -1000 +
    df['distance_to_city_center'] * -500 +
    df['local_school_rating'] * 2000 +
    np.random.normal(0, 25000, len(df))  # Adding some noise
)

# Step 2: Feature Selection
X = df.drop('house_price', axis=1)
y = df['house_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

selector = SelectKBest(f_regression, k=6)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Step 3: Linear Regression Model
model = LinearRegression()
model.fit(X_train_selected, y_train)
predictions = model.predict(X_test_selected)
mse = mean_squared_error(y_test, predictions)

# Step 4: Implementing Linear Regression with Gradient Descent
# Stochastic Gradient Descent
sgd_model = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000, tol=1e-3)
sgd_model.fit(X_train_selected, y_train)
sgd_predictions = sgd_model.predict(X_test_selected)
sgd_mse = mean_squared_error(y_test, sgd_predictions)


# Step 5: Analyze Results
# Experiment with different learning rates and analyze the results

# Print out the MSE for both models
print(f"Linear Regression MSE: {mse}")
print(f"SGD Regression MSE: {sgd_mse}")

# Note: The actual implementation of Batch Gradient Descent is not included here
# and would need to be added for a complete solution.

# Remember to split your data into training and testing sets to evaluate the model's performance properly.
