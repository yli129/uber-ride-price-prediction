import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset and drop rows with missing target values
df = pd.read_csv("cab_rides.csv")
df = df.dropna(subset=["price"])

# Define input features and target
features = ["distance", "cab_type", "source", "destination"]
target = "price"

# One-hot encode for Linear Regression
df_lin_encoded = pd.get_dummies(df[features], drop_first=True)
X_lin = df_lin_encoded
y_lin = df[target]

# Split data into 80% training and 20% testing
X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
lin_model = LinearRegression()
lin_model.fit(X_lin_train, y_lin_train)
y_pred_lin = lin_model.predict(X_lin_test)
rmse_lin = mean_squared_error(y_lin_test, y_pred_lin) ** 0.5
r2_lin = r2_score(y_lin_test, y_pred_lin)

# Print evaluation results
print("Linear Regression Result (on full dataset):")
print("RMSE:", rmse_lin)
print("R^2 Score:", r2_lin)
