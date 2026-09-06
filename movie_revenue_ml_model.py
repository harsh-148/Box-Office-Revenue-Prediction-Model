"""
Movie Revenue Prediction — Linear Regression Model
Author: Harsh Dagar

Extends the earlier Movie Correlation EDA project into a trained ML model
that predicts a movie's gross revenue from its production budget.

Dataset: Kaggle "Movie Industry" by Daniel Grijalva
Source: https://www.kaggle.com/datasets/danielgrijalvas/movies
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# STEP 1: Load the dataset
# Download movies.csv from Kaggle and place it in this same folder
# -------------------------------------------------------------
df = pd.read_csv("movies.csv")

# -------------------------------------------------------------
# STEP 2: Clean the data (same cleaning logic as your EDA project)
# -------------------------------------------------------------
df = df.dropna(subset=["budget", "gross"])
df = df[(df["budget"] > 0) & (df["gross"] > 0)]

# -------------------------------------------------------------
# STEP 3: Prepare features (X) and target (y)
# -------------------------------------------------------------
X = df[["budget"]]
y = df["gross"]

# -------------------------------------------------------------
# STEP 4: Train/test split
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------
# STEP 5: Train the model
# -------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------------------------------------
# STEP 6: Predict and evaluate
# -------------------------------------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Model coefficient (budget → gross): {model.coef_[0]:.2f}")
print(f"Model intercept: {model.intercept_:.2f}")
print(f"R² score: {r2:.3f}")
print(f"Mean Absolute Error: ${mae:,.0f}")

# -------------------------------------------------------------
# STEP 7: Visualize actual vs predicted
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, alpha=0.5, label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted (model)")
plt.xlabel("Budget ($)")
plt.ylabel("Gross Revenue ($)")
plt.title("Movie Revenue Prediction — Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("revenue_prediction_plot.png")
print("Plot saved as revenue_prediction_plot.png")
