# 📦 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# Load Kaggle datasets (make sure the files are in your project folder)
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#  Explore the Dataset (EDA)
print("\nData Info:")
print(train_data.info())

print("First 5 Rows of Training Data:")
print(train_data.head())
print()

print("Missing Values:")
print(train_data.isnull().sum().sort_values(ascending=False).head(10))
print()
plt.figure(figsize=(7, 5))
sns.histplot(train_data['SalePrice'], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Sale Price")
plt.ylabel("Count")
plt.show()
# Select only numeric columns for simplicity
numeric_features = train_data.select_dtypes(include=[np.number])

# Drop rows with missing values (simplified for beginner version)
numeric_features = numeric_features.dropna()
# Choose features correlated with SalePrice
corr = numeric_features.corr()
top_features = corr['SalePrice'].sort_values(ascending=False).head(10)
print("Top correlated features with SalePrice:")
print(top_features)
print()

selected_features = top_features.index.tolist()
selected_features.remove('SalePrice')

X = numeric_features[selected_features]
y = numeric_features['SalePrice']
# 6. Split Dataset into Train/Test

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Validation set size:", X_valid.shape)
print()
#  7. Model Training

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_valid)

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_valid)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_valid)
#  Model Evaluation

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📈 {model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    return r2
r2_lr = evaluate_model(y_valid, lr_pred, "Linear Regression")
r2_dt = evaluate_model(y_valid, dt_pred, "Decision Tree")
r2_rf = evaluate_model(y_valid, rf_pred, "Random Forest")
#  Select Best Model
# test
best_model = rf_model
print("\n✅ Best model selected: Random Forest")
# Save the Model

joblib.dump(best_model, "house_price_model.pkl")
print("\n💾 Model saved as 'house_price_model.pkl'")
# Generate Predictions for Test Data
# ============================================

# Keep only numeric columns in test data
test_numeric = test_data.select_dtypes(include=[np.number])
test_numeric = test_numeric.fillna(test_numeric.median())

# Use same features as training
X_test = test_numeric[selected_features]

# Predict
test_predictions = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

submission.to_csv("submission.csv", index=False)
print("\n📄 Predictions saved as 'submission.csv'")

print("\n🎉 House Price Prediction project completed successfully!")
# fku,ku,k



