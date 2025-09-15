
This study presents a comparative evaluation of the LightGBM and XGBoost algorithms for the task of next-day (J+1) daily precipitation forecasting. The analysis utilizes a comprehensive dataset of 8,765 daily meteorological observations spanning the entire continental United States over a 24-year period (2000-2023). The research focuses on assessing predictive performance in relation to seasonal climatic variables and evaluates model robustness against interannual variability. With a pedagogical objective, the study aims to identify the most influential climatic determinants for short-term hydrometeorological prediction.

# Dataset: https://www.kaggle.com/datasets/shivamshinde1904/weather-data2000-2023
# Acknowledgment: shivamShinde1904




# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (copy-paste method)
df = pd.read_clipboard()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Precipitation_Tomorrow'] = df['Precipitation_Sum'].shift(-1)
df = df.dropna(subset=['Precipitation_Tomorrow'])

# Descriptive statistics
print("Descriptive statistics of precipitation:")
print(df['Precipitation_Sum'].describe())
print(f"\nNumber of dry days: {(df['Precipitation_Sum'] == 0).sum()}")
print(f"Number of rainy days: {(df['Precipitation_Sum'] > 0).sum()}")

# Prepare data for modeling
# Features (X): remove target and date
X = df.drop(['Precipitation_Tomorrow', 'Date'], axis=1)
# Target (y): next day precipitation
y = df['Precipitation_Tomorrow']
# 7. Split data into training (80%) and test (20%) sets
# shuffle=False to preserve temporal order
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
print(f"\nDataset dimensions:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# TRAINING AND EVALUATION OF LIGHTGBM
print("\n" + "="*50)
print("LightGBM")
print("="*50)

lgb_model = lgb.LGBMRegressor(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)
lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_test)

# Evaluation
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
lgb_mae = mean_absolute_error(y_test, lgb_pred)
lgb_r2 = r2_score(y_test, lgb_pred)

print(f"RMSE: {lgb_rmse:.4f}")
print(f"MAE: {lgb_mae:.4f}")
print(f"R²: {lgb_r2:.4f}")

# TRAINING AND EVALUATION OF XGBOOST
print("\n" + "="*50)
print("XGBoost")
print("="*50)

xgb_model = xgb.XGBRegressor(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

# Evaluation
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"RMSE: {xgb_rmse:.4f}")
print(f"MAE: {xgb_mae:.4f}")
print(f"R²: {xgb_r2:.4f}")

# VISUAL COMPARISON OF PERFORMANCE
plt.figure(figsize=(12, 6))

# Real vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, lgb_pred, alpha=0.5, label='LightGBM', s=10)
plt.scatter(y_test, xgb_pred, alpha=0.5, label='XGBoost', s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Precipitation')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature importance (LightGBM)
plt.subplot(1, 2, 2)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': lgb_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=True)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - LightGBM')
plt.tight_layout()
plt.show()

# ERROR ANALYSIS ON FIRST 100 TEST DAYS
plt.figure(figsize=(15, 6))
sample_size = min(100, len(y_test))
plt.plot(y_test.values[:sample_size], label='Actual', marker='o', markersize=3)
plt.plot(lgb_pred[:sample_size], label='LightGBM', marker='x', markersize=3)
plt.plot(xgb_pred[:sample_size], label='XGBoost', marker='s', markersize=3)
plt.xlabel('Days')
plt.ylabel('Precipitation (mm)')
plt.title('Comparison of Predictions on First 100 Test Days')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# DISPLAY BEST AND WORST PREDICTIONS
results = pd.DataFrame({
    'Actual': y_test.values,
    'LightGBM': lgb_pred,
    'XGBoost': xgb_pred
})
results['LightGBM_Error'] = np.abs(results['Actual'] - results['LightGBM'])
results['XGBoost_Error'] = np.abs(results['Actual'] - results['XGBoost'])

print("\nTop 5 LightGBM predictions (lowest error):")
print(results.nsmallest(5, 'LightGBM_Error')[['Actual', 'LightGBM', 'LightGBM_Error']])

print("\nBottom 5 LightGBM predictions (highest error):")
print(results.nlargest(5, 'LightGBM_Error')[['Actual', 'LightGBM', 'LightGBM_Error']])

# Residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test - lgb_pred, alpha=0.5, label='LightGBM', s=10)
plt.scatter(y_test, y_test - xgb_pred, alpha=0.5, label='XGBoost', s=10)
plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors='r', linestyles='dashed', lw=2)
plt.xlabel('Actual Precipitation (mm)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(y_test - lgb_pred, bins=50, color='blue', alpha=0.5, label='LightGBM')
sns.histplot(y_test - xgb_pred, bins=50, color='green', alpha=0.5, label='XGBoost')
plt.xlabel('Prediction Error (mm)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

