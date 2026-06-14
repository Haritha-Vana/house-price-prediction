"""
House Price Prediction - Training Script
Loads California Housing dataset, trains models, and saves the best one.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Create output directories if they don't exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

# 1. Load the data
print("Loading dataset...")
df = pd.read_csv('data/housing.csv')
print(f"Dataset shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}")

# 2. Exploratory Data Analysis (EDA)
print("\n--- Data Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# 3. Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 4. Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png')
plt.close()
print("Correlation heatmap saved to outputs/correlation_heatmap.png")

# 5. Distribution of target variable (house price)
plt.figure(figsize=(8,5))
sns.histplot(df['MedHouseVal'], bins=50, kde=True)
plt.title('Distribution of House Prices (MedHouseVal)')
plt.xlabel('Median House Value ($100,000s)')
plt.tight_layout()
plt.savefig('outputs/target_distribution.png')
plt.close()

# 6. Split features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 8. Scale features (important for linear models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use in prediction
joblib.dump(scaler, 'models/scaler.pkl')

# 9. Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_score = -np.inf

print("\n--- Model Training and Evaluation ---")
for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    # Predict
    y_pred = model.predict(X_test_scaled)
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'R2': r2, 'RMSE': rmse}
    print(f"{name}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")
    
    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"  CV R2 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Keep best model
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

# 10. Save the best model
joblib.dump(best_model, 'models/best_model.pkl')
print(f"\nBest model: {best_model_name} with R2 = {best_score:.4f}")
print("Model saved to models/best_model.pkl")

# 11. Create a comparison table
results_df = pd.DataFrame(results).T
print("\n--- Model Comparison ---")
print(results_df)
results_df.to_csv('outputs/model_comparison.csv')

# 12. Plot predictions of best model vs actual
y_pred_best = best_model.predict(X_test_scaled)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title(f'Best Model: {best_model_name}\nActual vs Predicted (R2={best_score:.4f})')
plt.tight_layout()
plt.savefig('outputs/actual_vs_predicted.png')
plt.close()

print("\nTraining complete! Outputs saved in 'outputs/' folder.")
print("Best model saved in 'models/' folder.")