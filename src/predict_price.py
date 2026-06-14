"""
House Price Prediction - Prediction Script
Loads the trained model and predicts price for new house features.
Usage: python predict_price.py
       Or modify the sample input values in the script.
"""

import numpy as np
import joblib
import pandas as pd

# Load the trained model and scaler
print("Loading model and scaler...")
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
print("Model loaded successfully.")

# Feature names (must match the order used in training)
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                 'Population', 'AveOccup', 'Latitude', 'Longitude']

def predict_price(features):
    """
    Predict house price from a list or array of feature values.
    Features order: MedInc, HouseAge, AveRooms, AveBedrms, 
                    Population, AveOccup, Latitude, Longitude
    Returns predicted price (in $100,000s).
    """
    # Convert to numpy array and reshape for scaler
    features_array = np.array(features).reshape(1, -1)
    # Scale features
    features_scaled = scaler.transform(features_array)
    # Predict
    price = model.predict(features_scaled)[0]
    return price

# Example: Predict for a sample house (you can modify these values)
# This is just an example; you can enter your own values interactively
sample_house = {
    'MedInc': 5.0,      # Median income in block (tens of thousands)
    'HouseAge': 30.0,   # Median house age
    'AveRooms': 6.0,    # Average rooms per household
    'AveBedrms': 1.0,   # Average bedrooms per household
    'Population': 1500, # Population in block
    'AveOccup': 3.0,    # Average household occupancy
    'Latitude': 34.5,   # Latitude coordinate
    'Longitude': -118.0 # Longitude coordinate
}

print("\n--- Example Prediction ---")
print("Input features:")
for name, value in sample_house.items():
    print(f"  {name}: {value}")

price_pred = predict_price(list(sample_house.values()))
print(f"\nPredicted House Price: ${price_pred*100000:.2f} (in dollars)")
print(f"Or: {price_pred:.3f} (in $100,000s)")

# Optional: Allow user to input custom features
print("\n--- Enter your own values (or press Enter to skip) ---")
try:
    user_input = input("Do you want to enter custom features? (y/n): ").strip().lower()
    if user_input == 'y':
        print("\nEnter values for the following features (comma-separated):")
        print("Order: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude")
        values = input("> ").strip()
        if values:
            custom_features = [float(x) for x in values.split(',')]
            if len(custom_features) == 8:
                price_custom = predict_price(custom_features)
                print(f"\nPredicted House Price: ${price_custom*100000:.2f}")
            else:
                print("Error: Please enter exactly 8 numbers.")
except KeyboardInterrupt:
    print("\nExiting.")