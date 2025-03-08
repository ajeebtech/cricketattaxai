import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib  

# Load the pre-trained scaler (saved in training script)
scaler = joblib.load("scaler.pkl")  # Ensure this file exists

# Load the pre-trained model
model = keras.models.load_model("player_performance_model.keras", compile=False)

# Example new player data: [matches, runs_made, strike_rate, batting_avg, bowling_avg, wickets, economy_rate]
new_player = np.array([[1000, 85.4, 35.6, 22.8, 10, 6.5]])

# Scale the input
new_player_scaled = scaler.transform(new_player)

# Predict batting, runs, and bowling
predicted_values = model.predict(new_player_scaled)
predicted_values[0] = [int(x) for x in predicted_values[0]]
print("Predicted Batting, Runs, Bowling:", predicted_values[0])