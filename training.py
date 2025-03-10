import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# Load JSON data
with open("merged.json", "r") as f:
    data = json.load(f)

X, y = [], []

# Loop through each dictionary in the list
for player_dict in data:
    for player, stats in player_dict.items():  # Extract players from each dictionary
        try:
            features = [
                float(stats["matches"]) if stats.get("matches") is not None else np.nan,
                float(stats["runs_made"]) if stats.get("runs_made") is not None else np.nan,
                float(stats["strike_rate"]) if stats.get("strike_rate") is not None else np.nan,
                float(stats["batting_average"]) if stats.get("batting_average") is not None else np.nan,
                float(stats["bowling_average"]) if stats.get("bowling_average") is not None else np.nan,
                float(stats["wickets"]) if stats.get("wickets") is not None else np.nan,
                float(stats["economy_rate"]) if stats.get("economy_rate") is not None else np.nan,
            ]
            
            labels = [
                float(stats["batting"]) if stats.get("batting") is not None else np.nan,
                float(stats["runs"]) if stats.get("runs") is not None else np.nan,
                float(stats["bowling"]) if stats.get("bowling") is not None else np.nan,
            ]

            # Only add if no NaN values
            if not any(np.isnan(features)) and not any(np.isnan(labels)):
                X.append(features)
                y.append(labels)

        except Exception as e:
            pass

X = np.array(X)
y = np.array(y)

# Fill missing values with the median of each column
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)
y = imputer.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(7,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(3)
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train the model with tqdm progress bar
EPOCHS = 1000
BATCH_SIZE = 16

for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Save the model
model.save("player_performance_model.keras")  # New format
joblib.dump(scaler, "scaler.pkl")