import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Simulating traffic data
np.random.seed(42)

# Generate timestamps (for 1 day, every 5 minutes)
timestamps = pd.date_range('2024-09-01', periods=288, freq='5T')

# Simulate traffic volume (number of cars at a signal)
traffic_volume = np.random.randint(50, 500, size=288)

# Simulate weather conditions (0: clear, 1: rainy)
weather_conditions = np.random.randint(0, 2, size=288)

# Create DataFrame
traffic_data = pd.DataFrame({
    'timestamp': timestamps,
    'traffic_volume': traffic_volume,
    'weather_conditions': weather_conditions
})

# Data Preprocessing
# Handling missing values (if any)
traffic_data.fillna(method='ffill', inplace=True)

# Scaling traffic volume
scaler = StandardScaler()
traffic_data['scaled_volume'] = scaler.fit_transform(traffic_data[['traffic_volume']])

# One-hot encoding weather conditions
traffic_data = pd.get_dummies(traffic_data, columns=['weather_conditions'])

traffic_data.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Feature and target selection
X = traffic_data.drop(columns=['timestamp', 'traffic_volume'])
y = traffic_data['traffic_volume']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


import random
import numpy as np

# Traffic Signal Environment
class TrafficSignalEnv:
    def __init__(self):
        self.state = 0  # 0: Green, 1: Red
        self.traffic_volume = random.randint(50, 500)

    def step(self, action):
        if action == 1 and self.state == 1:
            self.state = 0  # Change to green
            reward = self.traffic_volume / 100  # Reduce congestion
        elif action == 0 and self.state == 0:
            self.state = 1  # Change to red
            reward = -self.traffic_volume / 100  # Increase congestion
        else:
            reward = 0
        self.traffic_volume = random.randint(50, 500)  # New traffic volume
        return self.state, reward

# Instantiate the environment
env = TrafficSignalEnv()

# Q-learning parameters
q_table = np.zeros([2, 2])  # 2 states (red, green) and 2 actions (keep, switch)
alpha = 0.1
gamma = 0.6
epsilon = 0.1
n_episodes = 10000

for episode in range(n_episodes):
    state = env.state
    if random.uniform(0, 1) < epsilon:
        action = random.choice([0, 1])  # Explore
    else:
        action = np.argmax(q_table[state])  # Exploit

    next_state, reward = env.step(action)

    old_value = q_table[state, action]
    next_max = np.max(q_table[next_state])

    # Q-learning formula
    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    q_table[state, action] = new_value

    state = next_state

# Display the learned Q-values
print("Q-table values:")
print(q_table)



from flask import Flask, jsonify
from datetime import datetime
import random

app = Flask(__name__)

# Simulated traffic environment
current_traffic = {
    'volume': random.randint(50, 500),
    'signal_state': 'Green' if env.state == 0 else 'Red'
}

@app.route('/traffic_status', methods=['GET'])
def get_traffic_status():
    current_time = datetime.now()
    signal_state = "Green" if env.state == 0 else "Red"
    
    # Simulate real-time traffic volume
    traffic_volume = random.randint(50, 500)
    
    return jsonify({
        'timestamp': current_time,
        'traffic_volume': traffic_volume,
        'signal_state': signal_state
    })

@app.route('/predict', methods=['GET'])
def predict_traffic():
    # Use the trained model to predict traffic volume
    example_data = np.array([[1, 0, random.randint(0, 1)]]).reshape(1, -1)  # Example input
    prediction = rf_model.predict(example_data)
    
    return jsonify({
        'predicted_traffic_volume': prediction[0]
    })

if __name__ == '__main__':
    app.run(debug=True)


# Running the Project:
# Install dependencies: Run pip install flask scikit-learn numpy pandas.
# Run the Flask app: In your terminal, run python app.py.
# Access API endpoints:
# Go to http://localhost:5000/traffic_status to get real-time traffic data.
# Go to http://localhost:5000/predict to get traffic predictions based on the trained model.