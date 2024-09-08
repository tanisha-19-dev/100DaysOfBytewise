import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Bidirectional, Conv1D, Flatten

# Generate synthetic time-series data
time_steps = 50
features = 1
X_train = np.random.randn(1000, time_steps, features)  # 1000 samples, each with 50 time steps, and 1 feature per time step
y_train = np.random.randn(1000, 1)

# 1. Basic RNN Model
print("Training Basic RNN Model...")
basic_rnn_model = Sequential()
basic_rnn_model.add(SimpleRNN(50, input_shape=(time_steps, features)))
basic_rnn_model.add(Dense(1))

basic_rnn_model.compile(optimizer='adam', loss='mse')
basic_rnn_model.fit(X_train, y_train, epochs=5)

# 2. Stacked RNN Model
print("\nTraining Stacked RNN Model...")
stacked_rnn_model = Sequential()
stacked_rnn_model.add(SimpleRNN(50, return_sequences=True, input_shape=(time_steps, features)))
stacked_rnn_model.add(SimpleRNN(50))
stacked_rnn_model.add(Dense(1))

stacked_rnn_model.compile(optimizer='adam', loss='mse')
stacked_rnn_model.fit(X_train, y_train, epochs=5)

# 3. Bi-directional RNN Model
print("\nTraining Bi-directional RNN Model...")
bi_rnn_model = Sequential()
bi_rnn_model.add(Bidirectional(SimpleRNN(50, input_shape=(time_steps, features))))
bi_rnn_model.add(Dense(1))

bi_rnn_model.compile(optimizer='adam', loss='mse')
bi_rnn_model.fit(X_train, y_train, epochs=5)

# 4. Hybrid RNN + CNN Model
print("\nTraining Hybrid (RNN + CNN) Model...")
hybrid_model = Sequential()
hybrid_model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(time_steps, features)))
hybrid_model.add(Flatten())
hybrid_model.add(SimpleRNN(50))
hybrid_model.add(Dense(1))

hybrid_model.compile(optimizer='adam', loss='mse')
hybrid_model.fit(X_train, y_train, epochs=5)

print("\nTraining Complete.")
