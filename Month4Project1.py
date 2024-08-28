import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess the MNIST Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape the data to fit the model (samples, height, width, channels)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# Convert the labels to one-hot encoded format
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 2: Define the ANN Architecture
model = models.Sequential()

# Flatten the input image to a vector
model.add(layers.Flatten(input_shape=(28, 28, 1)))

# Hidden layer 1
model.add(layers.Dense(512, activation='relu'))

# Hidden layer 2
model.add(layers.Dense(256, activation='relu'))

# Hidden layer 3
model.add(layers.Dense(128, activation='relu'))

# Output layer
model.add(layers.Dense(10, activation='softmax'))

# Step 3: Compile the Model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(X_train, y_train, 
                    epochs=10, 
                    batch_size=128, 
                    validation_split=0.2)

# Step 5: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Step 6: Visualize Training History
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Step 7: Make Predictions
# Predict the labels for the test set
predictions = model.predict(X_test)

# Display a test image and its predicted label
index = 0
plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
plt.title(f'Predicted label: {np.argmax(predictions[index])}')
plt.show()
