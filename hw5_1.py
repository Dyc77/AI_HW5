import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load and preprocess the Iris dataset
iris = load_iris()
data, targets = iris.data, iris.target

# One-hot encode the targets
encoder = OneHotEncoder(sparse_output=False)
targets = encoder.fit_transform(targets.reshape(-1, 1))

# Standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

# Build a simple Neural Network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define TensorBoard callback
log_dir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model with TensorBoard logging
model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Instruction to view TensorBoard
print("\nTo visualize training progress, run: tensorboard --logdir=logs")
