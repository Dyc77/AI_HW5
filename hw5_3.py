import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reduce dataset size for smaller memory usage
x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Data augmentation
data_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
data_gen.fit(x_train)

# Build a model using pretrained VGG16 or VGG19

def create_model(base_model):
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# Load pretrained VGG16
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
for layer in vgg16_base.layers:
    layer.trainable = False
vgg16_model = create_model(vgg16_base)

# Load pretrained VGG19
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
for layer in vgg19_base.layers:
    layer.trainable = False
vgg19_model = create_model(vgg19_base)

# Train and evaluate a model
def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=10):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(data_gen.flow(x_train, y_train, batch_size=64),
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        verbose=2)

    # Plot the results
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Train VGG16 model
print("Training VGG16 model...")
train_and_evaluate(vgg16_model, x_train, y_train, x_test, y_test, epochs=10)

# Train VGG19 model
print("Training VGG19 model...")
train_and_evaluate(vgg19_model, x_train, y_train, x_test, y_test, epochs=10)
