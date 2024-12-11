import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 加載 MNIST 數據集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 資料預處理
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 建立 Dense NN 模型
def create_dense_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    return model

# 建立 CNN 模型
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# 訓練模型並繪製結果
def train_and_plot(model, x_train, y_train, x_test, y_test, epochs=10):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, 
                        validation_data=(x_test, y_test), 
                        epochs=epochs, 
                        batch_size=128, 
                        verbose=2)

    # 繪製訓練損失和準確率
    plt.figure(figsize=(12, 5))

    # 繪製損失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 繪製準確率
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# 訓練 Dense NN 模型
dense_model = create_dense_model()
print("Training Dense NN model...")
train_and_plot(dense_model, x_train, y_train, x_test, y_test, epochs=10)

# 訓練 CNN 模型
cnn_model = create_cnn_model()
print("Training CNN model...")
train_and_plot(cnn_model, x_train, y_train, x_test, y_test, epochs=10)
