import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Step 1: Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images to range from 0 to 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape the data to add a channel dimension (for grayscale images)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# Step 2: Visualize the dataset (optional)
fig, axes = plt.subplots(1, 5, figsize=(10, 10))
for i in range(5):
    axes[i].imshow(X_train[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f"Label: {y_train[i]}")
    axes[i].axis('off')
plt.show()

# Step 3: Build the Convolutional Neural Network (CNN) model
model = models.Sequential([
    # First Convolutional Layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten the data for the fully connected layers
    layers.Flatten(),
    
    # Fully connected layer
    layers.Dense(64, activation='relu'),
    
    # Output layer (10 neurons for 10 classes, softmax for multi-class classification)
    layers.Dense(10, activation='softmax')
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Multi-class classification loss
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Step 6: Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Step 7: Make predictions on the test data
predictions = model.predict(X_test)

# Step 8: Visualize predictions on the first 5 test images
fig, axes = plt.subplots(1, 5, figsize=(10, 10))
for i in range(5):
    axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f"True: {y_test[i]}, Pred: {predictions[i].argmax()}")
    axes[i].axis('off')
plt.show()

# Step 9: Save the model (optional)
model.save('digit_recognition_model.h5')
