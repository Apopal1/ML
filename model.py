import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

# Φόρτωση MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Προεπεξεργασία
train_images = train_images.reshape((-1, 28, 28, 1)) / 255.0
test_images = test_images.reshape((-1, 28, 28, 1)) / 255.0

MODEL_FILE = 'V4.h5'  # Χρησιμοποίησε .h5 για σαφήνεια

if os.path.exists(MODEL_FILE):
    print("Φόρτωση μοντέλου...")
    model = tf.keras.models.load_model(MODEL_FILE)
    # Μεταγλώττιση με τις ίδιες παραμέτρους για να αποφύγουμε προειδοποιήσεις
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
else:
    print("Δημιουργία και εκπαίδευση νέου μοντέλου...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
    model.save(MODEL_FILE)

# Έλεγχος ακρίβειας πριν τις προβλέψεις
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# Πρόβλεψη σε τυχαία εικόνα
index = np.random.randint(0, len(test_images))
test_image = test_images[index]
prediction = model.predict(test_image[np.newaxis, ...])
predicted_label = np.argmax(prediction)

plt.imshow(test_image.squeeze(), cmap='gray')
plt.title(f"Predicted: {predicted_label}, Actual: {test_labels[index]}")
plt.axis('off')
plt.show()