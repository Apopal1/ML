import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

# Φόρτωση MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Κανονικοποίηση εικόνων (τιμές pixel από 0-255 σε 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Αλλαγή διαστάσεων για CNN (προσθήκη channel dimension)
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Ορισμός ονόματος αρχείου για το μοντέλο
MODEL_FILE = 'V5.h5'

# Έλεγχος αν το μοντέλο υπάρχει ήδη
if os.path.exists(MODEL_FILE):
    # Φόρτωση του εκπαιδευμένου μοντέλου
    print("Φόρτωση εκπαιδευμένου μοντέλου...")
    model = tf.keras.models.load_model(MODEL_FILE)
else:
    # Δημιουργία νέου μοντέλου και εκπαίδευση
    print("Δημιουργία και εκπαίδευση νέου μοντέλου...")
    
    # Ορισμός αρχιτεκτονικής μοντέλου
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Σύνθεση μοντέλου
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Εκπαίδευση
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    
    # Αποθήκευση μοντέλου
    model.save(MODEL_FILE)
    print("Το μοντέλο αποθηκεύτηκε ως", MODEL_FILE)

# Πρόβλεψη σε τυχαία εικόνα (αυτό τρέχει ΠΑΝΤΑ, είτε φορτώθηκε είτε εκπαιδεύτηκε το μοντέλο)
index = np.random.randint(0, len(test_images))
test_image = test_images[index]
prediction = model.predict(test_image[np.newaxis, ...])
predicted_label = np.argmax(prediction)

# Απεικόνιση αποτελεσμάτων
plt.imshow(test_image.squeeze(), cmap='gray')
plt.title(f"Predicted: {predicted_label}, Actual: {test_labels[index]}")
plt.axis('off')
plt.show()