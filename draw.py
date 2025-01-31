import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageDraw

# Φόρτωση του εκπαιδευμένου μοντέλου
model = tf.keras.models.load_model('V4.h5', compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Δημιουργία παραθύρου για ζωγραφική
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ζωγράφισε έναν αριθμό")
        
        # Δημιουργία καμβά
        self.canvas = Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()
        
        # Ρύθμιση πινέλου
        self.image = Image.new("L", (280, 280), 0)  # Μαύρο φόντο
        self.draw = ImageDraw.Draw(self.image)
        
        # Δέσμευση events
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Κουμπιά
        self.btn_predict = Button(root, text="Πρόβλεψη", command=self.predict)
        self.btn_predict.pack(side=LEFT)
        self.btn_clear = Button(root, text="Καθαρισμός", command=self.clear)
        self.btn_clear.pack(side=RIGHT)
        
    def paint(self, event):
        # Ζωγραφίζει στο canvas και στην εικόνα
        x, y = event.x, event.y
        radius = 8
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="white", outline="white")
        self.draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=255)  # Λευκός κύκλος

    def clear(self):
        # Επαναφορά καμβά
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # Μετατροπή εικόνας σε 28x28 (όπως το MNIST)
        img = self.image.resize((28, 28))  # Resize
        img = np.array(img) / 255.0        # Κανονικοποίηση (0-1)
       # img = 1 - img                      # Αντιστροφή χρωμάτων (μαύρο -> άσπρο)
        img = img.reshape(1, 28, 28, 1)    # Προσθήκη διαστάσεων batch και channel
        
        # Πρόβλεψη
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)
        
        # Εμφάνιση αποτελέσματος
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Πρόβλεψη: {predicted_label}")
        plt.axis('off')
        plt.show()
        self.clear()

# Εκκίνηση της εφαρμογής
root = Tk()
app = DrawingApp(root)
root.mainloop()