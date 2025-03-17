import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Set random seed for reproducibility
np.random.seed(42)

# Create structured synthetic data for three categories (digits, faces, objects)
num_samples_per_class = 200
image_size = 28 * 28  # 28x28 pixel images

# Generate structured data for digits (pixel values mostly low)
digits = np.random.randint(0, 100, size=(num_samples_per_class, image_size))
digits_labels = np.array(['digit'] * num_samples_per_class)

# Generate structured data for faces (pixel values moderate range)
faces = np.random.randint(50, 200, size=(num_samples_per_class, image_size))
faces_labels = np.array(['face'] * num_samples_per_class)

# Generate structured data for objects (pixel values high intensity)
objects = np.random.randint(150, 255, size=(num_samples_per_class, image_size))
objects_labels = np.array(['object'] * num_samples_per_class)

# Combine all data
X = np.vstack((digits, faces, objects))
y = np.concatenate((digits_labels, faces_labels, objects_labels))

# Convert to DataFrame for saving
df = pd.DataFrame(X)
df['label'] = y

# Save dataset as CSV
df.to_csv("Realistic_Image_Dataset.csv", index=False)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naïve Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

# Function to preprocess and predict a real image
def preprocess_and_predict(image_path):
    """
    Loads an actual image file, preprocesses it (grayscale, resize, edge detection),
    and predicts its category using the trained Naïve Bayes model.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Predicted label (digit, face, or object)
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28
    img_resized = cv2.resize(img, (28, 28))

    # Apply Gaussian blur (reduce noise)
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)

    # Apply adaptive thresholding (binarization)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Canny edge detection
    img_edges = cv2.Canny(img_thresh, 30, 150)

    # Flatten into 1D array (784 pixels)
    img_flattened = img_edges.flatten()

    # Predict category
    predicted_label = model.predict([img_flattened])[0]

    # Display the image
    plt.imshow(img_edges, cmap='gray')
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

    return predicted_label

# Function to open file dialog, select an image, and predict it
def open_and_predict_real_image():
    """
    Opens a file dialog for the user to select an image, processes it, and predicts its category.
    """
    # Open file dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])

    if not file_path:
        print("No file selected.")
        return

    predicted_label = preprocess_and_predict(file_path)
    print(f"Predicted Label for Uploaded Image: {predicted_label}")

# Run the function to manually select and classify a real image
open_and_predict_real_image()
