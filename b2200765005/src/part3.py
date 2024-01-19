import numpy as np
import os
import glob
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Logistic Regression class with additional multi-class capabilities
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.theta = None  # Use a single numpy array for theta, not a list

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = self.add_intercept(X)
        # Initialize weights for a single binary classifier
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

    def predict_prob(self, X):
        X = self.add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))  # Use self.theta directly

    def predict(self, X):
        return self.predict_prob(X).round()  

    def get_params(self, deep=True):
        return {"lr": self.lr, "num_iter": self.num_iter}

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

# Function to load images from a directory and preprocess them

def load_class_images(directory, class_name, target_size=(256, 256)):
    class_dir = os.path.join(directory, class_name)
    features = []
    for image_file in glob.glob(os.path.join(class_dir, '*.jpg')):
        image = Image.open(image_file).convert('RGB').resize(target_size)
        features.append(np.array(image).flatten())
    features = np.array(features)
    return features

def safe_load_class_images(directory, class_name, target_size=(256, 256)):
    try:
        return load_class_images(directory, class_name, target_size)
    except FileNotFoundError:
        return np.empty((0, target_size[0] * target_size[1] * 3)) 


# Function to load and preprocess query images
def load_query_images(directory, target_size=(256, 256)):
    features = []
    image_files = glob.glob(os.path.join(directory, '*.jpg'))
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB').resize(target_size)
        features.append(np.array(image).flatten())
    features = np.array(features)
    return features

models = {}
scaler = MinMaxScaler()

base_directory="src/Dataset_v1/Dataset2"




def train_model_for_class(base_directory, class_name, scaler):
    X = safe_load_class_images(base_directory, class_name)
    if X.size == 0:
        return None  # Return None if there are no images for this class

    X = scaler.fit_transform(X)  # Normalize features

    # Combine the current class images with all other classes
    X_combined = X
    y_combined = np.ones(X.shape[0])

    for other_class_name in os.listdir(base_directory):
        if other_class_name == class_name or other_class_name == 'QUERY_IMAGES':
            continue  # Skip the current class and the QUERY_IMAGES folder

        X_other = safe_load_class_images(base_directory, other_class_name)
        if X_other.size == 0:
            continue  # Skip if there are no images for this class

        X_other = scaler.transform(X_other)  # Use the same scaler
        y_other = np.zeros(X_other.shape[0])

        X_combined = np.vstack((X_combined, X_other))
        y_combined = np.hstack((y_combined, y_other))

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    model = LogisticRegression(lr=0.01, num_iter=3000)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Task 1: Train models for 'airplane' and 'bear' classes
specific_classes = ['airplane', 'bear']
specific_models = {}
accuracies = []

scaler = MinMaxScaler()
for class_name in specific_classes:
    model, X_test, y_test = train_model_for_class(base_directory, class_name, scaler)
    if model:
        specific_models[class_name] = model
        accuracy = model.score(X_test, y_test)
        accuracies.append([class_name, accuracy])
        print(f"Accuracy for class {class_name}: {accuracy}")

# Creating a table with these accuracies
plt.figure(figsize=(6, 2))  # Adjust size as needed
ax = plt.gca()
ax.axis('off')

# Create the table and add it to the plot
table = plt.table(cellText=accuracies, colLabels=['Class', 'Accuracy'], loc='center', cellLoc='center')

# Adjust table style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Adjust these parameters to fit your data

# Save the table as a JPEG file
plt.savefig('class_accuracies.jpeg', format='jpeg', dpi=300)
plt.close()

# Task 2: Train models for all classes
all_models = {}
accuracies = []

scaler = MinMaxScaler()
for class_name in os.listdir(base_directory):
    if class_name == 'QUERY_IMAGES' or not os.path.isdir(os.path.join(base_directory, class_name)):
        continue
    model, X_test, y_test = train_model_for_class(base_directory, class_name, scaler)
    if model:
        all_models[class_name] = model
        accuracy = model.score(X_test, y_test)
        accuracies.append([class_name, accuracy])
        print(f"Accuracy for class {class_name}: {accuracy}")

# Creating a table with these accuracies
plt.figure(figsize=(10, len(accuracies)/2))  # Adjust size based on the number of classes
ax = plt.gca()
ax.axis('off')

# Create the table and add it to the plot
table = plt.table(cellText=accuracies, colLabels=['Class', 'Accuracy'], loc='center', cellLoc='center')

# Adjust table style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Adjust these parameters to fit your data

# Save the table as a JPEG file
plt.savefig('class_accuracies_all.jpeg', format='jpeg', dpi=300)
plt.close()

# Function to predict query images and get predicted classes
def predict_query_images(models, query_directory, scaler):
   def predict_query_images(models, query_directory, scaler):
    query_features = load_query_images(query_directory)
    if query_features.size == 0:
        print("No query images found.")
        return []
    
    query_features = scaler.transform(query_features)  # Use the same scaler


    query_preds = np.zeros((query_features.shape[0], len(models)))
    for i, (class_name, model) in enumerate(models.items()):
        query_preds[:, i] = model.predict_prob(query_features).flatten()

    predicted_labels = np.argmax(query_preds, axis=1)
    predicted_classes = [list(models.keys())[label] for label in predicted_labels]
    return predicted_classes

# Predict using specific models
query_directory = "Dataset_v1/Dataset2/QUERY_IMAGES"
specific_predicted_classes = predict_query_images(specific_models, query_directory, scaler)

# Predict using all models
all_predicted_classes = predict_query_images(all_models, query_directory, scaler)

