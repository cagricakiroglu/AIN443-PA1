import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from PIL import Image
import os
import pandas as pd

# Function to load images from a directory and convert them to grayscale
def load_images(image_directory):
    image_paths = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith('.bmp')]
    images = []
    for file_path in image_paths:
        img = Image.open(file_path).convert('L')  # Convert image to grayscale
        images.append(np.array(img).flatten())  # Flatten the image to a 1D array
    return images, image_paths

# Define the path to the images
image_directory = 'src/Dataset_v1/Dataset1' 

# Load the images and image paths
images, image_paths = load_images(image_directory)

if not images:
    print("No images loaded. Check the file paths and formats.")
    exit()

# Convert the list of images to a matrix
M = np.array(images).T  # Shape will be (65536, number_of_images)

# Normalize the dataset by subtracting the mean
mean_vector = M.mean(axis=1, keepdims=True)
D = M - mean_vector

# Calculate the covariance matrix
Cov = np.dot(D.T, D) / (D.shape[1] - 1)

# Calculate eigenvalues and eigenvectors
values, vectors = eig(Cov)

# Sort the eigenvectors based on eigenvalues in descending order
sorted_indices = np.argsort(values)[::-1]
S = vectors[:, sorted_indices[:3]]  # Select the first 3 eigenvectors

# Create an intermediate representation I by multiplying D with S (D @ S)
I = np.dot(D, S)

# Project all images to the lower-dimensional space (I^T @ D)
P = np.dot(I.T, D)

# Extract the filenames from the full paths for the DataFrame
file_names = [os.path.basename(path) for path in image_paths]

# Create a DataFrame from the PCA values with filenames
df_pca = pd.DataFrame(P.T, columns=['PCA1', 'PCA2', 'PCA3'], index=file_names)

# Plotting the results in a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[0, :], P[1, :], P[2, :])
ax.set_title('PCA Projection to 3D')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

# Save the plot as a JPEG file
plt.savefig('pca_projection.jpeg', format='jpeg')
plt.close()

# Save the DataFrame as a JPEG file
fig, ax = plt.subplots(figsize=(12, 4))  # Adjust the size as needed
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df_pca.values, colLabels=df_pca.columns, rowLabels=df_pca.index, loc='center')
plt.savefig('df_pca.jpeg', format='jpeg', dpi=300)
plt.close()
