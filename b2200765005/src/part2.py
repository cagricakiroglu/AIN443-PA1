import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt

print("a")
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2, axis=1))

def custom_histogram(data, bins, range):
    hist = np.zeros(bins)
    min_val, max_val = range
    bin_width = (max_val - min_val) / bins

    for value in data.flatten():
        if min_val <= value < max_val:
            index = int((value - min_val) // bin_width)
            hist[index] += 1
    return hist



# Function to load images from a directory and convert them to a suitable format
def load_images_from_folders_corrected(base_directory, target_size=(256, 256)):
    image_classes = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]
    if 'QUERY_IMAGES' in image_classes:
        image_classes.remove('QUERY_IMAGES')  # Exclude the 'QUERY_IMAGES' folder if it exists
    dataset_images = {}
    for image_class in image_classes:
        image_paths = glob.glob(os.path.join(base_directory, image_class, '*.jpg'))  # Assuming JPG images
        images = [np.array(Image.open(file_path).convert('RGB').resize(target_size)) for file_path in image_paths]
        dataset_images[image_class] = images
    return dataset_images

def load_query_images_corrected(query_directory, target_size=(256, 256)):
    image_paths = glob.glob(os.path.join(query_directory, '*.jpg'))  # Assuming JPG images
    return [np.array(Image.open(file_path).convert('RGB').resize(target_size)) for file_path in image_paths]

# Function to compute color histograms for a list of images
def compute_color_histograms_corrected(image_list, bins=256):
    histograms = []
    for img in image_list:
        hist_red = custom_histogram(img[:, :, 0], bins, (0, 256))
        hist_green = custom_histogram(img[:, :, 1], bins, (0, 256))
        hist_blue = custom_histogram(img[:, :, 2], bins, (0, 256))
        histograms.append(np.concatenate([hist_red, hist_green, hist_blue]))
    return np.array(histograms)


# Function to retrieve images based on the nearest features
def retrieve_images(query_features, dataset_features, top_n=10):
    distances = np.array([euclidean_distance(query_feature, dataset_features) for query_feature in query_features])
    indices = np.argsort(distances, axis=1)[:, :top_n]
    return indices
# Function to show the 10 most similar images for a query image
def show_similar_images_corrected(indices, dataset_images_flat, image_size=(256, 256)):
    fig, axes = plt.subplots(1, len(indices), figsize=(20, 3))
    for ax, idx in zip(axes, indices):
        img = dataset_images_flat[idx].reshape(image_size[0], image_size[1], 3)
        ax.imshow(img.astype(np.uint8))
        ax.axis('off')
    plt.show()

def average_precision(retrieved, relevant):
    hits = 0
    sum_precisions = 0
    for i, img in enumerate(retrieved):
        if img in relevant:
            hits += 1
            precision = hits / (i + 1)
            sum_precisions += precision
    return sum_precisions / hits if hits > 0 else 0

# Function to calculate the Mean Average Precision for a specific class
def compute_MAP_for_class(target_class, ranked_results):
    AP_values = []
    for query, ranked_list in ranked_results.items():
        relevant = set(img_name for class_name, img_name, _ in ranked_list if class_name == target_class)
        retrieved = [img_name for _, img_name, _ in ranked_list]
        AP = average_precision(retrieved, relevant)
        AP_values.append(AP)
    return sum(AP_values) / len(AP_values) if AP_values else 0

# Main execution logic for image retrieval
base_directory = 'src/Dataset_v1/Dataset2'  # Update this path as needed
dataset_images = load_images_from_folders_corrected(base_directory)
query_directory = "src/Dataset_v1/Dataset2/QUERY_IMAGES"
query_images = load_query_images_corrected(query_directory)

color_histograms = []
all_images_flat = []
image_class_names = []  # Store the class names of each image
for image_class, images in dataset_images.items():
    histograms = compute_color_histograms_corrected(images)
    color_histograms.append(histograms)
    all_images_flat.extend([img.flatten() for img in images])
    image_class_names.extend([image_class] * len(images))
color_histograms = np.vstack(color_histograms)
all_images_flat = np.array(all_images_flat)

# Dictionary to store the ranked results for mAP calculation
ranked_results = {}

# Perform image retrieval for each query image
for i, query_img in enumerate(query_images):
    query_hist = compute_color_histograms_corrected([query_img])[0]
    retrieved_indices = retrieve_images([query_hist], color_histograms)

    # Adjusted to handle the 2D nature of retrieved_indices
    for query_indices in retrieved_indices:
        ranked_list = [(image_class_names[idx], 'Image' + str(idx), 0) for idx in query_indices]
        ranked_results['Query' + str(i)] = ranked_list
        show_similar_images_corrected(query_indices, all_images_flat)


# Compute mAP for each class
class_names = set(image_class_names)
MAP_values = {class_name: compute_MAP_for_class(class_name, ranked_results) for class_name in class_names}

# Print MAP values for each class
MAP_values = {class_name: map_value for class_name, map_value in MAP_values.items()}

# Prepare data for the MAP table
map_data = [[class_name, f"{map_value:.3f}"] for class_name, map_value in MAP_values.items()]

# Sort data by MAP value for better visualization
map_data.sort(key=lambda x: x[1], reverse=True)

# Create a figure for the table
plt.figure(figsize=(10, 4))  # Adjust size as needed
ax = plt.gca()
ax.axis('off')

# Create the table and add it to the plot
table = plt.table(cellText=map_data, colLabels=['Class', 'Mean Average Precision'], loc='center', cellLoc='center')

# Adjust table style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # You can adjust these parameters to fit your data

# Save the table as a JPEG file
plt.savefig('map_table.jpeg', format='jpeg', dpi=300)
plt.close()

classes = list(MAP_values.keys())
map_scores = [MAP_values[cls] for cls in classes]

# Creating the bar plot
plt.figure(figsize=(10, 6))
plt.bar(classes, map_scores, color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Mean Average Precision')
plt.title('Mean Average Precision for Each Class')
plt.xticks(rotation=45)  # Rotate class names for better readability
plt.savefig('map_bar_chart.jpeg', format='jpeg')
plt.close()
