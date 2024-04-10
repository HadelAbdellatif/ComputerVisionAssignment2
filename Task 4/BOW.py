import cv2
import numpy as np
import os
import random
import time
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve
from matplotlib import pyplot as plt

# Function to load images
def load_images(dataset_path):
    images = []
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        images.append(image_path)
    return images


# Function to extract SIFT features
def extract_SIFT_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read the image at {image_path}")
    sift = cv2.SIFT_create()
    _, features = sift.detectAndCompute(image, None)
    return features

# Function to create BOW using K-Means
def create_BOW(features, k_clusters):
    # Create a K-Means clustering model with the specified number of clusters (k_clusters)
    k_means = KMeans(n_clusters=k_clusters, n_init='auto')
    k_means.fit(features)
    return k_means


# Function to quantize features
def feature_quantization(features, k_means):
    # Use the learned K-Means model to predict cluster labels for input features.
    labels = k_means.predict(features)
    histogram, _ = np.histogram(labels, bins=range(k_means.n_clusters + 1))
    return histogram

# Function to get image category
def get_image_category(image_path):
    # The category is determined by the image number since images are named each 100 to the same category.
    image_number = int(os.path.basename(image_path).split('.')[0])
    return image_number // 100

# Function to compute metrics
def compute_metrics(sorted_distances, query_image, similarity_threshold=0.5):
    query_category = get_image_category(query_image)

    retrieved_images = []
    for distance, image in sorted_distances[:100]:
        retrieved_images.append(image)

    y_true = []
    for image in retrieved_images:
        y_true.append(get_image_category(image) == query_category)

    y_pred = []
    for dist, _ in sorted_distances[:100]:
        y_pred.append(dist <= similarity_threshold)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1


# Function to plot BOW histogram
def plot_BOW_histogram(bow_features, k_clusters):
    plt.bar(range(k_clusters), bow_features)
    plt.xlabel('Cluster Index')
    plt.ylabel('Frequency')
    plt.title('Bag of Words (BOW) Histogram')
    plt.show()

if __name__ == "__main__":
    DataSet_path = ".././Images"
    images = load_images(DataSet_path)

    # Ensure there are enough images for querying
    if len(images) < 10:
        raise ValueError("There are not enough pictures for a query.")

    query_images = random.sample(images, 10)

    # Choose a sample image for BOW representation plot
    sample_image = random.choice(images)

    start_time = time.time()

    # Extract SIFT features and handle potential errors
    dataset_features = []
    for image in images:
        features = extract_SIFT_features(image)
        dataset_features.append((image, features))

    # Create BOW using K-Means
    all_features = np.vstack([features for _, features in dataset_features])

    # Use the Elbow Method to determine the optimal number of clusters (k_clusters)
    distortions = []
    K_range = range(10, 20, 2)  # Adjust the range based on your dataset
    for k in K_range:
        k_means = KMeans(n_clusters=k)
        k_means.fit(all_features)
        distortions.append(k_means.inertia_)

    # Plot Elbow Method
    plt.plot(K_range, distortions, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Task 4 Elbow Method for Optimal k')
    plt.show()

    # Choose the optimal k_clusters based on the plot
    k_clusters = 20
    k_means = create_BOW(all_features, k_clusters)

    # Extract SIFT features and quantize
    sample_features = extract_SIFT_features(sample_image)
    sample_bow_features = feature_quantization(sample_features, k_means)

    # Plot BOW histogram for the sample image
    plot_BOW_histogram(sample_bow_features, k_clusters)

    # Quantize features
    dataset_bow_features = [(image, feature_quantization(features, k_means)) for image, features in dataset_features]

    all_precision = []
    all_recall = []
    all_f1 = []

    # Compute metrics for each query image
    for query_image in query_images:
        query_features = extract_SIFT_features(query_image)
        # 1. quantize features
        query_bow_features = feature_quantization(query_features, k_means)
        # 2. compute the distances
        distances = [(np.linalg.norm(query_bow_features - bow_features), image) for image, bow_features in dataset_bow_features]
        # 3. sort the distances
        sorted_distances = sorted(distances, key=lambda x: x[0])

        precision, recall, f1 = compute_metrics(sorted_distances, query_image)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time} seconds")
    print(f"Average Precision: {np.mean(all_precision)}")
    print(f"Average Recall: {np.mean(all_recall)}")
    print(f"Average F1 Score: {np.mean(all_f1)}")
