import cv2
import os
from matplotlib import pyplot as plt
from scipy.spatial import distance


# Loading images function with calling color extraction features function
def load_images(dataset_path):
    images = []
    features = []
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        color_feature_extraction = color_extraction_features(image_path)
        features.append(color_feature_extraction)
        images.append(image_path)
    return images, features

# color extraction features function
def color_extraction_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# calculate the distance between all images in DataSet images and the query image
def compute_distances(query_features, dataset_features):
    return [distance.euclidean(query_features, feature) for feature in dataset_features]

# rank the distances
def rank_images(distances, images):
    return sorted(zip(distances, images))


if __name__ == "__main__":

    dataset_path = ".././Images"

    # loading images and extracting features
    images, features = load_images(dataset_path)

    # select the image query and extracting its features
    query_image_path = ".././Images/113.jpg"
    query_features = color_extraction_features(query_image_path)

    # computing distances
    distances = compute_distances(query_features, features)

    # rank the images according to distance
    ranked_images = rank_images(distances, images)

    # plot the most 10 relevant
    for i, (dist, img_path) in enumerate(ranked_images[:10]):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Rank: {i + 1}\n With Distance: {dist:.2f}")  # Include distance in the title
        plt.axis('off')
        plt.show()