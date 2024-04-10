import cv2
import numpy as np
import os
import random
import time
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve

# Function to load all images from a given dataset path.
def load_images(dataset_path):
    images = []
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        images.append(image_path)
    return images


# Function to calculate the color histogram of an image according to pins.
def color_histogram_features(image_path, pin):
    # define 3 channels for each color
    channels = range(3)
    pin_range = [0, 256]

    # get the image from the path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # computes a histogram for each color channel (RGB)
    color_histograms = []
    for channel in channels:
        histogram = cv2.calcHist([image], [channel], None, [pin], pin_range)
        color_histograms.append(histogram)

    # normalizes the color_histograms, and concatenates into a single histogram.
    color_histograms = np.concatenate([cv2.normalize(h, h).flatten() for h in color_histograms])
    return color_histograms


# compute the Euclidean distance between two histograms.
def euclidean_distance(color_histogram_1, color_histogram_2):
    return np.sqrt(np.sum((color_histogram_1 - color_histogram_2) ** 2))


# get the category of an image based on its filename.
def get_image_category(image_path):
    # The category is determined by the image number since images are named each 100 to the same category.
    image_number = int(os.path.basename(image_path).split('.')[0])
    return image_number // 100

# compute the precision, recall, F1 score, tpr, and fpr.
def calculate_metrics_at_thresholds(sorted_distances, query_category, thresholds):
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for threshold in thresholds:

        TP = FP = FN = TN = 0

        # True Positives (TP): Images correctly retrieved as relevant (distance <= threshold, correct category)
        # False Positives (FP): Images incorrectly retrieved as relevant (distance <= threshold, incorrect category)
        # False Negatives (FN): Relevant images not retrieved (distance > threshold, correct category)
        # True Negatives (TN): Irrelevant images correctly identified as irrelevant (distance > threshold, incorrect category)

        for distance, image_path in sorted_distances:
            if distance <= threshold:
                if get_image_category(image_path) == query_category:
                    TP += 1
                else:
                    FP += 1
            else:
                if get_image_category(image_path) == query_category:
                    FN += 1
                else:
                    TN += 1

        TPR = TP / (TP + FN) if TP + FN > 0 else 0
        FPR = FP / (FP + TN) if FP + TN > 0 else 0
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TPR
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)

    return tpr_list, fpr_list, precision_list, recall_list, f1_list

# Function to plot the ROC curve using Matplotlib.
def plot_roc_curve(tpr, fpr, roc_auc, pin):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {pin} pins')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    DataSet_path = ".././Images"
    images = load_images(DataSet_path)
    query_images = random.sample(images, 10)  # Randomly select 10 query images from DataSet

    pins = [120, 180, 360]

    for pin in pins:
        start_time = time.time()

        # calculate the color_histogram features for all dataset images
        dataset_features = []
        for image in images:
            color_histogram = color_histogram_features(image, pin)
            dataset_features.append((image, color_histogram))

        # define arrays
        all_tpr = []
        all_fpr = []
        all_precision = []
        all_recall = []
        all_f1 = []
        true_Positive_Rate_interpolated = []


        false_Positive_Rate = np.linspace(0, 1, 101)

        # calculating the maximum Euclidean distance between any pair of color histograms in the dataset.
        max_distance = 0
        for _, color_histogram_1 in dataset_features:
            for _, color_histogram_2 in dataset_features:
                distance = euclidean_distance(color_histogram_1, color_histogram_2)
                if distance > max_distance:
                    max_distance = distance

        # Compute precision, recall, and F1 score for each query image
        for query_image in query_images:
            query_hist = color_histogram_features(query_image, pin)

            # calculate the distance between qurry image and dataset images
            distances = []
            for image, hist in dataset_features:
                distance = euclidean_distance(query_hist, hist)
                distances.append((distance, image))

            # sort the distances
            sorted_distances = sorted(distances, key=lambda x: x[0])


            # define a threshold
            thresholds = np.linspace(0, max_distance, 20)
            tpr_list, fpr_list, precision, recall, f1 = calculate_metrics_at_thresholds(sorted_distances, get_image_category(query_image), thresholds)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

            # For ROC calculation
            y_true = []
            y_scores = []

            for dist, img in sorted_distances:
                true_label = get_image_category(query_image) == get_image_category(img)
                y_true.append(true_label)
                y_scores.append(dist)

            # Convert distances to similarity scores
            y_scores = []
            for dist, img in sorted_distances:
                adjusted_score = max_distance - dist
                y_scores.append(adjusted_score)

            fpr, tpr, _ = roc_curve(y_true, y_scores)

            # Interpolate the TPR values to the base FPR scale
            true_Positive_Rate_interpolated.append(interp(false_Positive_Rate, fpr, tpr))

            # Ensuring that the curve starts at 0
            true_Positive_Rate_interpolated[-1][0] = 0.0

        # Calculate the mean of the interpolated TPR values
        mean_tpr = np.mean(true_Positive_Rate_interpolated, axis=0)
        mean_tpr[-1] = 1.0  # Ensuring that the curve ends at 1

        roc_auc = auc(false_Positive_Rate, mean_tpr)

        # Plot ROC curve
        plot_roc_curve(mean_tpr, false_Positive_Rate, roc_auc, pin)

        end_time = time.time()

        print(f"Time for {pin} pins: {end_time - start_time} seconds")
        print(f"Average Precision for {pin} pins: {np.mean(all_precision)}")
        print(f"Average Recall for {pin} pins: {np.mean(all_recall)}")
        print(f"Average F1 Score for {pin} pins: {np.mean(all_f1)}")
        print("-----------------------------------------------------")