import cv2
import numpy as np
import os
import random
import time
from matplotlib import pyplot as plt
from scipy.stats import skew
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve

# Function to load images
def load_images(dataset_path):
    images = []
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        images.append(image_path)
    return images

# Function to calculate color moments (mean, standard deviation, and skewness)
def color_moments(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    moments = []
    for channel in range(3):
        channel_data = image[:, :, channel].flatten()
        moments.extend([np.mean(channel_data), np.std(channel_data), skew(channel_data)])
    return np.array(moments)

# Euclidean distance function
def euclidean_distance(moment1, moment2):
    return np.sqrt(np.sum((moment1 - moment2) ** 2))

# Function to get image category
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

# Function to plot ROC curve
def plot_roc_curve(tpr, fpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":

    DataSet_path = ".././Images"
    images = load_images(DataSet_path)
    query_images = random.sample(images, 10)

    start_time = time.time()

    dataset_moments = [(image, color_moments(image)) for image in images]

    # Find the maximum distance between the dataset images
    max_distance = 0
    for _, moment1 in dataset_moments:
        for _, moment2 in dataset_moments:
            distance = euclidean_distance(moment1, moment2)
            max_distance = max(max_distance, distance)

    # Initialize arrays for metrics
    all_precision = []
    all_recall = []
    all_f1 = []
    true_Positive_Rate_interpolated = []
    false_Positive_Rate = np.linspace(0, 1, 101)

    # Compute metrics for each query image
    for query_image in query_images:
        query_moment = color_moments(query_image)

        # calculate the distance between dataset images and query image
        distances = [(euclidean_distance(query_moment, moment), image) for image, moment in dataset_moments]

        # sort the distances
        sorted_distances = sorted(distances, key=lambda x: x[0])

        y_true = [get_image_category(query_image) == get_image_category(img) for _, img in sorted_distances]
        y_scores = [max_distance - dist for dist, _ in sorted_distances]

        # define a threshold to calculate the precision, recall, and f1 in different values
        thresholds = np.linspace(0, max_distance, 20)
        tpr_list, fpr_list, precision, recall, f1 = calculate_metrics_at_thresholds(sorted_distances, get_image_category(query_image), thresholds)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        true_Positive_Rate_interpolated.append(np.interp(false_Positive_Rate, fpr, tpr))
        true_Positive_Rate_interpolated[-1][0] = 0.0

    mean_tpr = np.mean(true_Positive_Rate_interpolated, axis=0)
    mean_tpr[-1] = 1.0
    roc_auc = auc(false_Positive_Rate, mean_tpr)

    plot_roc_curve(mean_tpr, false_Positive_Rate, roc_auc)

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time} seconds")
    print(f"Average Precision: {np.mean(all_precision)}")
    print(f"Average Recall: {np.mean(all_recall)}")
    print(f"Average F1 Score: {np.mean(all_f1)}")
    print(f"AUC: {roc_auc}")
