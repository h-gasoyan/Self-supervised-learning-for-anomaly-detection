import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.utils import shuffle
import os
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def plot_mvtec_features_with_labels(features, labels):
    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1)

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Reduce dimensions with PCA to 100
    pca = PCA(n_components=100)
    features_reduced = pca.fit_transform(features_scaled)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_embedded = tsne.fit_transform(features_reduced)

    # Plotting results with different markers
    plt.figure(figsize=(10, 8))

    # Create mask for labels
    anomalous_mask = labels == 1
    non_anomalous_mask = ~anomalous_mask  # Inverse of anomalous

    # Scatter plots for each class
    plt.scatter(features_embedded[anomalous_mask, 0], features_embedded[anomalous_mask, 1],
                marker='x', color='red', label='Anomalous')
    plt.scatter(features_embedded[non_anomalous_mask, 0], features_embedded[non_anomalous_mask, 1],
                marker='o', color='black', label='Non-Anomalous')

    plt.title('t-SNE Visualization of MAE Reduced Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.show()

def plot_euclidean_distances(euclidian_distance_test, class_names):
    cmap = plt.cm.get_cmap('viridis', len(class_names))

    for column in euclidian_distance_test.columns:
        if column.isdigit():
            class_index = int(column)
            color = cmap(class_index)

            anomaly_class_names = euclidian_distance_test['Anomaly Class'].map(class_names)
            plt.scatter(anomaly_class_names, euclidian_distance_test[column], s=32, alpha=0.8, color=[color])

            plt.xlabel('Anomaly Class')
            plt.ylabel(f'{class_names[class_index]}')
            plt.title(f'Mean Distance of All Data Points to {class_names[class_index]} Centroid')

            plt.gca().spines[['top', 'right']].set_visible(False)
            plt.show()



def visualize_tsne(train_features, train_labels, selected_classes, class_names, n_components=100, tsne_components=2, perplexity=25):

    # Filter the data to include only the specified classes
    mask = np.isin(train_labels, selected_classes)
    filtered_features = train_features[mask]
    filtered_labels = train_labels[mask]

    # Map the filtered labels to the range [0, len(class_names) - 1]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_classes)}
    mapped_labels = np.array([label_mapping[label] for label in filtered_labels])

    # Reshape and apply PCA
    original_shape = filtered_features.shape
    reshaped_features = filtered_features.reshape(original_shape[0], -1)
    pca = PCA(n_components=n_components)
    pca_reduced_features = pca.fit_transform(reshaped_features)

    # Apply t-SNE
    tsne = TSNE(n_components=tsne_components, random_state=42, perplexity=perplexity)
    tsne_results = tsne.fit_transform(pca_reduced_features)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5, c=mapped_labels, cmap='viridis')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of {}-Dimensional Features'.format(n_components))
    plt.grid(True)

    # Add a color bar with class names
    colorbar = plt.colorbar(scatter, ticks=range(len(class_names)))
    colorbar.set_ticklabels(class_names)

    plt.show()


def plot_roc_for_anomalous_classes(cleaned_data, cleaned_labels, specific_classes, non_anomalous_class, class_names, n_clusters):
    """
    Plot ROC curves for multiple anomalous classes against a single non-anomalous class.

    Parameters:
    - cleaned_data: numpy array of shape (n_samples, n_features), the data to use for clustering
    - cleaned_labels: numpy array of shape (n_samples,), the labels for the data
    - specific_classes: list of integers, the classes to consider as anomalous
    - non_anomalous_class: integer, the class to consider as non-anomalous
    - class_names: dictionary, mapping from class number to class name
    """
    roc_auc_scores = []
    all_fpr = []
    all_tpr = []
    base_fpr = np.linspace(0, 1, 201)

    for i in specific_classes:
        # Create binary labels for the current anomalous and the non-anomalous class
        binary_labels = np.where(cleaned_labels == i, 1,
                                 np.where(cleaned_labels == non_anomalous_class, 0, -1))

        # Filter data and labels to keep only the relevant classes
        relevant_mask = binary_labels != -1
        relevant_labels = binary_labels[relevant_mask]
        relevant_data = cleaned_data[relevant_mask]

        if len(np.unique(relevant_labels)) < 2:
            print(
                f"Skipping class {i} ({class_names[i]}) vs class {non_anomalous_class} (automobile) due to lack of both positive and negative samples.")
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(relevant_data)
        distances = kmeans.transform(relevant_data)
        scores = -np.min(distances, axis=1)

        roc_auc = roc_auc_score(relevant_labels, scores)
        fpr, tpr, _ = roc_curve(relevant_labels, scores)
        roc_auc_scores.append(roc_auc)

        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0
        all_tpr.append(tpr_interp)

        plt.plot(fpr, tpr, alpha=0.3, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

    mean_tpr = np.mean(all_tpr, axis=0)
    mean_tpr[-1] = 1
    mean_auc = np.mean(roc_auc_scores)  # Calculate mean AUC from collected scores

    plt.plot(base_fpr, mean_tpr, 'b', linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for CIFAR-10 (DINO) Anomalous Classes')
    plt.legend(loc='lower right')
    plt.show()

def calculate_anomaly_distances(features, labels):
    # Calculate mean for each class
    class_means = []
    for i in range(10):  # Assuming 10 classes
        class_features = features[labels == i]
        class_mean = np.mean(class_features, axis=0)
        class_means.append(class_mean)

    distance_dict = {}

    for index, feature in enumerate(features):
        distances = {}

        for anomaly_class, mean in enumerate(class_means):
            distance = np.sqrt(np.sum((feature - mean) ** 2))
            distances[anomaly_class] = distance
        # Add distances and class label to the dictionary
        distance_dict[index] = {'Anomaly Class': labels[index], **distances}

    # Create a DataFrame from the dictionary
    distance_df = pd.DataFrame.from_dict(distance_dict, orient='index')

    # Convert 'Anomaly Class' to categorical data type with correct order
    distance_df['Anomaly Class'] = pd.Categorical(distance_df['Anomaly Class'], categories=np.arange(10))

    # Sort the DataFrame based on the 'Anomaly Class' column
    distance_df = distance_df.sort_values(by='Anomaly Class')

    return distance_df


def detect_outliers_and_plot_roc(reduced_data, mvtec_labels, classes_to_remove, cov_type, n_gmm_components, outlier_percentile=6, kmeans_clusters=2):
    """
    Detect outliers using Gaussian Mixture Model and plot ROC curves for different anomalous classes.

    Parameters:
    - reduced_data: numpy array of shape (n_samples, n_features), the data to use for clustering
    - mvtec_labels: numpy array of shape (n_samples,), the labels for the data
    - classes_to_remove: list of strings, the classes to exclude from analysis
    - n_gmm_components: integer, number of components for GMM (default: 4)
    - outlier_percentile: float, percentile threshold for outlier detection (default: 6)
    - kmeans_clusters: integer, number of clusters for KMeans (default: 2)
    """
    gmm = GaussianMixture(n_components=n_gmm_components, covariance_type=cov_type)
    gmm.fit(reduced_data)

    # Calculate the log probabilities
    log_prob = gmm.score_samples(reduced_data)

    # Determine a threshold for outliers
    threshold = np.percentile(log_prob, outlier_percentile)
    outliers = log_prob < threshold

    # Print the number of outliers detected
    print("Number of outliers detected:", np.sum(outliers))
    cleaned_data = reduced_data[~outliers]
    cleaned_labels = mvtec_labels[~outliers]
    unique_classes = np.unique(cleaned_labels)

    # Separate labels into anomalous and non-anomalous
    anomalous_classes = [label for label in unique_classes if label.startswith('anomalous_')]
    non_anomalous_classes = [label for label in unique_classes if label.startswith('non_anomalous')]

    # Filter out the classes to be removed
    anomalous_classes = [cls for cls in anomalous_classes if cls not in classes_to_remove]

    results = {'Class': [], 'ROC_AUC': []}
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    for anomalous_class in anomalous_classes:
        # Create binary labels: 1 for the current anomalous class, 0 for all non-anomalous classes
        binary_labels = np.array([1 if label == anomalous_class else 0 for label in cleaned_labels])

        # Ensure there are enough samples
        if np.sum(binary_labels) < 5 or np.sum(1 - binary_labels) < 5:
            print(f"Skipping class {anomalous_class} due to insufficient samples of both classes.")
            continue

        # Apply KMeans
        kmeans = KMeans(n_clusters=kmeans_clusters)
        kmeans.fit(cleaned_data)

        # Calculate distances to the nearest cluster center
        distances = np.min(cdist(cleaned_data, kmeans.cluster_centers_, 'euclidean'), axis=1)
        scores = -distances  # Use negative distances as anomaly scores

        try:
            # Compute ROC AUC
            roc_auc = roc_auc_score(binary_labels, scores)
            fpr, tpr, _ = roc_curve(binary_labels, scores)
            results['Class'].append(anomalous_class)
            results['ROC_AUC'].append(roc_auc)

            # Plot ROC curve
            plt.plot(fpr, tpr, label=f'Class {anomalous_class} (AUC = {roc_auc:.2f})')

            # Interpolate TPR
            tpr_interp = np.interp(base_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
        except ValueError as e:
            print(f"Error processing class {anomalous_class}: {e}")

    # Convert results to a DataFrame for better readability
    results_df = pd.DataFrame(results)
    print(results_df)

    # Plot the mean ROC curve if there are any valid TPRs
    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(base_fpr, mean_tpr)
        plt.plot(base_fpr, mean_tpr, 'k--', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for MVTec (DINO) Anomalous')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()