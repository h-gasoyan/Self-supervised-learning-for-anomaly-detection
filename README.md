# Self-supervised-learning-for-anomaly-detection

## Overview
This repository contains the code and resources for anomaly detection using self-supervised learning methods, specifically DINO's self-distillation and MAE's masked reconstruction.

## Abstract
This project enhances anomaly detection by leveraging self-supervised learning techniques, specifically DINO's self-distillation and MAE's masked reconstruction. DINO facilitates robust feature representation learning without the need for labeled data, while MAE reconstructs masked image parts to identify anomalies through reconstruction errors. K-means clustering was applied to understand how anomalous clusters differ from non-anomalous ones, assessing the separability and compactness of the clusters. Additionally, Gaussian Mixture Models (GMM) were utilized to model the distribution of normal data and identify outliers, comparing the results before and after implementing self-supervised learning techniques.

## Methodology

#### Distillation with No Labels (DINO)
Distillation with No Labels (DINO)  utilizes the Vision Transformer (ViT) architecture, capitalizing on the benefits of self-supervised learning without the need for labeled datasets. The essence of DINO lies in its student-teacher architecture, where both models are initialized with the same transformer structure

#### Masked Autoencoders (MAE)
The MAE method offers a transformer-based architecture specifically designed to capitalize on self-supervised learning principles. It employs an asymmetric encoder-decoder architecture to learn robust visual representations by reconstructing masked portions of input images.

## Experiments
In our experiments, we constructed a multi-modal anomaly detection dataset using CIFAR-10, COCO 2017, and MVTec. The multi-modal anomaly detection method is that in each iteration, one class was designated as anomalous while the remaining classes were considered non-anomalous. This iterative process simulated an anomaly detection scenario in which the model must identify a single anomalous class among various normal classes. The final performance score for each dataset was calculated as the mean score across all iterations. High-dimensional feature representations were extracted from these images using DINO and MAE models.

Feature extraction was performed using DINO and MAE models. Images were resized to 224x224 pixels and processed through the models to obtain high-dimensional feature representations. Dimensionality reduction and visualization techniques were then applied to understand the structure of the extracted features and the distribution of normal and anomalous data. PCA (Principal Component Analysis) reduced the high-dimensional features to 100 dimensions to retain the most significant components of the data. This was followed by t-SNE (t-distributed Stochastic Neighbor Embedding) to further reduce the data to 2D for visualization, allowing observation of the clustering behavior of normal and anomalous data points.

K-means clustering was applied to the PCA-reduced features to evaluate the effectiveness of the learned representations in separating normal and anomalous data. The data was partitioned into clusters, with the number of clusters determined using the elbow method. The clustering results provided insights into the compactness and separability of the normal and anomalous clusters.

To identify and remove outliers, Gaussian Mixture Models (GMM) were used. GMM modeled the distribution of normal data to identify outliers, which were then removed. K-means clustering was reapplied to the cleaned dataset to enhance the overall clustering performance.

The ROC AUC score was used to evaluate the effectiveness of the learned representations and the clustering results. This score measures the model's ability to distinguish between normal and anomalous data points.

### ROC Curve Results for K-Means Clustering
![K-Means](https://drive.google.com/uc?export=view&id=1TjUb2s3LJuKpM_vB9HPVaIE7qkFPBiXV)

### ROC Curve Results after GMM Removal and K-Means Clustering
![image](https://github.com/h-gasoyan/Self-supervised-learning-for-anomaly-detection/assets/72386745/87040f7c-273b-4ed0-a589-05a236608043)

### AUC Score Results
![auc](https://drive.google.com/uc?export=view&id=1CWfjMvjRNgPpi_YVvHeGkugwJ82UUhfh)

### Conclusion
The results of our analysis demonstrated that even though both the DINO and MAE models show good performance in downstream tasks, their effectiveness in anomaly detection was poor.  Specifically, while the DINO model's representations were generally better than those of the MAE model, the overall results indicated that the learned representations that are beneficial for downstream tasks do not translate well to anomaly detection. The use of GMM for outlier detection and reapplication of K-means clustering did improve the performance to some extent, but the anomaly detection outcomes were still suboptimal.

For a comprehensive overview of all results, including charts and detailed analysis, please refer to the https://drive.google.com/file/d/1SqFLAAPMlqOrHEZmWF8T6iwGB--FKC-C/view?usp=sharing
