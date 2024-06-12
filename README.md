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

![Example Image](https://drive.google.com/uc?export=view&id=1TjUb2s3LJuKpM_vB9HPVaIE7qkFPBiXV)
