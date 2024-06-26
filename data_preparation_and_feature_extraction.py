from google.colab import drive
drive.mount('/content/drive')

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
import timm
from sklearn.utils import shuffle
from zipfile import ZipFile, BadZipFile
import os
from pycocotools.coco import COCO
import pickle
from tqdm import tqdm
from PIL import Image
import json
import zipfile
import glob
from PIL import Image

"""### Load pretrained DINO  model

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load pretrained weights
model_dino=timm.create_model('vit_small_patch16_224.dino', pretrained=True,
                             num_classes=0 # remove classifier nn.Linear
                             )
model_dino=model_dino.to(device)
model_dino = model_dino.eval()

# get model specific transforms (normalization, resize)
data_config_dino = timm.data.resolve_model_data_config(model_dino)
transform_dino = timm.data.create_transform(**data_config_dino, is_training=False)

transform_dino

"""### Load pretrained MAE model"""

#load pretrained weights
model_mae = timm.create_model('vit_huge_patch14_224.mae', pretrained=True, num_classes=0)
model_mae=model_mae.to(device)
model_mae = model_mae.eval()

data_config = timm.data.resolve_model_data_config(model_mae)
transform_mae = timm.data.create_transform(**data_config, is_training=False)

transform_mae

"""# CIFAR Data preparation"""

cifar_train = CIFAR10(root="/content/drive/MyDrive/Self Supervised Anomaly Detection", train=True, download=True)
cifar_test = CIFAR10(root="/content/drive/MyDrive/Self Supervised Anomaly Detection", train=False, download=True)

#filter in each class test 100 images, train 1000 images
def filter_dataset(dataset, num_samples_per_class):
    filtered_data = []
    class_counts = {i: 0 for i in range(10)}
    for data, label in dataset:
        if class_counts[label] < num_samples_per_class:
            filtered_data.append((data, label))
            class_counts[label] += 1
        if all(count == num_samples_per_class for count in class_counts.values()):
            break
    return filtered_data

# Filter train and test datasets
cifar_filtered_train_data = filter_dataset(cifar_train, 1000)
cifar_filtered_test_data = filter_dataset(cifar_test, 100)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

cifar_train_dino = CustomDataset(cifar_filtered_train_data, transform=transform_dino)
cifar_test_dino = CustomDataset(cifar_filtered_test_data, transform=transform_dino)

cifar_train_mae = CustomDataset(cifar_filtered_train_data, transform=transform_mae)
cifar_test_mae = CustomDataset(cifar_filtered_test_data, transform=transform_mae)

#Dataloaders
cifar_train_loader_dino = torch.utils.data.DataLoader(cifar_train_dino, batch_size=64, shuffle=True)
cifar_test_loader_dino = torch.utils.data.DataLoader(cifar_test_dino, batch_size=64, shuffle=False)

cifar_train_loader_mae = torch.utils.data.DataLoader(cifar_train_mae, batch_size=64, shuffle=True)
cifar_test_loader_mae = torch.utils.data.DataLoader(cifar_test_mae, batch_size=64, shuffle=False)

#feature and label extraction
def extract_features(model, dataloader, device):
    features = []
    labels = []
    with torch.no_grad():
        for data_batch, label_batch in dataloader:
            data_batch = data_batch.to(device)
            output_batch = model.forward_features(data_batch)  # Forward pass
            features.append(output_batch.cpu().numpy())
            labels.append(label_batch.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

#feature and label extraction from dino and mae
train_features_cifar_dino, train_labels_cifar_dino = extract_features(model_dino,cifar_train_loader_dino,device)
train_features_cifar_mae, train_labels_cifar_mae = extract_features(model_mae,cifar_train_loader_mae,device)

# Save features and labels
np.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/cifar_train_features_dino.npy', train_features_cifar_dino)
np.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/cifar_train_labels_dino.npy', train_labels_cifar_dino)

np.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/cifar_train_features_mae.npy', train_features_cifar_mae)
np.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/cifar_train_labels_mae.npy', train_labels_cifar_mae)

"""# COCO Data preparation"""

#Unzip files
def unzip_file(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

zip_file_path = '/content/drive/MyDrive/Anomaly Detection Dataset/archive.zip'
extract_to = '/content/drive/MyDrive/Anomaly Detection Dataset'
unzip_file(zip_file_path, extract_to)

#count jpg number
def count_jpg_images(folder_path):
    jpg_files = glob.glob(folder_path + '/*.jpg')
    return len(jpg_files)

# Example usage
folder_path = '/content/drive/MyDrive/Anomaly Detection Dataset/filtered_images/train'
num_jpg_images = count_jpg_images(folder_path)
print(f"Number of JPG images in the folder: {num_jpg_images}")

"""### Filter COCO Dataset"""

def filter_annotations(annotations_file, included_subcategories, save_dir, filtered_images_dir):

    coco = COCO(annotations_file)

    filtered_annotations = {
        "categories": [],
        "images": [],
        "annotations": []  # Include annotations to store labels
    }

    # Get category IDs for included subcategories
    included_category_ids = set()
    for cat in coco.dataset['categories']:
        if cat['name'] in included_subcategories:
            filtered_annotations['categories'].append(cat)
            included_category_ids.add(cat['id'])

    # Filter images for included categories
    included_image_ids = set()
    for image in coco.dataset['images']:
        for annotation in coco.imgToAnns[image['id']]:
            if annotation['category_id'] in included_category_ids:
                included_image_ids.add(image['id'])
                # Include annotations to store labels
                filtered_annotations['annotations'].append(annotation)
                break

    # Filter images for included categories
    for image in coco.dataset['images']:
        if image['id'] in included_image_ids:
            filtered_annotations['images'].append(image)
            # Copy images to the filtered images directory
            image_filename = os.path.join(save_dir, image['file_name'])
            filtered_image_filename = os.path.join(filtered_images_dir, os.path.basename(image['file_name']))
            try:
                shutil.copy(image_filename, filtered_image_filename)
            except FileNotFoundError:
                print(f"File not found: {image_filename}. Skipping...")
                continue

    # Construct the new file name for the filtered annotations
    annotations_base_name = os.path.basename(annotations_file)
    new_annotations_base_name = annotations_base_name.replace('.json', '_filtered.json')
    new_filtered_annotations_file = os.path.join(save_dir, new_annotations_base_name)

    # Save filtered annotations to a new JSON file with a different name
    with open(new_filtered_annotations_file, 'w') as f:
        json.dump(filtered_annotations, f)

#Filter dataset and take 5 subcategories
annotations_file = "/content/drive/MyDrive/Anomaly Detection Dataset/coco2017/annotations/instances_val2017.json"
included_subcategories = ["bicycle", 'tv', 'banana', 'traffic light', 'pizza']
save_dir = "/content/drive/MyDrive/Anomaly Detection Dataset/coco2017/val2017"
filtered_images_dir = "/content/drive/MyDrive/Anomaly Detection Dataset/filtered_images/val"
filter_annotations(annotations_file, included_subcategories, save_dir, filtered_images_dir)


filtered_images_dir_train= "/content/drive/MyDrive/Anomaly Detection Dataset/filtered_images/train"
save_dir_train = "/content/drive/MyDrive/Anomaly Detection Dataset/coco2017/train2017"
annotations_train_file = "/content/drive/MyDrive/Anomaly Detection Dataset/coco2017/annotations/instances_train2017.json"
filter_annotations(annotations_train_file, included_subcategories, save_dir_train, filtered_images_dir_train)

class CustomDataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None):
        self.annotations_file = annotations_file
        self.image_dir = image_dir
        self.transform = transform
        self.data = self.load_annotations()

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        image_info = self.data['images'][idx]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        category_id = self.get_category_id(image_info['id'])  # Get category ID based on image ID
        return image, category_id

    def load_annotations(self):
        with open(self.annotations_file, 'r') as f:
            annotations = json.load(f)
        return annotations

    def get_category_id(self, image_id):
        # Find corresponding annotation for the given image ID
        for annotation in self.data['annotations']:
            if annotation['image_id'] == image_id:
                return annotation['category_id']
        return None

image_folder = "/content/drive/MyDrive/Anomaly Detection Dataset/filtered_images/val"
annotations_file = "/content/drive/MyDrive/Anomaly Detection Dataset/coco2017/val2017/instances_val2017_filtered.json"

image_folder_train = "/content/drive/MyDrive/Anomaly Detection Dataset/filtered_images/train"
annotations_file_train = "/content/drive/MyDrive/Anomaly Detection Dataset/coco2017/train2017/instances_train2017_filtered.json"

dataset_val = CustomDataset(annotations_file, image_folder, transform=transform_dino)
dataset_train = CustomDataset(annotations_file_train, image_folder_train, transform=transform_dino)

dataset_train_mae = CustomDataset(annotations_file_train, image_folder_train, transform_mae)
dataset_val_mae = CustomDataset(annotations_file, image_folder, transform=transform_mae)

#Dataloaders
loader_coco_test_dino=DataLoader(dataset_val,batch_size=64,shuffle=False)
loader_coco_train_dino=DataLoader(dataset_train,batch_size=64,shuffle=False)

loader_coco_test_mae=DataLoader(dataset_val_mae,batch_size=64,shuffle=True)
loader_coco_train_mae=DataLoader(dataset_train_mae,batch_size=64,shuffle=False)

#Feature and label extraction
def extract_features(model, dataloader, device):
    features = []
    labels = []
    with torch.no_grad():
        for data_batch, label_batch in dataloader:
            data_batch = data_batch.to(device)
            output_batch = model.forward_features(data_batch)  # Forward pass
            features.append(output_batch.cpu().numpy())
            labels.append(label_batch.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

train_features_coco_dino, train_labels_coco_dino = extract_features(model_dino,loader_coco_train_dino,device)
train_features_coco_mae, train_labels_coco_mae = extract_features(model_mae,loader_coco_train_mae,device)

no.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/coco_train_features_dino.npy', train_features_coco_dino)
np.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/coco_train_labels_dino.npy', train_labels_coco_dino)

np.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/coco_train_features_mae.npy', train_features_coco_mae)
np.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/coco_train_labels_mae.npy', train_labels_coco_mae)

"""#MvTec Dataset preparation"""

base_path = "/content/drive/MyDrive/Anomaly Detection Dataset/Multimodal protocol/MvTec"

# Create the 'non_anomalous' and 'anomalous' folders with category subfolders
non_anomalous_path = os.path.join(base_path, "/content/drive/MyDrive/Anomaly Detection Dataset/MVTec/non_anomalous")
anomalous_path = os.path.join(base_path, "/content/drive/MyDrive/Anomaly Detection Dataset/MVTec/anomalous")

# Ensure the main folders and category subfolders are created
for main_folder in [non_anomalous_path, anomalous_path]:
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    # Create subfolders for each category within 'non_anomalous' and 'anomalous'
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path) and category not in ["non_anomalous", "anomalous"]:
            subfolder_path = os.path.join(main_folder, category)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

# Iterate over the 15 category folders
for category in os.listdir(base_path):
    category_path = os.path.join(base_path, category)
    if os.path.isdir(category_path) and category not in ["non_anomal", "anomal"]:
        # Path to the 'good' images in the training set for non_anomalous
        good_path = os.path.join(category_path, "train", "good")
        if os.path.exists(good_path):
            # Copy and rename images from 'good' folder to 'non_anomalous/category'
            for image in os.listdir(good_path):
                source_image_path = os.path.join(good_path, image)
                # New image name with category as prefix
                new_image_name = f"{category}_{image}"
                destination_image_path = os.path.join(non_anomalous_path, category, new_image_name)
                # Copy the image
                shutil.copy(source_image_path, destination_image_path)

        # Path to the 'test' folder for anomalous
        test_path = os.path.join(category_path, "test")
        if os.path.exists(test_path):
            # Iterate through all subfolders in 'test' except 'good'
            for subfolder in os.listdir(test_path):
                if subfolder == "good":
                    continue  # Skip the 'good' folder
                subfolder_path = os.path.join(test_path, subfolder)
                if os.path.isdir(subfolder_path):
                    # Copy and rename images from each subfolder to 'anomalous/category'
                    for image in os.listdir(subfolder_path):
                        source_image_path = os.path.join(subfolder_path, image)
                        # New image name with category and subfolder as prefixes
                        new_image_name = f"{category}_{subfolder}_{image}"
                        destination_image_path = os.path.join(anomalous_path, category, new_image_name)
                        # Copy the image
                        shutil.copy(source_image_path, destination_image_path)

class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_prefix=""):
        self.root_dir = root_dir
        self.transform = transform
        self.label_prefix = label_prefix
        self.image_paths = []
        self.labels = []

        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for image_name in os.listdir(category_path):
                    if image_name.endswith(('png', 'jpg', 'jpeg')):
                        self.image_paths.append(os.path.join(category_path, image_name))
                        self.labels.append(f"{self.label_prefix}_{category}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

mvtec_dataset_anomal_dino = MVTecDataset('/content/drive/MyDrive/Anomaly Detection Dataset/MVTec/anomalous', transform=transform_dino,label_prefix="anomalous")
mvtec_dataset_non_anomal_dino = MVTecDataset('/content/drive/MyDrive/Anomaly Detection Dataset/MVTec/non_anomalous', transform=transform_dino,label_prefix="non_anomalous")
mvtec_dataset_anomal_mae = MVTecDataset('/content/drive/MyDrive/Anomaly Detection Dataset/MVTec/anomalous', transform=transform_mae,label_prefix="anomalous")
mvtec_dataset_non_anomal_mae = MVTecDataset('/content/drive/MyDrive/Anomaly Detection Dataset/MVTec/non_anomalous', transform=transform_mae,label_prefix="non_anomalous")

dataloader_mvtec_anomal_dino = torch.utils.data.DataLoader(mvtec_dataset_anomal_dino, batch_size=64, shuffle=False)
dataloader_mvtec_non_anomal_dino = torch.utils.data.DataLoader(mvtec_dataset_non_anomal_dino, batch_size=64, shuffle=False)
dataloader_mvtec_anomal_mae = torch.utils.data.DataLoader(mvtec_dataset_anomal_mae, batch_size=64, shuffle=False)
dataloader_mvtec_non_anomal_mae = torch.utils.data.DataLoader(mvtec_dataset_non_anomal_mae, batch_size=64, shuffle=False)

def extract_features(model, dataloader, device):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data_batch, label_batch in dataloader:
            data_batch = data_batch.to(device)
            output_batch = model.forward_features(data_batch)
            features.append(output_batch.cpu().numpy())
            labels.extend(label_batch)  # Keep labels as strings for now

    features = np.concatenate(features, axis=0)
    return features, labels

anomalous_features_dino, anomalous_labels_dino = extract_features(model_dino, dataloader_mvtec_anomal_dino, device)
non_anomalous_features_dino, non_anomalous_labels_dino = extract_features(model_dino, dataloader_mvtec_non_anomal_dino, device)

# Combine features and labels
features = np.concatenate([anomalous_features_dino, non_anomalous_features_dino], axis=0)
labels = np.concatenate([anomalous_labels_dino, non_anomalous_labels_dino], axis=0)

np.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/new_mvtec_dino_features.npy', features)
np.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/new_mvtec_dino_labels.npy', labels)

anomalous_features_mae, anomalous_labels_mae = extract_features(model_mae, dataloader_mvtec_anomal_mae, device)
non_anomalous_features_mae, non_anomalous_labels_mae = extract_features(model_mae, dataloader_mvtec_non_anomal_mae, device)

    # Combine features and labels
features = np.concatenate([anomalous_features_mae, non_anomalous_features_mae], axis=0)
labels = np.concatenate([anomalous_labels_mae, non_anomalous_labels_mae], axis=0)

np.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/new_mvtec_mae_features.npy', features)
np.save('/content/drive/MyDrive/Anomaly Detection Dataset/Extracted features/new_mvtec_mae_labels.npy', labels)
