**🫁 COVID-19 Chest X-Ray Image Classification**

**📌 Project Overview**

This project develops a Convolutional Neural Network (CNN) based image classification model to classify chest X-ray images into two categories:

COVID-19
Normal

The objective is to build a fast and automated first-level screening/decision-support tool that can assist healthcare professionals in prioritizing potential COVID-19 cases.

Note: This project is intended as a screening/decision-support system and is not a replacement for radiologists or confirmatory medical testing.

**🎯 Problem Statement**

Manual analysis of chest X-ray images is time-consuming and requires experienced radiologists. During a pandemic, the high number of patients can increase the workload on healthcare professionals.

This project aims to build a CNN model that can automatically classify chest X-ray images as COVID-19 or Normal with good accuracy.

**🎯 Objectives**

Build a CNN-based COVID-19 image classification model.
Preprocess chest X-ray images for deep learning.
Perform Exploratory Data Analysis (EDA).
Compare different CNN architectures.
Evaluate model performance using classification metrics.
Predict COVID-19 from unseen chest X-ray images.
Develop a reusable prediction pipeline for new X-ray images.

**🏥 Business / Healthcare Context**

Hospitals and diagnostic centers may face a high volume of patients during disease outbreaks. A fast first-level screening system can help prioritize cases for further examination.

The proposed model can:

Assist radiologists as a decision-support tool.
Help prioritize patients who may require further testing.
Reduce the time required for first-level screening.
Provide a scalable and low-cost screening approach.
Support healthcare professionals in resource-constrained environments.

**📊 Dataset**

The dataset contains 251 chest X-ray images belonging to two classes:

Class	Label
Normal	0
COVID	1
Image Details
Image size: 128 × 128 pixels
Channels: 3 (RGB)
Total images: 251
Classes: 2

The images are stored in a NumPy .npy file and their corresponding labels are stored in a .csv file.

**🔍 Exploratory Data Analysis**
The project performs EDA to understand the dataset before model training.

EDA includes:

Checking the shape of the image dataset.
Visualizing individual chest X-ray images.
Displaying multiple images from different classes.
Examining the distribution of COVID and Normal images.

This helps verify image-label alignment and understand the class distribution.

**🧹 Data Preprocessing**

The following preprocessing steps are performed:

Convert class labels:
Covid → 1
Normal → 0
Split the dataset using a stratified approach:
70% Training
15% Validation
15% Testing
Normalize image pixel values from the range:
0–255 → 0–1
Resize images to:
128 × 128 × 3
Use the preprocessed images as input to the CNN models.

**🧠 Model Development**

Two CNN architectures were developed and compared.

CNN Model 1

A deeper CNN architecture containing 3 convolutional blocks.

CNN Model 2

A simpler CNN architecture containing 2 convolutional blocks.

The models were evaluated based on:

Training Accuracy
Validation Accuracy
COVID Recall
Generalization
Overfitting

**📈 Model Comparision**

Metric	CNN Model 1	CNN Model 2
Convolution Layers	3	2
Training Accuracy	100%	96%
Validation Accuracy	97.37%	97.37%
COVID Validation Recall	100%	94.12%
Complexity	High	Low
Overfitting	Yes	No / Very Little
Generalization	Poorer	Better

**🏆 Final Model**

CNN Model 2 was selected as the final model because it achieved comparable validation performance while using a simpler architecture and showing better generalization.

**📊 Final Model Performance**

The selected CNN Model 2 achieved:

Accuracy: 97.37%
COVID Recall: 94.12%

The model demonstrated strong classification performance on unseen data.

For a screening-oriented application, COVID recall is particularly important, because missing a true COVID case can be more costly than generating a false positive.

**🔮 Prediction on Unseen Images**

The project supports prediction on:

1. Test Images

The model predicts COVID/Normal on images from the held-out test dataset.

2. New X-Ray Images

A completely new chest X-ray image can be uploaded and passed through the trained model.

The prediction pipeline performs:

New X-Ray Image
        ↓
Resize to 128 × 128
        ↓
Pixel Normalization
        ↓
Reshape
        ↓
CNN Model 2
        ↓
Prediction Probability
        ↓
COVID / NORMAL

The reusable function:

predict_covid_xray(image_path)

returns:

Predicted class
Confidence score
Raw prediction probability

**🛠️ Tech Stack**

Python
NumPy
Pandas
OpenCV
Matplotlib
Seaborn
TensorFlow
Keras
Scikit-learn
CNN (Convolutional Neural Network)
Google Colab
Google Drive

**📚 Key Libraries**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

**💡 Key Findings**

The deeper CNN model achieved 100% training accuracy, indicating possible overfitting.
The simpler CNN Model 2 achieved 96% training accuracy and 97.37% validation accuracy.
Model 2 provided a better balance between performance and complexity.
The final model achieved 94.12% COVID recall.
The model can classify unseen chest X-ray images automatically.

**⚠️ Limitations**

The project has several limitations:

The dataset contains only 251 images, which is relatively small for deep learning.
The model was trained on only two classes: COVID and Normal.
Other lung diseases, such as different types of pneumonia, were not included.
Performance may differ on X-rays from different hospitals, scanners, and patient populations.
The model should not be used as a standalone clinical diagnostic system.

**🚀 Future Improvements**

Future work can include:

Increasing the dataset size.
Using images from multiple hospitals and sources.
Applying data augmentation.
Adding additional classes such as Other Pneumonia.
Experimenting with transfer learning models such as ResNet, VGG, or EfficientNet.
Performing external validation on a completely independent dataset.
Conducting clinical validation with radiologists.
Monitoring false negatives carefully before any real-world deployment.
