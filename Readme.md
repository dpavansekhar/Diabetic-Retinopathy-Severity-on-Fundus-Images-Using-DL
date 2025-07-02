# Multi-Model Comparison of Attention-Augmented CNNs for Diabetic Retinopathy Detection

## 💡 Project Title

**Automated Grading of Diabetic Retinopathy Severity Using EfficientNet and Attention Mechanisms on Fundus Images**

## 🎯 Objective

To develop a deep learning-based system that automatically classifies the severity of Diabetic Retinopathy (DR) from retinal fundus images, enabling early diagnosis and reducing dependency on manual grading by ophthalmologists.

## 🔍 Background

Diabetic Retinopathy is a leading cause of blindness in diabetic patients. Detecting and grading it early is crucial but time-consuming when done manually. Automated systems can provide fast, scalable solutions, especially beneficial for resource-limited healthcare settings.

## ⚙️ Methodology

### 1. Dataset

* **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy)
* **Composition**: Unified dataset combining EyePACS, APTOS (including Gaussian filtered), and Messidor datasets.
* **Images**: \~143,669 high-resolution fundus images after manual augmentation and resizing.
* **Labels**: DR Severity Levels:

  * 0 – No DR
  * 1 – Mild
  * 2 – Moderate
  * 3 – Severe
  * 4 – Proliferative DR

### 2. Image Preprocessing

* Removed black borders
* Resized images to 299x299 (to support InceptionV3/Xception)
* Applied ImageNet preprocessing (mean subtraction, normalization)
* Augmentation via:

  * Random rotations
  * Horizontal/vertical flips
  * Brightness shifts

### 3. Model Architectures

* **Base Models**:

  * EfficientNetB3
  * EfficientNetB5
  * DenseNet121
  * InceptionV3
  * Xception
  * ResNet101

* **Attention Mechanism**:

  * CBAM (Convolutional Block Attention Module) integrated after backbone feature extraction.

* **Classifier Head**:

  * Global Average Pooling
  * Dense Layer with Softmax Activation

### 4. Training Strategy

* **Loss Function**: Categorical Crossentropy (weighted where necessary)
* **Optimizer**: Adam with learning rate scheduler
* **Callbacks**:

  * EarlyStopping (patience=5)
  * ModelCheckpoint (save best weights)
  * ReduceLROnPlateau
* **Epochs**: 50
* **Batch Size**: 32

## 📊 Evaluation Metrics

* **Quadratic Weighted Kappa (QWK)**: Measures agreement between model predictions and ground truth
* **Accuracy**
* **Precision, Recall, F1-Score**: Evaluated per class
* **Confusion Matrix**: Visualizes class-wise prediction accuracy

## 🎯 Results Summary

| Model                 | Accuracy | QWK    | Best F1 Classes |
| --------------------- | -------- | ------ | --------------- |
| EfficientNetB3        | 69.23%   | 0.8144 | 0, 4            |
| EfficientNetB3 + CBAM | 60.33%   | 0.5876 | 0               |
| EfficientNetB5 + CBAM | 70.69%   | 0.8451 | 0, 4            |
| ResNet101 + CBAM      | 63.01%   | 0.6946 | 0, 1            |
| XceptionNet + CBAM    | 66.57%   | 0.7636 | 0, 4            |
| DenseNet121 + CBAM    | 64.11%   | 0.7132 | 0               |
| InceptionV3 + CBAM    | 65.33%   | 0.7745 | 0, 4            |

> Note: EfficientNetB5 + CBAM achieved the **highest QWK score of 0.8451** and the most balanced performance across severity classes.

## 🏥 Clinical Impact

* Enables **automated DR screening** in primary healthcare setups
* Reduces workload on ophthalmologists
* Supports **teleophthalmology** and mobile deployments
* Early diagnosis helps **prevent irreversible vision loss**

## 🚀 Future Work

* Integrate **Grad-CAM** to highlight lesion areas
* Develop **mobile application** for on-device screening
* Extend to **multi-modal** prediction (OCT, patient records)

## 📓 Code Overview (Training Pipeline)

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

base_dir = '/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

IMG_SIZE = 299
BATCH_SIZE = 32
NUM_CLASSES = 5

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = datagen.flow_from_directory(train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')
val_gen   = datagen.flow_from_directory(val_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
test_gen  = datagen.flow_from_directory(test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
```

## 🤖 CBAM Custom Layer

```python
class CBAM(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_dense_one = layers.Dense(channel // self.ratio, activation='relu')
        self.shared_dense_two = layers.Dense(channel)
        self.conv_spatial = layers.Conv2D(1, 7, padding='same', activation='sigmoid')

    def call(self, input_feature):
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))
        channel_attention = tf.nn.sigmoid(avg_out + max_out)
        x = input_feature * channel_attention

        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_attention = self.conv_spatial(concat)
        return x * spatial_attention
```

## 🔗 Dataset Summary

* Unified set from EyePACS, APTOS, Messidor
* Original size: \~92,501 images
* After augmentation: \~143,669 images
* Train/Val/Test: 80% / 10% / 10%
* Resized to 600x600 during preprocessing to reduce storage

---

> 🏆 **Best Model:** EfficientNetB5 + CBAM
> 🎡 **Best Evaluation Metric:** QWK = 0.8451

---

## 📅 Last Updated

**June 2025**
