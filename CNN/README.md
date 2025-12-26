# Rice Image Classification Project

##  Overview
This project uses **Convolutional Neural Networks (CNNs)** to classify rice grain images into five categories:
- Arborio  
- Basmati  
- Ipsala  
- Jasmine  
- Karacadag  

The dataset comes from [Murat Koklu’s Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset).  
The notebook demonstrates the full pipeline: dataset import, preprocessing, model building, training, evaluation, and prediction.



##  Project Workflow

### 1. **Dataset Import**
- The dataset is downloaded using `kagglehub.dataset_download`.
- Images are organized into subfolders by rice type.

### 2. **Exploration**
- The notebook lists dataset folders and counts images per class.
- A few sample images are displayed with `matplotlib`.

### 3. **Train/Test Split**
- Images are shuffled and split into **80% training** and **20% testing** using `train_test_split`.

### 4. **Preprocessing**
- Images are read with OpenCV (`cv2.imread`).
- Resized to `(128,128)` for uniform input shape.
- Normalized to `[0,1]` by dividing pixel values by 255.
- Labels are mapped to integers (`0–4`) and converted to **one-hot encoding** with `to_categorical`.

### 5. **Model Architecture**
A CNN built with `tensorflow.keras`:
- Input: `(128,128,3)`
- Convolutional layers with ReLU activation
- MaxPooling layers to reduce spatial dimensions
- Dense layers for classification
- Output: 5 neurons with softmax activation

### 6. **Training**
- Optimizer: Adam  
- Loss: categorical crossentropy  
- Metrics: accuracy  
- Trained for 10 epochs with batch size 32  
- Validation performed on the test set

### 7. **Evaluation**
- Accuracy curves are plotted for training and validation.
- Predictions are made on test samples, with class names displayed.



## ❓ Why the Dataset Was Minimized

The original dataset contains **75,000+ images**, which exceeds the RAM limits of Kaggle and Colab environments.  
To prevent memory overflow and ensure smooth training:
- Only **9,000 training images** and **2,000 test images** were used.  
- Images were resized to `(128,128)` to reduce memory usage.  
- This subset still provides enough data to train and evaluate the CNN effectively, while keeping resource usage manageable.

This trade-off balances **model accuracy** with **hardware constraints**. For larger-scale experiments, you can:
- Use **data generators** (`ImageDataGenerator`) to load images in batches.  
- Train on cloud GPUs/TPUs with higher memory.  
- Experiment with transfer learning (e.g., MobileNet, ResNet) to leverage pretrained models on smaller datasets.

## Future Improvements
- Implement **Grad-CAM** to visualize model attention on rice grains.  
- Use **data augmentation** (rotation, zoom, flip) to improve generalization.  
- Explore **transfer learning** for higher accuracy with fewer samples.  
- Optimize hyperparameters (learning rate, batch size, epochs).

## Results
- The CNN achieves good accuracy on the reduced dataset. (accuracy > 95%)
- Predictions correctly identify rice types with confidence scores.  
- The project demonstrates how to adapt large datasets to limited environments like Kaggle/Colab.
