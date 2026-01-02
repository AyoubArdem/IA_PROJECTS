
#  Brain Tumor Detection with YOLOv8 and ML Classifier

## 📌 Project Overview
This project focuses on **brain tumor detection and classification** using a combination of:
- **YOLOv8 (Ultralytics)** for object detection and tumor localization.
- **Feature extraction** from detected regions of interest (ROIs).
- **Machine learning classifiers (XGBoost)** for tumor classification based on extracted features.

The dataset used is the **Medical Image Dataset - Brain Tumor Detection** available on Kaggle.



## 📂 Dataset
- **Training Path:**  
  `/kaggle/input/medical-image-dataset-brain-tumor-detection/BrainTumor/BrainTumorYolov8/train`
- **Testing Path:**  
  `/kaggle/input/medical-image-dataset-brain-tumor-detection/BrainTumor/BrainTumorYolov8/test`

### Classes:
1. `glioma_tumor`  
2. `meningioma_tumor`  
3. `no_tumor`  
4. `pituitary_tumor`


## ⚙️ Installation
```bash
pip install ultralytics
pip install xgboost
```



##  Workflow

### 1. **Data Loading**
- Images and labels are loaded from the dataset.
- Labels are parsed to extract class IDs.

### 2. **Visualization**
- A few sample images are displayed with their corresponding labels using `matplotlib`.

### 3. **YOLOv8 Training**
- YOLOv8n model (`yolov8n.pt`) is fine-tuned on the dataset.
- Training configuration is stored in `data.yaml`.

### 4. **Detection**
- YOLOv8 is used to detect tumors in images.
- Detected bounding boxes are visualized.

### 5. **Feature Extraction**
For each detected tumor, the following features are extracted:
- Tumor area, perimeter, diagonal
- Mean, std, min, max intensity
- Aspect ratio, fill ratio, compactness
- Tumor center (x, y), normalized coordinates
- Brain side (left/right)
- Risk level (low/high)
- Class ID

### 6. **Data Balancing**
- Classes are balanced using **upsampling** with `sklearn.utils.resample`.

### 7. **Preprocessing**
- Features are standardized using `StandardScaler`.

### 8. **Classification**
- **XGBoost** classifier is trained on extracted features.
- Performance is evaluated using:
  - Accuracy
  - Classification report (precision, recall, F1-score)


## 📊 Results
- **Accuracy** is printed after training.( > 90%)
- **Classification report** provides detailed metrics for each tumor class.
- Heatmap of feature correlations with `class_id` is visualized using `seaborn`.



## 🛠️ Technologies Used
- **Python**  
- **OpenCV (cv2)** – Image processing  
- **Matplotlib / Seaborn** – Visualization  
- **Ultralytics YOLOv8** – Object detection  
- **Scikit-learn** – ML utilities (train/test split, scaling)  
- **XGBoost** – Gradient boosting classifier  

---

## 🚀 How to Run
1. Clone the repository or open the Kaggle notebook.
2. Install dependencies:
   ```bash
   pip install ultralytics xgboost
   ```
3. Run the notebook step by step.
4. View detection results and classification metrics.



## Future Improvements
- Add more advanced feature engineering (GLCM, texture features).
- Use ensemble methods combining YOLOv8 + ML classifiers.
- Deploy as a web app for medical professionals.

