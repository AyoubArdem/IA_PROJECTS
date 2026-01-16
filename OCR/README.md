
# OCR Pipeline using COCO-Text v2.0 Dataset

This project demonstrates how to build a simple **Optical Character Recognition (OCR)** workflow in Python using the **COCO-Text v2.0 dataset**. The notebook covers dataset import, image preprocessing, bounding box extraction, character segmentation, and text recognition with **EasyOCR**.



##  Features

- **Dataset Import**  
  - Downloads the COCO-Text v2.0 dataset from Kaggle using `kagglehub`.  
  - Prepares training and validation text files (`train.txt`, `val.txt`) and image data.

- **Data Exploration**  
  - Lists available files in the dataset directory.  
  - Randomly samples and displays images with their dimensions.

- **Feature Extraction**  
  - Reads annotation files containing bounding box coordinates.  
  - Matches images with their corresponding bounding box features.

- **Bounding Box & Character Segmentation**  
  - Converts normalized bounding box coordinates into pixel values.  
  - Extracts regions of interest (ROI) containing words.  
  - Converts ROI to grayscale and binarizes.  
  - Detects contours to isolate individual characters.  
  - Draws bounding boxes around detected characters.

- **OCR Prediction with EasyOCR**  
  - Selects a random image from the dataset.  
  - Runs EasyOCR’s `reader.readtext()` to detect text.  
  - Prints recognized text, confidence scores, and bounding box coordinates.



##  Requirements

- Python 3.x  
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `opencv-python`
  - `easyocr`
  - `kagglehub`

Install missing dependencies with:
```bash
pip install numpy pandas matplotlib opencv-python easyocr kagglehub
```


##  How to Run

1. Open the notebook in **Google Colab**.  
2. Ensure Kaggle API credentials are configured for `kagglehub`.  
3. Run the first cell to download the dataset.  
4. Execute all cells sequentially.  
5. View sampled images, bounding boxes, segmented characters, and OCR predictions.



##  Workflow

1. **Dataset Import**  
2. **Image Sampling**  
3. **Feature Extraction**  
4. **Bounding Box Conversion**  
5. **ROI Segmentation**  
6. **Character Detection**  
7. **OCR Prediction**



##  Notes

- Colab and Kaggle environments differ; some libraries may need manual installation.  
- Small contours (<1000 area) are ignored to reduce noise.  
- EasyOCR requires initialization of the `reader` object:
  ```python
  import easyocr
  reader = easyocr.Reader(['en'])
  ```


