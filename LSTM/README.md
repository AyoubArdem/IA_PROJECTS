
# 🎵 Music Structure Analysis & Classification using Librosa + LSTM

##  Project Overview

This project focuses on **Music Information Retrieval (MIR)** and **audio structure understanding**.
It downloads an audio track from YouTube, extracts meaningful **time–frequency audio features**, analyzes the **musical structure**, clusters musical frames into semantic states, and finally trains an **LSTM deep learning model** to learn and classify these states.

The final goal is to build a **model that understands the internal structure of a song** (calm sections, energetic parts, transitions, etc.) based purely on extracted audio features.



##  Objectives

* Download and process music audio automatically from YouTube
* Extract low-level and mid-level audio features
* Represent music as a **sequence of frames**
* Discover musical states using **unsupervised clustering**
* Train an **LSTM model** to learn temporal musical patterns
* Classify song segments into meaningful musical states



##  Technologies & Libraries

* **Python**
* **Librosa** – Audio processing & feature extraction
* **yt-dlp** – YouTube audio downloading
* **NumPy / Pandas** – Numerical computing & data handling
* **Matplotlib** – Visualization
* **Scikit-learn** – Feature scaling & clustering
* **TensorFlow / Keras** – LSTM deep learning model



##  Pipeline Overview

```
YouTube Audio
      ↓
Audio Loading (y, sr)
      ↓
Frame-based Signal Segmentation
      ↓
Feature Extraction per Frame
      ↓
Feature Aggregation → DataFrame
      ↓
KMeans Clustering (State Discovery)
      ↓
Label Assignment
      ↓
LSTM Training (Temporal Learning)
```



##  Audio Acquisition

* Audio is downloaded directly from YouTube using **yt-dlp**
* Converted to MP3 (192 kbps)
* Loaded using `librosa.load()` with:
        * Sample rate: **22050 Hz**
* Temporary audio file is deleted after loading



##  Signal Framing Strategy

* **Frame Length:** `2048 samples` (~93 ms)
* **Hop Length:** `512 samples` (~23 ms)
* Overlapping frames allow smooth temporal analysis
* Each frame represents a short musical moment


##  Extracted Audio Features

###  Energy & Rhythm

* RMS Energy (mean, std)
* Zero Crossing Rate (mean, std)
* Delta RMS, Delta ZCR

### Spectral / Frequency Features

* Spectral Centroid
* Spectral Bandwidth
* Spectral Rolloff
* Spectrogram dB statistics

###  Timbre Features

* MFCCs (13 coefficients)
* Delta MFCCs

Each frame is summarized into a **statistical feature vector**, producing a structured dataset suitable for ML models.


##  Visual Analysis Modules

The project supports multiple MIR visualizations:

* Waveform
* RMS Energy evolution
* Spectrogram + Spectral Rolloff
* Feature evolution over time
* MFCC heatmaps
* Bandwidth evolution

These plots help interpret musical dynamics and transitions.


##  Unsupervised State Discovery

* **KMeans clustering** is applied on standardized features
* Number of clusters: **5**
* Discovered clusters are interpreted as musical states:

| Label      | Description                 |
| ---------- | --------------------------- |
| Calm       | Low energy, smooth sections |
| BuildUp    | Gradual energy increase     |
| Energetic  | High intensity sections     |
| Transition | Structural change points    |
| Repetitive | Repeating musical patterns  |

These states act as **pseudo-labels** for supervised learning.



##  LSTM Model Architecture

The LSTM model learns **temporal dependencies between frames**.

### Architecture:

* LSTM (64 units, return sequences)
* Dropout (0.2)
* LSTM (32 units)
* Dropout (0.2)
* LSTM (16 units)
* Dense Softmax Output (5 classes)

### Training Setup:

* Loss: `Categorical Crossentropy`
* Optimizer: `Adam`
* Early stopping enabled
* Train/Test split: 90% / 10%



##  Model Output

* Learns musical structure evolution over time
* Predicts the **state of each musical segment**
* Accuracy and validation curves are visualized


##  Use Cases

* Automatic music structure analysis
* Song segmentation & annotation
* Music recommendation systems
* Intelligent audio indexing
* MIR research & experimentation



##  Future Improvements

* Replace KMeans with expert-labeled data
* Use bidirectional LSTM or Transformers
* Add tempo and beat-tracking features
* Apply the model to full song segmentation
* Real-time music analysis



##  Key Takeaway

 This project demonstrates how raw audio signals can be transformed into structured data that allows deep learning models to **understand music as a temporal, evolving system**, not just as sound.
