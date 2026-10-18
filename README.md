# Real-Time Facial Emotion Detection

A real-time facial emotion recognition system that benchmarks **four machine learning and deep learning architectures** on the same dataset, with live webcam inference.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white" alt="Keras" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/CustomTkinter-2FA84F?style=flat-square&logo=python&logoColor=white" alt="CustomTkinter" />
</p>

---

## Demo

<img width="349" height="221" alt="Screenshot 2026-05-04 213659" src="https://github.com/user-attachments/assets/68665922-92d7-44f7-b818-3ae36548eb6f" />
<img width="378" height="237" alt="Screenshot 2026-05-04 213714" src="https://github.com/user-attachments/assets/b2576ff0-4dfc-45bd-ad28-b15324153600" />
<img width="390" height="238" alt="Screenshot 2026-05-04 213727" src="https://github.com/user-attachments/assets/485da3f4-9803-4bf4-b60f-9e15c7ed29d9" />

---

## Table of Contents

- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Models & Results](#models--results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Files](#model-files)

---

## Features

- **Real-time detection** — emotions analysed live from the webcam stream
- **4 models compared** — InceptionV3, CNN V2, SVM (RBF), DenseNet121, all trained and evaluated on the same data
- **7 emotion classes** — Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Confidence scores** — per-emotion probability scores displayed next to each detected face
- **Desktop GUI** — CustomTkinter application with dynamic model selection and animated probability bars
- **Evaluation visualisations** — confusion matrices, ROC/AUC curves and learning curves for every model

---

## Pipeline Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌─────────────────┐
│   RAF-DB     │───▶│  Preprocessing   │───▶│   4 Models     │───▶│  Live Inference │
│  ~15K images │    │ resize · normalize│   │ InceptionV3    │    │  OpenCV webcam  │
│  7 emotions  │    │ augmentation      │    │ CNN V2 · SVM   │    │  + Tkinter GUI  │
└──────────────┘    │ class balancing   │    │ DenseNet121    │    └─────────────────┘
                    └──────────────────┘    └────────────────┘
```

| Step | Description |
|------|-------------|
| **Dataset** | RAF-DB (Real-world Affective Faces Database) — ~15,000 real-world face images |
| **Preprocessing** | Resize to 128×128 (deep models) or 48×48 (SVM), normalisation, augmentation (rotation, zoom, flip) |
| **Class balancing** | Undersampling of the majority class + oversampling of minority classes |
| **Training** | All four models trained and evaluated on identical splits for a fair comparison |
| **Deployment** | Real-time inference through OpenCV, exposed via a desktop GUI |

---

## Models & Results

### Performance Comparison

| Model | Test Accuracy | Average AUC | Training Time | Parameters |
|-------|--------------|-------------|---------------|------------|
| **InceptionV3** ⭐ | **85.59 %** | **0.97 – 0.99** | ~252 min | ~22.9 M |
| CNN V2 | 80.62 % | 0.95 – 1.00 | ~240 min | ~4.9 M |
| SVM (RBF) | 75.78 % | 0.87 – 0.97 | ~17 min | ~10 K |
| DenseNet121 | 73.01 % | 0.91 – 0.98 | ~280 min | ~7.6 M |

> **InceptionV3 is the selected model** — the best accuracy/robustness trade-off, backed by excellent per-class AUC scores (0.97 – 0.99).

### Model Details

#### InceptionV3 (transfer learning)
- ImageNet-pretrained backbone
- Custom head: Dropout 20 % → Flatten → Dense 128 → Dense 128 → Dense 7 (softmax)
- Fine-tuning of the last layers
- **Per-class AUC**: Surprise 0.99 · Fear 0.99 · Happy 0.99 · Angry 0.99 · Disgust 0.97 · Neutral 0.97 · Sad 0.96

#### CNN V2 (custom architecture)
- 4 Conv2D + MaxPooling blocks (32 → 64 → 128 → 512 filters)
- Flatten → Dense 512 → Dropout 50 % → Dense 7 (softmax)
- Lightweight and fast to deploy

#### SVM with RBF kernel
- Flattened pixel features
- `StandardScaler` normalisation before training
- Kernel comparison: Linear 60.31 % · **RBF 75.78 %** · Polynomial 71.71 %

#### DenseNet121 (dense connections)
- ImageNet-pretrained backbone
- GlobalAveragePooling2D → Dense 512 → Dropout 40 % → Dense 7 (softmax)
- Dense inter-layer connections mitigate the vanishing gradient problem

### Detailed Results

#### Confusion Matrix — InceptionV3

<img width="519" height="438" alt="Confusion matrix" src="https://github.com/user-attachments/assets/71265e0c-69df-4163-8d98-dce1310161ba" />

**Observations**
- Near-perfect recognition for Happy (410/410), Neutral (402/402) and Angry (359/359)
- Minor confusions: Sad ↔ Neutral (99 cases), Disgust ↔ Happy (17 cases)

#### ROC Curves — InceptionV3

<img width="520" height="405" alt="ROC curves" src="https://github.com/user-attachments/assets/cd459a1d-52e6-4347-bedb-bfd3f7a105ac" />

All ROC curves rise steeply towards 1.0, confirming the strong discriminative power of the model.

---

## Project Structure

```
emotion_detection/
├── Detector_CNN.py             # Webcam demo — CNN V2
├── Detector_DenseNet.py        # Webcam demo — DenseNet121
├── Detector_inception_v3.py    # Webcam demo — InceptionV3
├── detector_svm.py             # Webcam demo — SVM (RBF)
├── interface.py                # CustomTkinter desktop GUI (camera + probability bars)
├── style_config.py             # Shared colour palette and fonts for the GUI
│
├── train_model.ipynb           # Training notebook (deep learning models)
├── modl.ipynb                  # Model experiments and variants
├── svm[1].ipynb                # SVM training and kernel comparison
│
├── scaler_svm_emotion.pkl      # Fitted StandardScaler for the SVM pipeline
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8 or newer
- A working webcam

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ayaa-ch/emotion_detection.git
cd emotion_detection

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

# 3. Install the dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train the models (optional)

Open the notebooks in Jupyter and run them end-to-end:

```bash
jupyter notebook train_model.ipynb   # deep learning models
jupyter notebook "svm[1].ipynb"      # SVM pipeline
```

Each notebook exports its trained model (see [Model Files](#model-files)).

### 2. Run a standalone webcam demo

> **Note:** each script loads its model from a local file path defined at the top of the script. Update the `MODEL_PATH` / `joblib.load(...)` path so it points to your model files before running.

```bash
python Detector_inception_v3.py   # InceptionV3
python Detector_CNN.py            # CNN V2
python Detector_DenseNet.py       # DenseNet121
python detector_svm.py            # SVM (RBF)
```

Press **`q`** to quit the webcam window.

### 3. Run the desktop application

```bash
python interface.py
```

1. Click **"Démarrer la Caméra"** *(Start Camera)*
2. Look at the camera — detected faces are boxed and labelled with their emotion
3. Live probability bars show the score for each of the 7 emotions
4. Click the button again to stop the camera

---

## Model Files

Trained model artifacts are **not** included in the repository (see [.gitignore](.gitignore)). Train them yourself with the notebooks, or drop pre-trained files next to the scripts:

| File | Consumed by |
|------|-------------|
| `best_InceptionV3Model.keras` | `Detector_inception_v3.py` |
| `best_CNNModel.keras` | `Detector_CNN.py` |
| `best_DenseNetModel.keras` | `Detector_DenseNet.py` |
| `svm_model_rbf_emotion.pkl` | `detector_svm.py`, `interface.py` |
| `scaler_svm_emotion.pkl` | `detector_svm.py`, `interface.py` *(included)* |

---

## Limitations & Future Work

- Haar cascades are sensitive to pose and lighting; a modern face detector (e.g. MTCNN, RetinaFace) would improve robustness
- Class imbalance in RAF-DB (Disgust is under-represented) remains the main source of error
- Planned: video-file inference, model switching from the GUI, and a lightweight ONNX export for faster inference

---

*Academic project — Department of Data Science & AI, ENSA Tétouan.*
