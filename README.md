---

# 🌿 Plant Disease Detection with Hybrid CNN-ViT Model

A computer vision system using **CNN (ResNet-50)** and **Vision Transformer (ViT-B/16)** to detect plant diseases from leaf images.

---

## 📖 Overview

This project implements a **hybrid deep learning model** for automated plant disease classification using the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease). It combines:

- **ResNet-50** for local feature extraction (edges, textures),
- **ViT-B/16** for global context understanding (entire leaf structure).

**Highlights:**

- 98.88% validation accuracy.
- Web-based interface using **Flask** for real-time predictions.
- Model interpretability with **Grad-CAM** heatmaps.

---

## 📑 Table of Contents

- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training the Model](#training-the-model)
- [Running the Flask App](#running-the-flask-app)
- [Training Results](#-training-results)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## ✨ Features

- ✅ Hybrid model: **ResNet-50 + ViT-B/16**
- ✅ Explainable AI with **Grad-CAM**
- ✅ Easy-to-use **Flask Web App**
- ✅ Compatible with **Google Colab** (GPU-enabled)
- ✅ Balanced and cleaned dataset with EDA

---

## 📊 Dataset

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes**: 15 selected disease classes from pepper, potato, and tomato plants.
- **Total Images**: 54,306 (original), cleaned and reduced to relevant classes.

### Selected Classes

```
Pepper_bell___Bacterial_spot
Pepper_bell___healthy
Potato___Early_blight
Potato___healthy
Potato___Late_blight
Tomato___Target_Spot
Tomato___Tomato_mosaic_virus
Tomato___Tomato_Yellow_Leaf_Curl_Virus
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___healthy
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___Septoria_leaf_spot
Tomato___Spider_mites_Two_spotted_spider_mite
```

### Preprocessing Steps

- Removed corrupted and low-quality images.
- Balanced dataset through oversampling.
- Split: **70% training**, **21% validation**, **9% test**.

---

## 🧠 Model Architecture

- **ResNet-50**: Local feature extractor (2048 features)
- **ViT-B/16**: Global context extractor (768 features)
- **Fusion Layer**: Concatenates both → Linear → 512D
- **Classifier**: Final Linear layer → 15 classes
- **Input Size**: 224 × 224

---

## 🛠️ Installation

### 📦 Prerequisites

- Python 3.11+
- NVIDIA GPU + CUDA 12.4 (optional)
- Google Colab (recommended for ease)

### 🔧 Local Setup

```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection

python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 🔌 Install PyTorch

```bash
# For CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For CPU-only
pip install torch torchvision
```

### 🧳 Google Colab Setup

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn albumentations flask pyngrok opencv-python
```

---

## 🚀 Usage

### 🧠 Training the Model

#### 📁 On Google Colab

```python
# Extract dataset
import zipfile
zip_path = '/content/drive/My Drive/PlantDiseaseProject/PlantVillage.zip'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/PlantVillage')
```

Open `plant_disease_detection.ipynb`, follow the notebook cells to:
- Preprocess and clean data
- Train and evaluate the model
- Save the best model to Google Drive

```python
import shutil
shutil.copy('/content/models/plantvillage_hybrid_best.pth', '/content/drive/My Drive/PlantDiseaseProject/')
```

#### 💻 Locally

- Place dataset in: `Data/PlantVillage/`
- Open `notebooks/train_model.ipynb` in Jupyter
- Model saved to: `models/plantvillage_hybrid_best.pth`

---

## 🌐 Running the Flask App

### 🔁 On Colab (with ngrok)

```bash
!pip install pyngrok flask
!ngrok authtoken YOUR_AUTH_TOKEN
```

- Run `app.py` via notebook
- Access via ngrok URL

### 💻 Locally

```bash
python app.py
```

- Visit: `http://127.0.0.1:5000`
- Upload leaf image → View prediction + Grad-CAM heatmap

---

## 📈 Training Results

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|
| 1     | 0.2600     | 0.0486   | 0.9849  |
| 2     | 0.1021     | 0.0326   | 0.9903  |
| 3     | 0.0836     | 0.0482   | 0.9845  |
| 4     | 0.0747     | 0.0679   | 0.9764  |
| 5     | 0.0667     | 0.0393   | 0.9851  |
| 6     | 0.0547     | 0.0805   | 0.9721  |
| 7 ✅ | 0.0538     | 0.0259   | 0.9888  |
| 8     | 0.0459     | 0.0598   | 0.9818  |

✅ **Best Model** at Epoch 7 – **Accuracy: 98.88%**

---

## 🚢 Deployment

### 🖥️ Flask Web App

- Upload image
- Returns:
  - Predicted disease class
  - Confidence score
  - Grad-CAM heatmap

### 🛠️ Deployment Options

- ✅ Colab via `ngrok`
- ✅ Localhost via `app.py`
- ☁️ (Future) Cloud platforms: Heroku, AWS, etc.

---

## 📁 Project Structure

```
plant-disease-detection/
├── Data/
│   └── PlantVillage/
├── models/
│   └── plantvillage_hybrid_best.pth
├── notebooks/
│   ├── eda_cleaning.ipynb
│   └── train_model.ipynb
├── templates/
│   └── index.html
├── static/
├── app.py
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# Fork and clone
git checkout -b feature/your-feature
# Make changes, then
git commit -m "Add your feature"
git push origin feature/your-feature
```

Then, open a **Pull Request**.

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **PlantVillage Dataset** – Kaggle
- **PyTorch**, **Torchvision**
- **Google Colab** – GPU acceleration
- **Albumentations** – Image augmentation
- **Grad-CAM** – Explainable AI
- **xAI's Grok** – Inspiration engine

---