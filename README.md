# Cancer-Cells-Prediction-
Cancer Cells Prediction Using Deep Learning with the help of a CNN Model with Python Using Keras, Pandas and Tensorflow, Seaborn



# 🩺 Lung Cancer Prediction using CNN and Transfer Learning

This project aims to classify **lung CT scan images** into different cancer types — *Adenocarcinoma*, *Large Cell Carcinoma*, *Squamous Cell Carcinoma*, and *Normal* — using **Convolutional Neural Networks (CNN)** combined with **transfer learning**.

---

## 🚀 Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection through automated image classification can significantly improve patient outcomes.
This project demonstrates how **deep learning** models can accurately classify lung cancer types from CT images using **TensorFlow** and **Keras**.

---

## 📂 Dataset

The dataset used is derived from the **Lung and Colon Cancer Histopathological Images (LC25000)** dataset, publicly available on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).

**Dataset structure:**

```
dataset/
│
├── train/
│   ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/
│   ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/
│   ├── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/
│   └── normal/
│
├── valid/
└── test/
```

Each class folder contains CT scan images of that particular category.

---

## 🧠 Model Architecture

The model leverages **transfer learning** using pre-trained networks like **VGG16**, **ResNet50**, or **InceptionV3** as the base, followed by custom classification layers.

**Key Steps:**

1. Image preprocessing and normalization
2. Data augmentation to prevent overfitting
3. Transfer learning feature extraction
4. Dense layers for classification
5. Model evaluation and accuracy visualization

---

## ⚙️ Installation and Setup

### Prerequisites

Make sure you have:

* Python ≥ 3.8
* TensorFlow ≥ 2.10
* Keras, NumPy, Matplotlib, Seaborn, Scikit-learn

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook "Cancer_Prediction.ipynb"
```

If using Google Colab, upload the notebook and dataset ZIP, then run all cells.

---

## 📊 Results and Evaluation

* Accuracy: ~95–98% on validation data (depending on the pretrained model)
* Loss and accuracy plots show stable convergence
* Confusion matrix and classification report visualize class-level performance

---

## 📈 Example Visualization

Example confusion matrix and prediction output:

| Actual         | Predicted      |
| :------------- | :------------- |
| Adenocarcinoma | Adenocarcinoma |
| Normal         | Normal         |
| Squamous       | Large Cell     |

Training and validation curves are plotted using **Matplotlib**.

---

## 🧩 Technologies Used

* Python
* TensorFlow / Keras
* NumPy / Pandas / Matplotlib / Seaborn
* Scikit-learn
* Google Colab

---

## 🧪 Performance Metrics

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**

---

## 📚 References

* [LC25000 Dataset - Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
* Kermany, D. S. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 172(5), 1122–1131.
* TensorFlow and Keras Documentation

---

## 🧑‍💻 Author

**Shree Harsh**
🎓 Jain University



---

## 🏁 Future Work

* Experiment with additional pretrained models (DenseNet, EfficientNet)
* Integrate explainable AI (Grad-CAM visualization)
* Develop a web-based prediction interface using Flask or Streamlit

---

REQUIREMENTS

tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
Pillow>=9.5.0
jupyter>=1.0.0
notebook>=7.0.0
zipfile36>=0.1.3

See the [LICENSE](LICENSE) file for details.
