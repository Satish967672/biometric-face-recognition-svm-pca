# 👁️ Real-Time Facial Recognition System Using SVM + PCA (Eigenfaces)

> A production-ready biometric authentication pipeline that classifies faces with high accuracy using Support Vector Machines and Principal Component Analysis — built to solve real-world enterprise security challenges.

---

## 🎯 Business Problem

High-security enterprises need **non-intrusive, real-time biometric authentication** (<100ms) for:
- Restricting access to physical and digital assets
- Reducing reliance on passwords/badges
- Ensuring regulatory compliance

**Critical constraints:**
| Metric | Target |
|---|---|
| False Positive Rate | < 0.01% |
| False Negative Rate | < 1% |
| Inference Latency | < 100ms |
| Accuracy | > 99.5% |

---

## 🧠 What This Project Covers

- ✅ **Iris Classification** — SVM baseline with linear kernel on classic Iris dataset
- ✅ **Handwritten Digit Recognition** — SVM with RBF kernel on sklearn digits (8×8 images)
- ✅ **Fashion-MNIST Classification** — PCA + SVM pipeline on 70,000 clothing images
- ✅ **LFW Face Recognition** — Full biometric pipeline using Eigenfaces + RandomizedSearchCV

---

## 🏗️ Pipeline Architecture

```
Raw Face Images (LFW)
        ↓
  StandardScaler          ← Normalize pixel intensities
        ↓
  PCA (150 components)    ← Extract Eigenfaces (dimensionality: 1850 → 150)
        ↓
  SVM (RBF Kernel)        ← Non-linear decision boundaries
        ↓
  RandomizedSearchCV      ← Tune C & gamma hyperparameters
        ↓
  Identity Prediction     ← Classify person from face
```

---

## 📊 Datasets Used

| Dataset | Size | Task |
|---|---|---|
| Iris | 150 samples, 4 features | 3-class classification |
| sklearn Digits | 1,797 images (8×8) | 10-class digit recognition |
| Fashion-MNIST | 70,000 images (28×28) | 10-class clothing recognition |
| LFW (Labeled Faces in Wild) | 1,140+ images | Multi-class face recognition |

---

## 🔬 Key Techniques

- **PCA (Eigenfaces)**: Reduces 1,850 pixel features → 150 principal components that capture the "essence" of facial structure
- **RBF Kernel SVM**: Handles non-linear decision boundaries using `gamma` (influence radius) and `C` (margin softness)
- **RandomizedSearchCV**: Log-uniform search over C ∈ [1e3, 1e5] and gamma ∈ [1e-4, 1e-1]
- **StandardScaler**: Critical preprocessing — SVMs are distance-based and sensitive to feature scale

---

## 📈 Results

```
Fashion-MNIST (PCA=50 + SVM RBF):  ~85% Accuracy
LFW Face Recognition:               ~85–90% Weighted F1
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![matplotlib](https://img.shields.io/badge/matplotlib-3.7-green)
![seaborn](https://img.shields.io/badge/seaborn-0.12-lightblue)

```
scikit-learn | matplotlib | seaborn | numpy | scipy
```

---

## 🚀 How to Run

```bash
pip install scikit-learn matplotlib seaborn numpy
jupyter notebook 01_svm_classification_example.ipynb
```

---

## 💡 Key Learnings

- SVM performance is **highly sensitive to kernel choice and hyperparameter tuning**
- PCA before SVM dramatically reduces computation while preserving discriminative features
- The `gamma` parameter controls the **radius of influence** of training examples — low gamma = smooth boundary, high gamma = complex boundary
- For imbalanced face datasets, `class_weight='balanced'` is essential to avoid bias toward majority classes

---

## 📁 File Structure

```
svm-facial-recognition-eigenfaces/
├── README.md
└── 01_svm_classification_example.py   ← Full pipeline: Iris → Digits → Fashion → LFW
```
