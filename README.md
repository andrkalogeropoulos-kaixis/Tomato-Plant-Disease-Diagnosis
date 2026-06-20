# Tomato Leaf Disease Detection using Machine Learning

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/andreaskalog/plant-disease-diagnosis)

An advanced, end-to-end computer vision and tabular machine learning pipeline to classify 10 distinct categories of tomato leaf conditions (9 diseases + healthy). This project engineers custom color and texture features from raw images and leverages optimized gradient boosting models to deliver highly robust diagnostics.

<p align="center">
  <img src="images/sample.jpg" alt="Tomato Leaf Late Blight Sample" width="300">
</p>

## 🚀 Performance Summary
* **Baseline Random Forest:** 90.73% (Test Set)
* **Final Optimized XGBoost:** **91.80%** (Test Set)
* **Independent Generalization Test:** **90.95%** (Validation Set)

---

## 📂 Project Architecture & Pipeline

### 1. Data Ingestion & Quality Control
* Filtered and stratified a multi-class dataset into 10 target categories.
* Executed data sanity checks using **Z-score anomaly detection** on the Red channel to isolate overexposed ("burnt") field images. These were intentionally retained to train a model highly resilient to direct sunlight and lens glare.

### 2. Feature Engineering Infrastructure
This project builds a lightweight, interpretable tabular feature space from $128 \times 128$ images:
* **Color Moments (Global):** Extracted Mean, Standard Deviation, and Skewness across RGB and HSV color spaces to map unique disease discoloration (chlorosis, browning, necrosis).
* **Texture Features (Local):** Computed Grey-Level Co-Occurrence Matrices (**GLCM**) to derive Contrast, Dissimilarity, Homogeneity, Energy, Correlation, and ASM to capture fine-grained surface speckling, fungal mycelium, and lesion structures.

### 3. Model Optimization & Iterative Learning
* Trained an initial parallel Bagging Ensemble (Random Forest) optimized via Grid Search.
* Advanced to sequential Gradient Boosting (**XGBoost**) utilizing an **Early Stopping Strategy** (monitored on a 15% training sub-split). The pipeline dynamically converged at exactly **653 trees**, successfully halting before the onset of overfitting.

---

## 📊 Key Insights & Error Diagnostics

### Confusion Matrix Observations
Our diagnostic heatmap proved that the model's errors are heavily tied to biological similarities rather than random noise:
* **The Spot & Mite Overlap:** A minor confusion loop exists between `Target_Spot` and `Spider_mites`, likely because early mite stippling closely mimics the fine micro-texture of young target spot lesions.
* **Fungal vs. Silk Webs:** The model occasionally confuses the fine, white silk webs of spider mites with the grayish-white fungal mycelium or faded centers of `Septoria_leaf_spot`.
* **Chlorosis Similarities:** A symmetric mix-up occurs between `Bacterial_spot` and `Tomato_Yellow_Leaf_Curl_Virus` due to the highly overlapping yellow halos present in both conditions.

---

## ⚠️ Limitations & Future Work

### The Plain Background Constraint
A critical limitation of the current model is its structural dependency on the **uniform, controlled background** (studio lighting/plain backdrops) present in the training dataset. 

* **The Challenge:** While our custom HSV Color Moments and GLCM Texture features perform exceptionally well in isolating the leaf profile against a clean background, they are highly sensitive to noise. In a realistic environment (e.g., a farmer taking a smartphone photo in a field), the background contains soil, weeds, complex shadows, and other non-target plants. The current feature extraction pipeline would struggle to separate these background artifacts from the actual leaf symptoms.
* **The Solution (Future Work):** To transition this project into a production-ready field application, the next deployment phase will isolate and crop the leaf boundary dynamically, wiping out background noise before features are calculated.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Frameworks:** Scikit-Learn, XGBoost
* **Computer Vision:** OpenCV (cv2)
* **Analysis & Visualization:** Pandas, NumPy, Matplotlib, Seaborn

## 📦 How to Run the Notebook

### Option 1: Run Online (Recommended)
Simply open the interactive notebook directly on [Kaggle](https://www.kaggle.com/code/andreaskalog/plant-disease-diagnosis).

### Option 2: Run Locally
If you prefer to run the project on your local machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/andrkalogeropoulos-kaixis/Tomato-Plant-Disease-Diagnosis.git](https://github.com/andrkalogeropoulos-kaixis/Tomato-Plant-Disease-Diagnosis.git)
