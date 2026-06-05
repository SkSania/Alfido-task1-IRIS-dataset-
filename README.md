# Iris Flower Species Classification Engine

A production-ready Machine Learning classification pipeline designed to automatically identify Iris 
flower species (*Setosa*, *Versicolor*, *Virginica*) based on physical measurements: sepal length, sepal width, petal length, and petal width.
This project evaluates multiple classification algorithms, identifies the most optimal architecture, and exports a serialized workflow for seamless deployment.


## 🚀 Key Features
* **Exploratory Data Analysis (EDA):** Feature-interaction analysis and cluster visualization utilizing Seaborn pair plots and box plots.
* **Leakage-Free Preprocessing:** Features automated missing value protection, label encoding, and scaling (`StandardScaler`) combined systematically.
* **Algorithm Benchmarking:** Compares performance across Logistic Regression, k-Nearest Neighbors (k-NN), and Decision Tree Classifiers.
* **Production Deployment Ready:** Serializes the complete pipeline (including transformation objects) into a portable `.joblib` artifact.

## 📊 Dataset Context
The project utilizes the classic **Iris Dataset** sourced from Kaggle. The dataset contains 150 instances tracking four localized petal and sepal dimensions:
* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

## 🏆 Model Performance Summary
The dataset was split using an **80/20 train-to-test ratio** with stratification to ensure fair distribution.


| Algorithm | Test Accuracy | Precision (Avg) | Recall (Avg) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **100.0%** | **1.00** | **1.00** |
| **k-Nearest Neighbors (k=3)** | **100.0%** | **1.00** | **1.00** |
| **Decision Tree** | 96.67% | 0.97 | 0.97 |

*Note: Logistic Regression was selected as the final production model due to its high efficiency and clean, stable probabilistic boundaries.*

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd iris-classification-engine
   ```

2. **Install necessary dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib
   ```

3. **Run the pipeline:**
   Place the downloaded `IRIS.csv` from Kaggle into the project root directory and run the script:
   ```bash
   python iris_classification.py
   ```

---

## 💻 How to Run Inference (Sample)

Once the training runs successfully, it generates a `best_iris_model.joblib` file. You can load this artifact to run predictions on fresh data instances with just a few lines of code:

```python
import joblib
import numpy as np

# Load the saved deployment package
artifacts = joblib.load("best_iris_model.joblib")
loaded_model = artifacts["model"]
loaded_scaler = artifacts["scaler"]
loaded_le = artifacts["label_encoder"]

# Fresh incoming data sample [sepal_length, sepal_width, petal_length, petal_width]
raw_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Scale input metrics as required by the final selected model
scaled_sample = loaded_scaler.transform(raw_sample)

# Generate prediction and decode target label back to text string
predicted_id = loaded_model.predict(scaled_sample)
species_label = loaded_le.inverse_transform(predicted_id)

print(f"Predicted Iris Species: {species_label[0]}")
```

---

## 📁 Repository Structure
```text
├── IRIS.csv                       # Kaggle Raw Dataset File
├── iris_classification.py         # Complete Python Pipeline Code
├── best_iris_model.joblib         # Exported Model & Transformed Artifacts Package
├── eda_pairplot.png               # Generated Visual Matrix Image
├── best_model_confusion_matrix.png# Final Model Evaluation Plot
└── README.md                      # Project Documentation Portfolio
```
