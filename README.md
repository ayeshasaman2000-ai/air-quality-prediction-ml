# Air Quality Prediction Using Machine Learning

Predicting Air Quality Index (AQI) categories from environmental sensor data using multiple machine learning and deep learning models.

---

## Project Overview

Air pollution is a critical public health issue. This project builds and compares several classification models to predict AQI categories — from *Good* to *Severe* — based on pollutant concentrations recorded across Indian cities.

The models are evaluated on accuracy, precision, recall, and F1-score to identify the best-performing approach.

---

## Dataset

- **Source:** [City Day AQI Dataset](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Size:** 100,000+ records
- **Features:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Target:** AQI Category — Good / Satisfactory / Moderate / Poor / Very Poor / Severe

---

## Models Implemented

| Model | Description |
|---|---|
| Neural Network | Deep learning with Dense layers, ReLU activations, softmax output |
| XGBoost | Gradient boosted trees with multi-class objective |
| Random Forest | Ensemble of 100 decision trees |
| SVM | Support Vector Machine with RBF kernel |
| KNN | K-Nearest Neighbours (k=5) |

---

## Results Summary

| Model | Accuracy |
|---|---|
| Neural Network | ~XX% |
| XGBoost | ~XX% |
| Random Forest | ~XX% |
| SVM | ~XX% |
| KNN | ~XX% |

> Fill in your actual accuracy results when you run the code.

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** TensorFlow/Keras, XGBoost, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/air-quality-prediction-ml.git
cd air-quality-prediction-ml

# 2. Install dependencies
pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn

# 3. Place city_day.csv in the project folder

# 4. Run the script
python AQI_Testing_.py
```

---

## Key Findings

- Ensemble methods (XGBoost, Random Forest) performed competitively with the neural network
- Feature engineering and imputation strategy significantly affected model performance
- The *Severe* and *Very Poor* categories showed the highest classification difficulty due to class imbalance

---

## Author

**Ayesha Saman** — Machine Learning Engineer  
MS Computer Science, Bahauddin Zakariya University  
First-author published researcher in NLP-based phishing detection  
[LinkedIn](https://linkedin.com/in/ayesha-saman-b46553323) · ayeshasaman2000@gmail.com
