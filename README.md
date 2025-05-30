# 🧠 Logistic Regression Binary Classifier

This project implements a binary classification model using **Logistic Regression** on the **Breast Cancer Wisconsin Dataset**. It includes model training, evaluation, visualization, and a brief explanation of key concepts relevant to interviews.

---

## 📌 Objective

Build and evaluate a binary classifier using **Scikit-learn**, **Pandas**, and **Matplotlib**, while understanding and applying key machine learning concepts and evaluation metrics.

---

## 📊 Dataset

- **Dataset Used**: [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Task**: Predict whether a tumor is malignant or benign based on diagnostic features.

---

## 🛠 Tools & Libraries

- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🔁 Workflow

1. **Data Loading and Preprocessing**
   - Load dataset
   - Train/test split (80/20)
   - Standardize features

2. **Model Training**
   - Train a `LogisticRegression` model on the training set

3. **Evaluation**
   - Confusion Matrix
   - Precision, Recall, Accuracy
   - ROC Curve and AUC score
   - Threshold tuning

4. **Visualization**
   - Confusion matrix heatmap
   - ROC-AUC curve

5. **Threshold Tuning**
   - Explanation and application of sigmoid function
   - Adjusting decision threshold to balance precision and recall

---

## 🧠 What I Learned

### 📍 Logistic Regression vs Linear Regression
- **Linear Regression** predicts continuous values.
- **Logistic Regression** predicts probabilities and is used for classification tasks.
- It uses the **sigmoid function** to squash output between 0 and 1.

### 📍 Sigmoid Function
- Mathematical function: `σ(z) = 1 / (1 + e^(-z))`
- Converts linear model output into a probability between 0 and 1.

### 📍 Evaluation Metrics
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **Accuracy**: Overall correctness
- **Confusion Matrix**: 2x2 table showing TP, FP, FN, TN
- **ROC-AUC Curve**: Plots TPR vs FPR; AUC represents overall performance

### 📍 Imbalanced Classes
- If one class dominates, accuracy becomes misleading.
- Use metrics like **F1-score**, **AUC**, or **class weighting** to counteract imbalance.

### 📍 Threshold Tuning
- Logistic regression predicts probabilities. Default threshold is 0.5.
- Adjusting threshold can improve precision or recall based on context.

### 📍 Multi-class Logistic Regression
- Yes, logistic regression can handle multiple classes using **one-vs-rest (OvR)** or **multinomial** strategies in Scikit-learn.

---

## 📈 Results

- Achieved high AUC score (~0.98)
- Model correctly classified most cases of malignant and benign tumors
- Balanced performance with tuned threshold

---
