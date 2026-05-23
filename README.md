# Credit Approval Prediction Using Logistic Regression

## Project Overview

This project analyzes a credit approval dataset using Logistic Regression to predict whether a credit card application will be approved or rejected.

The objective of the project is to demonstrate a complete machine learning workflow including data cleaning, preprocessing, feature transformation, model training, and evaluation.

The project uses the UCI Credit Approval Dataset, which contains both numerical and categorical features along with missing values, making it suitable for classification and preprocessing tasks.

---

## Dataset

Dataset used: **UCI Credit Approval Dataset**

The dataset contains:

- 690 rows
- 16 columns
- Credit card application records

### Important Features

- `A1`
- `A2`
- `A3`
- `A4`
- `A5`
- `A6`
- `A7`
- `A8`
- `A9`
- `A10`
- `A11`
- `A12`
- `A13`
- `A14`
- `A15`
- `A16` → Target variable

### Target Variable

- `+` = Approved
- `-` = Rejected

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Data Cleaning

The following preprocessing steps were performed:

- Replaced missing values represented by `?`
- Converted numeric columns into numerical data types
- Filled missing numerical values using the median
- Filled missing categorical values using the mode
- Verified dataset consistency
- Checked missing values and feature types

---

## Data Visualization

Several visualizations were created to better understand the dataset.

### Visualizations Included

- Credit approval distribution
- Confusion matrix visualization
- Model evaluation metrics

---

## Data Preprocessing

The dataset contains both numerical and categorical variables, so preprocessing techniques were applied before training the model.

### Numerical Features

The following columns were standardized using `StandardScaler`:

```python
numeric_columns = ["A2", "A3", "A8", "A11", "A14", "A15"]
```

### Categorical Features

The following columns were encoded using `OneHotEncoder`:

```python
categorical_columns = [
    "A1", "A4", "A5", "A6",
    "A7", "A9", "A10",
    "A12", "A13"
]
```

---

## Machine Learning Model

A Logistic Regression model was used for binary classification.

The preprocessing pipeline was built using:

- `ColumnTransformer`
- `Pipeline`
- `StandardScaler`
- `OneHotEncoder`
- `LogisticRegression`

### Train-Test Split

The dataset was divided into:

- 80% training data
- 20% testing data

Using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## Model Evaluation

The model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

### Evaluation Results

| Metric | Score |
|---|---|
| Accuracy | 0.87 |
| Precision | 0.85 |
| Recall | 0.85 |
| F1 Score | 0.85 |

---

## Classification Report

```python
              precision    recall  f1-score   support

    Rejected       0.88      0.88      0.88        77
    Approved       0.85      0.85      0.85        61

    accuracy                           0.87       138
   macro avg       0.87      0.87      0.87       138
weighted avg       0.87      0.87      0.87       138
```

---

## Confusion Matrix

The confusion matrix was used to visualize the model predictions and classification errors.

The model correctly classified most approved and rejected applications while maintaining balanced precision and recall scores.

---

## Conclusion

This project demonstrates how Logistic Regression can be applied to a real-world credit approval dataset for binary classification tasks.

The workflow included:

- Data cleaning
- Missing value handling
- Feature preprocessing
- Feature encoding
- Feature scaling
- Model training
- Model evaluation

The model achieved strong performance with approximately 87% accuracy and balanced evaluation metrics across both classes.

---

## Dataset Source

UCI Machine Learning Repository — Credit Approval Dataset