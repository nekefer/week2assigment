
# In[1]:


import pandas as pd
import numpy as np

  # Import libraries for visualization
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

  # Import scikit-learn tools for preprocessing, modeling, and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
      confusion_matrix,
      classification_report,
      accuracy_score,
      precision_score,
      recall_score,
      f1_score
  )

# Set a clean visual style
sns.set(style="whitegrid")


# In[2]:


# Define column names based on the UCI Credit Approval dataset documentation
columns = [
      "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
      "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"
  ]

# Load the dataset
df = pd.read_csv("creditApproval/crx.data", header=None, names=columns)

  # Display the first five rows
print("First five rows:")
print(df.head())


# In[3]:


# Check dataset shape
print("\nDataset shape:")
print(df.shape)
  # Check the target variable distribution
print("\nTarget variable distribution:")
print(df["A16"].value_counts())
sns.countplot(data=df, x="A16")
plt.title("Credit Approval Distribution")
plt.xlabel("Application Status")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("approval_distribution.png")
plt.close()
print("\nSaved approval distribution chart to approval_distribution.png")


# In[4]:


# Replace question marks with NaN
df = df.replace("?", np.nan)

# Define numeric and categorical columns
numeric_columns = ["A2", "A3", "A8", "A11", "A14", "A15"]

categorical_columns = [
    "A1", "A4", "A5", "A6", "A7",
    "A9", "A10", "A12", "A13"
  ]

# Convert numeric columns stored as text into numeric values
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

  # Check missing values before cleaning
print("Missing values before cleaning:")
print(df.isna().sum())


# In[5]:


# Fill missing numeric values with the median
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing categorical values with the mode
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Check missing values after cleaning
print("Missing values after cleaning:")
print(df.isna().sum())


# In[6]:


# Separate features from target variable
X = df.drop("A16", axis=1)

# Convert target variable to binary values
# + = approved, - = rejected
y = df["A16"].map({"+": 1, "-": 0})


# In[7]:


# Confirm feature and target dimensions
print("\nFeature and target dimensions:")
print(X.shape, y.shape)


# In[8]:


# Check target class distribution
print("\nTarget class distribution:")
print(y.value_counts())


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size=0.2,
      random_state=42,
      stratify=y
  )


# In[10]:


# Check the size of each split
print("\nTrain/test split sizes:")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[11]:


print("\nTraining target distribution:")
print(y_train.value_counts())


# In[12]:


print("\nTest target distribution:")
print(y_test.value_counts())


# In[13]:


# Identify numeric and categorical columns
numeric_columns = ["A2", "A3", "A8", "A11", "A14", "A15"]

categorical_columns = [
      "A1", "A4", "A5", "A6", "A7",
      "A9", "A10", "A12", "A13"
  ]

  # Create preprocessing transformer
preprocessor = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), numeric_columns),
          ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
      ]
  )


# In[14]:


# Create a pipeline with preprocessing and logistic regression model
model = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          ("classifier", LogisticRegression(max_iter=1000, random_state=42))
      ]
  )

  # Train the model
model.fit(X_train, y_train)


# In[15]:


# Check model training score
print("\nModel training score:")
print(model.score(X_train, y_train))


# In[16]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[17]:


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = pd.DataFrame({
      "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
      "Score": [accuracy, precision, recall, f1]
  })
print("\nEvaluation metrics:")
print(metrics)


# In[18]:


# Display detailed classification report
report = classification_report(y_test, y_pred, target_names=["Rejected", "Approved"])
print(report)


# In[19]:


# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion matrix:")
print(cm)


# In[20]:


 # Visualize confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Rejected", "Predicted Approved"],
    yticklabels=["Actual Rejected", "Actual Approved"]
  )

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("\nSaved confusion matrix chart to confusion_matrix.png")

