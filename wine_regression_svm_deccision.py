#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[11]:


# Load the Wine Recognition Dataset
wine_data = load_wine()
X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
y = pd.DataFrame(wine_data.target, columns=['target'])


# In[12]:


# Explore the dataset
print("Dataset Information:")
print(X.info())
print("\nFirst 5 rows of the dataset:")
print(X.head())
print("\nTarget labels:")
print(y['target'].value_counts())


# In[13]:


# Data Preprocessing
# Check for missing values
print("\nMissing values in the dataset:")
print(X.isnull().sum())


# In[14]:


# Split the dataset into features (X) and target labels (y)
# Perform scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[15]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[16]:


# Model 1: Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train.values.ravel())


# In[17]:


# Model 2: Decision Trees


decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train.values.ravel())


# In[18]:


# Model 3: Support Vector Machines (SVM)
svm_model = SVC()
svm_model.fit(X_train, y_train.values.ravel())


# In[19]:


# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, f1, confusion


# In[20]:


# Evaluate all three models
logistic_results = evaluate_model(logistic_model, X_test, y_test)
decision_tree_results = evaluate_model(decision_tree_model, X_test, y_test)
svm_results = evaluate_model(svm_model, X_test, y_test)


# In[21]:


# Print results
print("\nLogistic Regression Results:")
print("Accuracy:", logistic_results[0])
print("Precision:", logistic_results[1])
print("Recall:", logistic_results[2])
print("F1 Score:", logistic_results[3])
print("Confusion Matrix:")
print(logistic_results[4])

print("\nDecision Tree Results:")
print("Accuracy:", decision_tree_results[0])
print("Precision:", decision_tree_results[1])
print("Recall:", decision_tree_results[2])
print("F1 Score:", decision_tree_results[3])
print("Confusion Matrix:")
print(decision_tree_results[4])

print("\nSVM Results:")
print("Accuracy:", svm_results[0])
print("Precision:", svm_results[1])
print("Recall:", svm_results[2])
print("F1 Score:", svm_results[3])
print("Confusion Matrix:")
print(svm_results[4])


# In[ ]:




