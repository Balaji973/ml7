# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data
2.Split Dataset into Training and Testing Sets
3.Train the Model Using Stochastic Gradient Descent (SGD)
4.Make Predictions and Evaluate Accuracy
5.Generate Confusion Matrix

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: BALAJI J
RegisterNumber: 212224040042
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Iris dataset
iris = load_iris()
# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
# Display the first few rows of the dataset
print(df.head())
# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)
# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="1421" height="673" alt="image" src="https://github.com/user-attachments/assets/b9ee6543-7581-4bdd-9fb5-c0ca174c51fb" />

<img width="1400" height="447" alt="image" src="https://github.com/user-attachments/assets/891a922d-9fa1-4cca-bdc4-257dc8eb2877" />

<img width="1424" height="678" alt="image" src="https://github.com/user-attachments/assets/09586c82-87dd-41d2-bb25-21ebb5c438e0" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
