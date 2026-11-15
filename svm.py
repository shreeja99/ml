#assgn 3
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Load dataset
data = pd.read_csv('emails.csv')

print("Total Instances of Dataset:", data.shape[0])

# Remove Email No. column
data = data.drop('Email No.', axis=1)

# Separate features and target
X = data.drop('Prediction', axis=1)
y = data['Prediction']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create SVM classifier
clf = LinearSVC(random_state=42, max_iter=10000)

# Train the model
clf.fit(X_train, y_train)

# Predict test data
y_pred = clf.predict(X_test)

# Evaluate model
print("\nSVM Classification Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
