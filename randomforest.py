#assgn 5
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import matplotlib.pyplot as plt 
# Load dataset 
df = pd.read_csv('car_evaluation.csv', names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]) 
# Encode categorical variables 
le = LabelEncoder() 
for column in df.columns: 
    df[column] = le.fit_transform(df[column])
X = df.drop('class', axis=1) 
y = df['class'] 
# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Train model 
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)
# Predict and Evaluate 
y_pred = model.predict(X_test) 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("Classification Report:\n", classification_report(y_test, y_pred)) 
# Feature Importance 
importances = model.feature_importances_ 
plt.figure(figsize=(8, 6)) 
plt.barh(X.columns, importances) 
plt.xlabel("Feature Importance") 
plt.title("Random Forest Feature Importance")
plt.grid() 
plt.show() 