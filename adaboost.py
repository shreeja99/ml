#assgn 6
from sklearn.datasets import load_iris 
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report 
iris = load_iris() 
X, y = iris.data, iris.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
ada = AdaBoostClassifier(n_estimators=50) 
ada.fit(X_train, y_train) 
y_pred_ada = ada.predict(X_test) 
gb = GradientBoostingClassifier(n_estimators=100) 
gb.fit(X_train, y_train) 
y_pred_gb = gb.predict(X_test) 
xgb = XGBClassifier( eval_metric='mlogloss') 
xgb.fit(X_train, y_train) 
y_pred_xgb = xgb.predict(X_test) 
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada)) 
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb)) 
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb)) 