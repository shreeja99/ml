#assgn 1
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets, linear_model 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 

# Load the diabetes dataset 
diabetes = datasets.load_diabetes() 

# X - feature vectors, y - target values 
X = diabetes.data 
y = diabetes.target 

# Splitting X and y into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.4, random_state=1 )

# Create linear regression object 
lin_reg = linear_model.LinearRegression() 

# Train the model 
lin_reg.fit(X_train, y_train) 

# Predict values for X_test data 
predicted = lin_reg.predict(X_test) 

# Regression coefficients 
print('\nCoefficients:\n', lin_reg.coef_) 
print('\nIntercept:', lin_reg.intercept_) 

# Variance score: 1 means perfect prediction 
print('Variance score:', lin_reg.score(X_test, y_test)) 

# Mean Squared Error 
print("Mean squared error: %.2f\n" % mean_squared_error(y_test, predicted)) 

# Plot expected vs predicted values 
plt.title('Linear Regression (Diabetes Dataset)') 
plt.scatter(y_test, predicted, c='b', marker='.', s=36) 
plt.plot(np.linspace(0, 330, 100), np.linspace(0, 330, 100), '--r', linewidth=2) 
plt.xlabel("Expected values") 
plt.ylabel("Predicted values") 
plt.show()