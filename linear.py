#assgn 1 w/o dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
Y = np.array([35,42,50,55,63,70,78,85])

model = LinearRegression()
model.fit(X, Y)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

hours = np.array([[6.5]])
print("Predicted Score:", model.predict(hours)[0])
print("R2 Score:", model.score(X, Y))

plt.scatter(X, Y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Linear Regression")
plt.legend()
plt.grid()
plt.show()