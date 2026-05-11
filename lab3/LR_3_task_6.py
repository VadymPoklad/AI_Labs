import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend()
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")

# Дані з варіанту 5
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.4 * X ** 2 + X + 4 + np.random.randn(m, 1)

# 1. Криві для лінійної моделі
plt.figure(figsize=(10, 4))
plt.subplot(131)
plot_learning_curves(LinearRegression(), X, y)
plt.title("Linear Regression")

# 2. Криві для полінома 10-го ступеня
from sklearn.pipeline import Pipeline
poly_10_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plt.subplot(132)
plot_learning_curves(poly_10_reg, X, y)
plt.axis([0, 80, 0, 3])
plt.title("Degree 10")

# 3. Криві для полінома 2-го ступеня
poly_2_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plt.subplot(133)
plot_learning_curves(poly_2_reg, X, y)
plt.axis([0, 80, 0, 3])
plt.title("Degree 2")

plt.tight_layout()
plt.show()