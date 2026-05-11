import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Мій варіант 5
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.4 * X ** 2 + X + 4 + np.random.randn(m, 1)

# Візуалізація початкових даних
plt.scatter(X, y, color='gray')
plt.title('Generated Data (Variant 5)')
plt.show()

# Додаю поліноміальні ознаки (ступінь 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Навчання лінійної моделі на поліноміальних даних
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

print("Intercept:", lin_reg.intercept_)
print("Coefficients:", lin_reg.coef_)

# Малюю результат регресії
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_new, y_new, color='red', linewidth=2, label='Polynomial Prediction')
plt.legend()
plt.show()

# Оцінка якості
y_pred = lin_reg.predict(X_poly)
print("R2 Score:", r2_score(y, y_pred))