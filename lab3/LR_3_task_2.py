import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Мій варіант 5, тому використовую файл data_regr_5.txt
input_file = 'data_regr_5.txt'

# Завантаження та підготовка даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Навчання моделі
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогноз
y_test_pred = regressor.predict(X_test)

# Візуалізація
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_test_pred, color='red', linewidth=3)
plt.title('Regression for Variant 5')
plt.show()

# Метрики
print("Performance for Variant 5:")
print("MAE =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("MSE =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))