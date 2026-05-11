import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантажую датасет діабету
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розбиваю навпіл за завданням
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# Створюю та навчаю модель
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

# Прогноз
ypred = regr.predict(Xtest)

# Вивід характеристик
print("Coefficients:", regr.coef_)
print("Intercept:", regr.intercept_)
print("R2 score:", round(r2_score(ytest, ypred), 2))
print("MAE:", round(mean_absolute_error(ytest, ypred), 2))
print("MSE:", round(mean_squared_error(ytest, ypred), 2))

# Графік
fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.title('Diabetes Dataset Regression')
plt.show()