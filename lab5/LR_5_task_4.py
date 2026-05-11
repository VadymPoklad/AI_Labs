import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Беремо актуальний датасет з нерухомістю
data = datasets.fetch_california_housing()
X, y = shuffle(data.data, data.target, random_state=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Налаштовуємо AdaBoost на деревах рішень
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), 
                              n_estimators=400, random_state=7)
regressor.fit(X_train, y_train)

# Перевіряємо помилки
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\nADABOOST REGRESSOR")
print("Середньоквадратична помилка =", round(mse, 2))
print("Показник дисперсії =", round(evs, 2))

# Витягуємо важливість кожної ознаки
feature_importances = regressor.feature_importances_
feature_names = data.feature_names

# Нормалізуємо для графіку
feature_importances = 100.0 * (feature_importances / max(feature_importances))
index_sorted = np.flipud(np.argsort(feature_importances))
pos = np.arange(index_sorted.shape[0]) + 0.5

# Малюємо стовпчики
plt.figure()
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, np.array(feature_names)[index_sorted])
plt.ylabel('Відносна важливість')
plt.title('Важливість ознак в AdaBoost')
plt.show()