import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Читаємо файл з трафіком
input_file = 'traffic_data.txt'
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line.strip().split(',')
        data.append(items)

data = np.array(data)

# Кодуємо рядки (текст) у числа, бо регресор не розуміє слова
label_encoders = []
X_encoded = np.empty(data.shape)

for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(data[:, i])
        label_encoders.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Вчимо регресор на випадкових лісах
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

# Перевіряємо точність
y_pred = regressor.predict(X_test)
print("Середня абсолютна помилка (MAE):", round(mean_absolute_error(y_test, y_pred), 2))

# Тест на конкретному прикладі (Субота, ранок, без матчу)
test_point = ['Saturday', '10:20', 'Atlanta', 'no']
test_point_encoded = []
le_count = 0

for i, item in enumerate(test_point):
    if item.isdigit():
        test_point_encoded.append(int(item))
    else:
        # Трансформуємо через відповідний енкодер
        val = label_encoders[le_count].transform([item])[0]
        test_point_encoded.append(int(val))
        le_count += 1

# Виводимо прогноз
prediction = regressor.predict([test_point_encoded])[0]
print("Прогнозована інтенсивність трафіку:", int(prediction))