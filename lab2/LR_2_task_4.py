import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot

# Мій код для порівняння всіх класифікаторів на датасеті доходів
input_file = 'income_data.txt'

X = []
max_datapoints = 10000 # Оптимальна кількість для швидкого, але точного порівняння

# Завантаження та очищення даних
with open(input_file, 'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        X.append(data)
        if len(X) >= max_datapoints:
            break

X = np.array(X)

# Кодування категоріальних ознак
X_encoded = np.empty(X.shape)
for i in range(X.shape[1]):
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])

X_data = X_encoded[:, :-1].astype(int)
y_data = X_encoded[:, -1].astype(int)

# Розподіл на вибірки
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=5)

# Список моделей (використовуємо універсальні налаштування)
models = [
    ('LR', LogisticRegression(solver='lbfgs', max_iter=2000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

results = []
names = []
print("--- Результати порівняння на Income Data ---")

for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} (std: {cv_results.std():.4f})")

# Візуалізація без попереджень (labels -> tick_labels)
pyplot.figure(figsize=(10, 6))
# Перевірка версії matplotlib для уникнення warning
try:
    pyplot.boxplot(results, tick_labels=names)
except AttributeError:
    pyplot.boxplot(results, labels=names)

pyplot.title('Порівняння алгоритмів (Income Data)')
pyplot.ylabel('Accuracy')
pyplot.grid(True, linestyle='--', alpha=0.6)
pyplot.show()