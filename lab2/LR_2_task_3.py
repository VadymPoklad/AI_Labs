import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Мій код для аналізу та класифікації сортів ірисів
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# --- КРОК 2. ВІЗУАЛІЗАЦІЯ ДАНИХ (ЯК У ПДФ) ---

# 1. Одновимірні графіки: Діаграма розмаху ("скриня з вусами")
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.suptitle("Діаграма розмаху атрибутів вхідних даних")
pyplot.show()

# 2. Гістограми розподілу атрибутів
dataset.hist()
pyplot.suptitle("Гістограма розподілу атрибутів датасета")
pyplot.show()

# 3. Багатовимірні графіки: Матриця діаграм розсіювання
scatter_matrix(dataset)
pyplot.suptitle("Матриця діаграм розсіювання")
pyplot.show()

# --- КРОК 3. ПІДГОТОВКА ДАНИХ ---

array = dataset.values
X = array[:, 0:4]
y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# --- КРОК 4. ПОБУДОВА ТА ПОРІВНЯННЯ МОДЕЛЕЙ ---

models = [
    ('LR', LogisticRegression(solver='lbfgs', max_iter=1000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

results = []
model_names = []
print("--- Результати оцінки моделей ---")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    model_names.append(name)
    print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')

# Графік порівняння алгоритмів
pyplot.figure(figsize=(10, 5))
try:
    pyplot.boxplot(results, tick_labels=model_names)
except AttributeError:
    pyplot.boxplot(results, labels=model_names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# --- КРОК 6-8. ПРОГНОЗ ТА ОЦІНКА ---

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\n--- Фінальний звіт (SVM) ---")
print(f"Accuracy: {accuracy_score(Y_validation, predictions)}")
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Передбачення для нової квітки
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
prediction = model.predict(X_new)
print(f"\nНова квітка {X_new} -> Прогноз: {prediction[0]}")