import argparse 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from utilities import visualize_classifier

# Налаштовуємо парсер для вибору типу лісу через термінал
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Класифікація даних методами ансамблевого навчання')
    parser.add_argument('--classifier-type', dest='classifier_type', 
            required=True, choices=['rf', 'erf'], help="Тип класифікатора: 'rf' або 'erf'")
    return parser

if __name__=='__main__':
    # Парсимо аргументи
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type

    # Тягнемо дані з файлу
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    # Розбиваємо на класи для візуалізації (0, 1, 2)
    class_0 = np.array(X[y==0])
    class_1 = np.array(X[y==1])
    class_2 = np.array(X[y==2])

    # Малюємо вхідні точки
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white', 
                    edgecolors='black', linewidth=1, marker='s')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', 
                    edgecolors='black', linewidth=1, marker='o')
    plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white', 
                    edgecolors='black', linewidth=1, marker='^')
    plt.title('Вхідні дані')

    # Ось тут виправив: просто train_test_split, а не подвійний виклик
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=5)

    # Вибираємо параметри для ансамблю
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(**params)
    else:
        classifier = ExtraTreesClassifier(**params)

    # Тренуємо модель
    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_train, y_train, 'Навчальна вибірка')

    # Тестуємо та малюємо результат
    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test, 'Тестова вибірка')

    # Виводимо статистику по навчанню та тесту
    class_names = ['Class-0', 'Class-1', 'Class-2']
    print("\n" + "#"*40)
    print("\nРезультати на навчальній вибірці:\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("#"*40 + "\n")

    print("#"*40)
    print("\nРезультати на тестовій вибірці:\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#"*40 + "\n")

    # Тестові точки для перевірки довірливості (confidence)
    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])

    print("\nМіри довірливості для точок:")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = 'Class-' + str(np.argmax(probabilities))
        print('\nТочка:', datapoint)
        print('Прогнозований клас:', predicted_class) 

    # Візуалізуємо ці специфічні точки на мапі рішень
    visualize_classifier(classifier, test_datapoints, [0]*len(test_datapoints), 
            'Тестові точки даних')

    plt.show()