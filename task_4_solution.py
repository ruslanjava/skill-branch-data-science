import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


# Задание 1
# Написать функцию `calculate_data_stats`, которая принимает на вход датафрейм с данными и возвращает
# размерность набора данных, количество числовых признаков, количество категориальных признаков и
# долю целевого события (`isFraud`). Долю целевого события округлить до 2-го знака.
def calculate_data_stats(x):
    shape = x.shape
    ints = x.select_dtypes(include=['float64', 'int64']).dtypes.count()
    objects = x.select_dtypes(include=['object']).dtypes.count()
    is_fraud_positive = x[x['isFraud'] == True].count()
    is_fraud_total = x['isFraud'].count()
    fraud_share = (is_fraud_positive / is_fraud_total).round(2)
    return [shape, ints, objects, fraud_share]


# Задание 2
# Наш набор данных содержит служебную информацию, которая нам не нужна при обучении модели.
# Написать функцию `prepare_data`, которая принимает на вход датафрейм, в теле функции выделяет целевую переменную
# `isFraud` в отдельный вектор и удаляет ненужные для обучения признаки: `isFraud`, `TransactionID`, `TransactionDT`.
# Функция должна возвращать преобразованный набор данных и вектор целевой переменной.
def prepare_data(x):
    target_vector = x['isFraud']
    filtered = x.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    return [filtered, target_vector]


# Задание 3
# Начнем обучать модель и сравнивать их качество.
# Основная схема будет выглядеть следующим образом - будем разбивать выборку `train` на обучение / контроль / тест,
# обучать и оценивать качество модели на этих выборках, а в конце задания будем делать предсказания
# на отдельной выборке `test`. Будем сранвивать качество модели на нашей локальной валидации и на выборке `test`.
# Для оценки качества модели будем использовать метрику - __ROC_AUC__
#
# Написать функцию `fit_first_model`, которая принимает аргументы `X`, `y`, где `X` - набор данных для обучения,
# `y` - вектор целевой переменной, а также `x_test` и `y_test`. Разбить исходную выборку на `x_train`, `x_valid`
# в соотношении 70/30, зафиксированть параметр `random_state=1` и перемешать данные.
# После разбиения обучить логистическую регрессию с параметром `random_state=1` на `x_train`,
# и оценить качество модели на `x_valid`. При необходимости, заполнить пропуски 0.
# Функция должна возвращать оценку качества модели на выборках `x_valid` и `x_test`.
# Значение метрики качества оценить с точностью до 4-го знака.
#
# __Пример__:
# `def fit_first_model(X, y, x_test, y_test) -> Tuple[float, float]:`
def fit_first_model(x, y, x_test, y_test):
    x = x.fillna(0)
    x_test = x_test.fillna(0)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=1, shuffle=True)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    score1 = roc_auc_score(y_valid, y_pred)
    y_test_pred = model.predict(x_test)
    score2 = roc_auc_score(y_test_pred, y_test)
    return [round(score1, 4), round(score2, 4)]


# Задание 4
# мы обучили нашу первую модель, но почему мы используем заполнение пропусков 0? Давай попробуем заполнить
# пропуски средним значением по каждому отдельному признаку и выполнить задание 3 еще раз. Среднее посчитать
# по той выборке, которая используется для определенного действия (если обучение модели - то по обучающей,
# если тестирование модели - то по тестовой). Функцию для этого задания, назовем `fit_second_model`.
# Устно проанализируйте изменения качества модели, стала модель лучше? Если да, то почему? Этот анализ сделать устно,
# ответы на эти вопросы сдавать автоматизированной системе не требуется.
def fit_second_model(x, y, x_test, y_test):
    columns = x.columns
    for column in columns:
        mean = x[column].mean()
        x = x.fillna(value={column: mean})
    columns = x_test.columns
    for column in columns:
        mean = x_test[column].mean()
        x_test = x_test.fillna(value={column: mean})
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=1, shuffle=True)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    score1 = roc_auc_score(y_valid, y_pred)
    y_test_pred = model.predict(x_test)
    score2 = roc_auc_score(y_test_pred, y_test)
    return [round(score1, 4), round(score2, 4)]


# Задание 5
# В задании на линейную регрессию, мы говорили, что среднее - статистика, которая сильно зависит от выбросов,
# она ориентируется на выбросы. А вот медиана - статистика, которая более устойчива к выбросам.
# Обработаем пропуски медианой, и выполним задание 3. Функцию назовем `fit_third_model`.
def fit_third_model(x, y, x_test, y_test):
    columns = x.columns
    for column in columns:
        median = x[column].median()
        x = x.fillna(value={column: median})
    columns = x_test.columns
    for column in columns:
        median = x_test[column].median()
        x_test = x_test.fillna(value={column: median})
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=1, shuffle=True)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    score1 = roc_auc_score(y_valid, y_pred)
    y_test_pred = model.predict(x_test)
    score2 = roc_auc_score(y_test_pred, y_test)
    return [round(score1, 4), round(score2, 4)]


# Задание 6
# Линейные модели сильно зависят от масштаба признаков. Наши данные содержат признаки в разном масштабе.
# Давай попробуем отмасштабировать данные с помощью `StandardScaler`.
# Для удобства, можно создать единый пайплайн, который и предобработает данные, и обучит модель логистической регрессии.
# Выполнить задание 3 - функция `fit_fourth_model` и задание 4 - функция `fit_fifth_model`.
def fit_fourth_model(x, y, x_test, y_test):
    x = x.fillna(0)
    x_scaled_array = StandardScaler().fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled_array, columns=x.columns)

    x_test = x_test.fillna(0)
    x_train, x_valid, y_train, y_valid = train_test_split(x_scaled, y, test_size=0.3, random_state=1, shuffle=True)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    score1 = roc_auc_score(y_valid, y_pred)
    y_test_pred = model.predict(x_test)
    score2 = roc_auc_score(y_test_pred, y_test)
    return [round(score1, 4), round(score2, 4)]


def fit_fifth_model(x, y, x_test, y_test):
    columns = x.columns
    for column in columns:
        mean = x[column].mean()
        x = x.fillna(value={column: mean})
    x_scaled_array = StandardScaler().fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled_array, columns=x.columns)

    columns = x_test.columns
    for column in columns:
        mean = x_test[column].mean()
        x_test = x_test.fillna(value={column: mean})
    x_train, x_valid, y_train, y_valid = train_test_split(x_scaled, y, test_size=0.3, random_state=1, shuffle=True)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    score1 = roc_auc_score(y_valid, y_pred)
    y_test_pred = model.predict(x_test)
    score2 = roc_auc_score(y_test_pred, y_test)
    return [round(score1, 4), round(score2, 4)]