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
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=1, shuffle=True)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    score1 = roc_auc_score(y_valid, y_pred)
    y_test_pred = model.predict(x_test)
    score2 = roc_auc_score(y_test_pred, y_test)
    return [round(score1, 4), round(score2, 4)]