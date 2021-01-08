import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Задание 1.
# Для начала, попробуем разбить данные на обучающую часть и валидационную часть в соотношении 70 / 30 и
# предварительным перемешиванием. Параметр `random_state` зафиксировать 42.
# Назовите функцию `split_data_into_two_samples`, которая принимает полный датафрейм,
# а возвращает 2 датафрейма: для обучения и для валидации.
def split_data_into_two_samples(x):
    x_train, x_test = train_test_split(x, test_size=0.3, random_state=42)
    return [x_train, x_test]


# Задание 2.
# продолжим выполнение предварительной подготовки данных: в данных много категориальных признаков
# (они представлены типами `object`), пока мы с ними работать не умеем, поэтому удалим их из датафрейма.
# Кроме того, для обучения нам не нужна целевая переменная, ее нужно выделить в отдельный вектор (`price_doc`).
# Написать функцию `prepare_data`, которая принимает датафрейм, удаляет категориальные признаки, удаляет `id`,
# и выделяет целевую переменную в отдельный вектор.
# Кроме того, некоторые признаки содержат пропуски, требуется удалить такие признаки.
# Функция должна возвращать подготовленную матрицу признаков и вектор с целевой переменной.
def prepare_data(x):
    target_vector = x['price_doc']
    objects = x.select_dtypes(include=['object']).columns
    filtered = x.drop(columns=objects).drop(columns=["id"]).drop(columns=['price_doc']).dropna(axis=1)
    return [filtered, target_vector]


# Задание 3
# Перед обучением линейной модели также необходимо масштабировать признаки.
# Для этого мы можем использовать `MinMaxScaler` или `StandardScaler`.
# Написать функцию, которая принимает на вход датафрейм и трансформем,
# а возвращает датафрейм с отмасштабированными признаками.
def scale_data(x, transformer):
    return transformer.fit_transform(x)


# Задание 4
# объединить задание 2 и 3 в единую функцию `prepare_data_for_model`,
# функция принимает датафрейм и трансформер для масштабирования,
# возвращает данные в формате задания 3 и вектор целевой переменной.
def prepare_data_for_model(x, transformer):
    x_train, x_test = prepare_data(x)
    x_scaled_array = scale_data(x_train, transformer)
    x_train_scaled = pd.DataFrame(x_scaled_array, columns=x_train.columns)
    return [x_train_scaled, x_test]


# Задание 5.
# разбить данные на обучение / валидацию в соответствии с заданием 1.
# Обучить линейную модель (`LinearRegression`) на данных из обучения.
# При подготовке данных использовать функцию из задания 4,
# в качестве трансформера для преобразования данных использовать - `StandardScaler`.
# Создать функцию `fit_first_linear_model`, которая принимает на вход `x_train` и `y_train`, а возвращает модельку.
def fit_first_linear_model(x_train, y_train):
    x_train_scaled = scale_data(x_train, StandardScaler())
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    return model


# Задание 6.
# выполнить задание 5, но с использованием `MinMaxScaler`.
def fit_first_linear_model_2(x_train, y_train):
    x_train_scaled = scale_data(x_train, MinMaxScaler())
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    return model


# Задание 7
# написать функцию для оценки качества модели - `evaluate_model`, которая принимает на вход обученную модель,
# выборку для построения прогнозов и вектор истинных ответов для выборки. Внутри функции вычислить метрики `MSE`,
# `MAE`, `R2`, вернуть значения метрик, округленных до 2-го знака.
# Для построения / оценки качества использовать разбиение из задания 1
def evaluate_model(linreg, x_test, y_test):
    y_pred = linreg.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return [round(mse, 2), round(mae, 2), round(r2, 2)]


# Задание 8
# написать функцию, которая принимает на вход обученную модель и список названий признаков,
# и создает датафрейм с названием признака и абсолютным значением веса признака.
# Датафрейм отсортировать по убыванию важности признаков и вернуть.
# Назвать функцию `calculate_model_weights`. Для удобства, колонки датафрейма назвать `features` и `weights`.
def calculate_model_weights(linreg, names):
    df = pd.DataFrame({
        'features': names,
        'weights': linreg.coef_
    })
    df.sort_values(by=['weights'])
    return df




