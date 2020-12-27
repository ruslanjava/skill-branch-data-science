import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


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
    objects = x.select_dtypes(include=['object']).columns
    filtered = x.drop(columns=objects)
    filtered = filtered.drop(columns=["id"])
    filtered = filtered.dropna()
    target_vector = filtered['price_doc']
    matrix = filtered.drop(columns=['price_doc'])
    return [matrix, target_vector]


# Задание 3
# Перед обучением линейной модели также необходимо масштабировать признаки.
# Для этого мы можем использовать `MinMaxScaler` или `StandardScaler`.
# Написать функцию, которая принимает на вход датафрейм и трансформем,
# а возвращает датафрейм с отмасштабированными признаками.
def fit_first_linear_model(x, transformer):
    transformer.fit(x)
    return transformer.transform(x)


#df = pd.read_csv('housing_market.csv')
#df = prepare_data(df)
#print(df[0])
#print(fit_first_linear_model(df[0], MinMaxScaler()))
#print(fit_first_linear_model(df[0], StandardScaler()))
