import numpy as np
import pandas as pd


# Задание 1
# написать функцию `calculate_data_shape`, которая принимает на вход датафрейм `X` и возвращает его размерность
def calculate_data_shape(x):
    return x.shape


# Задание 2
# написать функцию `take_columns`, которая принимает на вход датафрейм `X` и возвращает название его столбцов.
def take_columns(x):
    return x.columns


# Задание 3
# написать функцию `calculate_target_ratio`, которая принимает на вход датафрейм `X` и название целевой переменной
# `target_name` - возвращает среднее значение целевой переменной. Округлить выходное значение до 2-го знака внутри
# функции.
def calculate_target_ratio(x, target_name):
    mean = x[target_name].mean()
    return round(mean, 2)


# Задание 4
# написать функцию `calculate_data_dtypes`, которая принимает на вход датафрейм `X` и возвращает количество числовых
# признаков и категориальных признаков. Категориальные признаки имеют тип `object`.
def calculate_data_dtypes(x):
    ints = x.select_dtypes(include=['float64', 'int64']).dtypes.count()
    objects = x.select_dtypes(include=['object']).dtypes.count()
    return [ints, objects]


# Задание 5
# написать функцию `calculate_cheap_apartment`, которая принимает на вход датафрейм `X` и возвращает количество квартир,
# стоимость которых меньше 1 млн. рублей.
def calculate_cheap_apartment(x):
    return x[x['price_doc'] < 1000000]['price_doc'].count()


# Задание 6
# написать функцию `calculate_squad_in_cheap_apartment`, которая принимает на вход датафрейм `X` и возвращает среднюю
# площадь квартир, стоимость которых меньше 1 млн .рублей. Признак, отвечающий за площадь - `full_sq`. Ответ округлить
# целого значения.
def calculate_squad_in_cheap_apartment(x):
    return x[x['price_doc'] < 1000000]['full_sq'].mean()


# Задание 7
# написать функцию `calculate_mean_price_in_new_housing`, которая принимает на вход датафрейм `X` и возвращает
# среднюю стоимость трехкомнатных квартир в доме, который не страше 2010 года. Ответ округлить до целого значения.
def calculate_mean_price_in_new_housing(x):
    mean = x[(x['num_room'] == 3) & (x['build_year'] >= 2010)]['price_doc'].mean()
    return round(mean)


df = pd.read_csv('housing_market.csv')
print(calculate_data_dtypes(df))
print(calculate_cheap_apartment(df))
print(calculate_squad_in_cheap_apartment(df))
print(calculate_mean_price_in_new_housing(df))
