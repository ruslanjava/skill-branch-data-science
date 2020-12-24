import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Задание 1.
# Для начала, попробуем разбить данные на обучающую часть и валидационную часть в соотношении 70 / 30 и
# предварительным перемешиванием. Параметр `random_state` зафиксировать 42.
# Назовите функцию `split_data_into_two_samples`, которая принимает полный датафрейм,
# а возвращает 2 датафрейма: для обучения и для валидации.
def split_data_into_two_samples(x):
    x_train, x_test = train_test_split(x, test_size=0.3, random_state=42)
    return [x_train, x_test]


#df = pd.read_csv('train.csv')
#split = split_data_into_two_samples(df)
#print(split[0].head())
#print(split[1].head())
