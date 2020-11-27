import math
import numpy


# Задание 1:
# посчитать значений производной функции $\cos(x) + 0.05x^3 + \log_2{x^2}$ в точке $x = 10$.
# Ответ округлить до 2-го знака.
def function1(x):
    return math.cos(x) + 0.05 * (x**3) + math.log(x**2, 2)


def derivation(x, function):
    dx = 0.0000001
    df = function(x + dx) - function(x)
    result = df / dx
    return round(result, 2)


# Задание 2:
# посчитать значение градиента функции $x_1^2\cos(x_2) + 0.05x_2^3 + 3x_1^3\log_2{x_2^2}$ в точке $(10, 1)$.
def function2(point):
    x = point[0]
    y = point[1]
    return (x**2) * math.cos(y) + 0.05 * (x**3) + 3*(x**3) * math.log(y**2, 2)


def gradient_internal(point, function):
    x = point[0]
    y = point[1]
    fxy = function([x, y])

    dx = 0.00001
    df_dx = (function([x + dx, y]) - fxy) / dx

    dy = 0.00001
    df_dy = (function([x, y + dy]) - fxy) / dy

    return [df_dx, df_dy]


def gradient(point, function):
    g = gradient_internal(point, function)
    return [round(g[0], 2), round(g[1], 2)]


# Задание 3:
# найти точку минимуму для функции $\cos(x) + 0.05x^3 + \log_2{x^2}$.
# Зафиксировать параметр $\epsilon = 0.001$, начальное значение принять равным 10.
# Выполнить 50 итераций градиентного спуска. Ответ округлить до второго знака;
def function3(x):
    return math.cos(x) + 0.05 * (x**3) + math.log(x**2, 2)


def gradient_optimization_one_dim(function):
    x = 10
    dx = 0.00001
    epsilon = 0.001
    for step in range(50):
        df = (function(x + dx) - function(x)) / dx
        x = x - epsilon * df
    return round(x, 2)


# Задание 4:
# найти точку минимуму для функции $x_1^2\cos(x_2) + 0.05x_2^3 + 3x_1^3\log_2{x_2^2}$.
# Зафиксировать параметр $\epsilon = 0.001$, начальные значения весов принять равным [4, 10].
# Выполнить 50 итераций градиентного спуска. Ответ округлить до второго знака
def function4(point):
    x = point[0]
    y = point[1]
    xx = x**2
    yy = y**2
    return xx * math.cos(y) + 0.05 * (yy * y) + 3 * (xx * x) * math.log(yy, 2)


def gradient_optimization_multi_dim(function):
    x = 4
    y = 10
    epsilon = 0.001
    for step in range(50):
        g = gradient([x, y], function)
        x = x - round(epsilon * g[0], 2)
        y = y - round(epsilon * g[1], 2)
    x = round(x, 2)
    y = round(y, 2)
    return [x, y]


print(derivation(10, function1))
print(gradient([10, 1], function2))
print(gradient_optimization_one_dim(function3))
print(gradient_optimization_multi_dim(function4))

