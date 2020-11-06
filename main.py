import math


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
def function2(x, y):
    return (x**2) * math.cos(y) + 0.05 * (y**3) + 3*(x**3) * math.log(y**2, 2)


def gradient(point, function):
    dx = 0.0000001
    dy = 0.0000001
    df_dx = (function(point[0] + dx, point[1]) - function(point[0], point[1])) / dx
    df_dy = (function(point[0], point[1] + dy) - function(point[0], point[1])) / dy
    result = math.sqrt(df_dx**2 + df_dy**2)
    return round(result, 2)


# Задание 3:
# найти точку минимуму для функции $\cos(x) + 0.05x^3 + \log_2{x^2}$.
# Зафиксировать параметр $\epsilon = 0.001$, начальное значение принять равным 10.
# Выполнить 50 итераций градиентного спуска. Ответ округлить до второго знака;
def function3(x):
    return math.cos(x) + 0.05 * (x**3) + math.log(x**2, 2)


def gradient_optimization_one_dim(function):
    x = 10
    dx = 0.000001
    epsilon = 0.001
    for step in range(50):
        df = (function(x + dx) - function(x)) / dx
        x = x - epsilon * df
    return round(x, 2)


# Задание 4:
# найти точку минимуму для функции $x_1^2\cos(x_2) + 0.05x_2^3 + 3x_1^3\log_2{x_2^2}$.
# Зафиксировать параметр $\epsilon = 0.001$, начальные значения весов принять равным [4, 10].
# Выполнить 50 итераций градиентного спуска. Ответ округлить до второго знака
def function4(x, y):
    return (x**2) * math.cos(y) + 0.05 * (y**3) + 3 * (x**3) * math.log(y**2, 2)


def gradient_optimization_multi_dim(function):
    x = 4
    y = 10
    dx = 0.000001
    dy = 0.000001
    epsilon = 0.001
    for step in range(50):
        df_dx = (function(x + dx, y) - function(x, y)) / dx
        df_dy = (function(x, y + dy) - function(x, y)) / dy
        x = x - epsilon * df_dx
        y = y - epsilon * df_dy
    return [x, y]


print(derivation(10, function1))
print(gradient([10, 1], function2))
print(gradient_optimization_one_dim(function3))
print(gradient_optimization_multi_dim(function4))

