import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import random as rnd
import copy


def my_chi_square(f_obs, f_exp, ddof=0):
    f_exp = np.asanyarray(f_exp)
    terms = (f_obs.astype(np.float64) - f_exp) ** 2 / f_exp
    stat = terms.sum()
    num_obs = ss.stats._count(terms)
    ddof = np.asarray(ddof)
    p = ss.chi2.sf(stat, num_obs - 1 - ddof)
    return stat, p


expected_value = np.array([rnd.random() * 10 - 5, rnd.random() * 10 - 5, rnd.random() * 10 - 5])
print(expected_value)

while True:
    x11 = rnd.random() * 5 + 5
    x12 = rnd.random() * 20 - 10
    x13 = rnd.random() * 20 - 10
    x22 = rnd.random() * 5 + 5
    x23 = rnd.random() * 20 - 10
    x33 = rnd.random() * 5 + 5
    covariance = np.array([[x11, x12, x13], [x12, x22, x23], [x13, x23, x33]])

    if abs(x12) < 0.1 or abs(x13) < 0.1 or abs(x23) < 0.1:
        continue

    if abs(np.linalg.det(covariance)) < 0.1:
        continue

    flag = False
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(covariance[i][j] / (covariance[i][i] ** (1 / 2)) * (covariance[j][j] ** (1 / 2))) > 0.97:
                flag = True
                break
    if flag:
        continue

    if np.all(np.linalg.eigvals(covariance) > 0):
        break

error_squares = np.array([])

print("Пункы с 1 по 4", '\n')

for i in [0, 1, 2, 3, 99]:
    print("Номер итерации: ", i + 1)
    first_set = ss.multivariate_normal.rvs(expected_value, covariance, 101)

    number = rnd.randint(0, 100)
    saved_observation = first_set[number]
    dataset = np.delete(first_set, number, 0)

    dataset = np.transpose(dataset)
    Y = dataset[0]
    X = np.delete(dataset, 0, 0)
    X = np.transpose(X)

    X = np.c_[X, np.ones(X.shape[0])]

    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    residuals = Y - X.dot(B)
    print("Коэффициенты линейной регрессии:")
    print(B)
    print("Ожидаемое значение вектора ошибок:")
    print(np.mean(residuals), '\n')
    print("Изменение вектора ошибок:")
    print(np.var(residuals, ddof=1), '\n')

    print("Сохраненное наблюдение:")
    print(saved_observation, '\n')
    y = saved_observation[0]

    saved_observation = np.delete(saved_observation, 0, 0)
    saved_observation = np.append(saved_observation, 1)
    prediction = saved_observation.dot(B)
    print("Предположительный x1:")
    print(prediction, '\n')

    error_squares = np.append(error_squares, (prediction - y) * (prediction - y))
    print("Квадрат ошибки:")
    print((prediction - y) * (prediction - y), '\n\n')
print("Усредненный квадрат ошибки:")
print(np.mean(error_squares), '\n\n')

x1 = dataset[0]
x2 = dataset[1]
plt.scatter(x1, x2)
plt.title("Проекция линейной регрессии на диаграмму рассеяния X1-X2")
plt.xlabel("X1")
plt.ylabel("X2")

x = np.linspace(np.amin(x1), np.amax(x1), 20)
y = np.linspace(np.amin(x2), np.amax(x2), 20)
X1, Y1 = np.meshgrid(x, y)
h = B[0] * X1 + B[1] * Y1 + B[2]
cs = plt.contour(X1, Y1, h, levels=15)
plt.show()

print("Строим регрессионную модель без константы:")
Y = dataset[0]
X = np.delete(dataset, 0, 0)
X = np.transpose(X)
b = np.linalg.lstsq(X, Y, rcond=None)[0]
resid = Y - X.dot(b)
print("Коэффициенты линейной регрессии:")
print(b, '\n')
print("Ожидаемое значение вектора ошибок:")
print(np.mean(resid), '\n')
print("Изменение вектора ошибок:")
print(np.var(resid, ddof=1), '\n\n\n')

print("Хи-квадрат критерий Пирсона:")
print("Хи-квадрат критерий Пирсона для k от 3 до 100")

for k in range(3, 100):
    print('k = ', k)
    res = ss.relfreq(residuals, numbins=k, defaultreallimits=(np.amin(residuals), np.amax(residuals)))
    observed = res.frequency * len(residuals)

    mu = np.mean(residuals)
    sigma = np.std(residuals, ddof=1)
    inter = np.linspace(np.amin(residuals), np.amax(residuals), k + 1)
    expected = np.array([])
    for i in range(k):
        n = (ss.norm.cdf(inter[i + 1], mu, sigma) - ss.norm.cdf(inter[i], mu, sigma)) * len(residuals)
        expected = np.append(expected, n)
    print('Встроенная функция хи-квадрат:')
    print(ss.chisquare(observed, expected, ddof=k - 2), '\n')
    print("Наш хи-квадрат:")
    stat, p_val = my_chi_square(observed, expected, ddof=k - 2)
    print("statistic=", stat, "p-value=", p_val, '\n\n')

print('Переобучение')
M = 5
N = 20
print("M=", M)
print("N=", N)

while True:
    x11 = rnd.random() * 5 + 5
    x12 = rnd.random() * 20 - 10
    x22 = rnd.random() * 5 + 5
    if abs(x12) < 0.1:
        continue
    covariance = np.array([[x11, x12], [x12, x22]])
    if abs(np.linalg.det(covariance)) < 0.1:
        continue
    if np.all(np.linalg.eigvals(covariance) > 0):
        break

print('\n\n')
error_squares = np.zeros((3, M))

first_data = np.zeros((N, 2))
coefs_1 = np.array([])
coefs_2 = np.array([])
coefs_3 = np.array([])

for i in range(M):
    print()
    print("M =", i + 1)
    data = ss.multivariate_normal.rvs(np.array([0, 0]), covariance, N)

    if i == 0:
        first_data = copy.deepcopy(data)

    n = N // 10
    saved = np.zeros((n, 2))
    for j in range(n):
        integer = rnd.randint(0, N - j - 1)
        saved[j] = data[integer]
        data = np.delete(data, integer, 0)

    data = np.transpose(data)
    Y = data[0]

    p = np.array([2, 5, 8])

    for m in range(3):
        print("P =", p[m])
        X = np.ones((p[m] + 1, N - n))
        X[1] = data[1]
        for k in range(2, p[m] + 1):
            X[k] = X[1] ** k
        X = np.transpose(X)

        B = np.linalg.lstsq(X, Y, rcond=None)[0]
        residuals = Y - X.dot(B)
        print('\n')
        print("Коэффициенты полиномиальной регрессии:")
        print(B)
        print('\n')

        if i == 0 and m == 0:
            coefs_1 = copy.deepcopy(B)
        if i == 0 and m == 1:
            coefs_2 = copy.deepcopy(B)
        if i == 0 and m == 2:
            coefs_3 = copy.deepcopy(B)

        error_squares_for_n = np.array([])
        for k in range(n):
            print("n =", k + 1, '\n')
            print("Сохраненное наблюдение:", '\n')
            print(saved[k])
            obs = copy.deepcopy(saved[k])
            y = obs[0]
            obs[0] = 1
            for l in range(2, p[m] + 1):
                obs = np.append(obs, obs[1] ** l)
            prediction = obs.dot(B)
            print("Предположительный x1")
            print(prediction, '\n')
            error_squares_for_n = np.append(error_squares_for_n, (prediction - y) * (prediction - y))
            print("Квадрат ошибки:")
            print((prediction - y) * (prediction - y), '\n\n')
        error_squares[m][i] = np.mean(error_squares_for_n)

print("Средняя усредненных ошибок:")
print("Для p = 2:")
print(np.mean(error_squares[0]), '\n')
print("Для p = 5:")
print(np.mean(error_squares[1]), '\n')
print("Для p = 8:")
print(np.mean(error_squares[2]), '\n')

first_data = np.transpose(first_data)
x1 = first_data[0]
x2 = first_data[1]
plt.scatter(x1, x2)
plt.title("Полиномиальная регрессия для X1 и X2 с p = 2")
plt.xlabel("X1")
plt.ylabel("X2")
x = np.linspace(np.amin(x1), np.amax(x1), 1000)
plt.plot(x, coefs_1[0] + coefs_1[1] * x + coefs_1[2] * x ** 2)
plt.show()

plt.scatter(x1, x2)
plt.title("Полиномиальная регрессия для X1 и X2 с p = 5")
plt.xlabel("X1")
plt.ylabel("X2")

plt.plot(x, coefs_2[0] + coefs_2[1] * x + coefs_2[2] * x ** 2 + coefs_2[2] * x ** 3 + coefs_2[4] * x ** 4 + coefs_2[
    5] * x ** 5)
plt.show()

plt.scatter(x1, x2)
plt.title("Полиномиальная регрессия для X1 и X2 с p = 8")
plt.xlabel("X1")
plt.ylabel("X2")

plt.plot(x, coefs_3[0] + coefs_3[1] * x + coefs_3[2] * x ** 2 + coefs_3[2] * x ** 3 + coefs_3[4] * x ** 4 + coefs_3[
    5] * x ** 5 + coefs_3[5] * x ** 6 + coefs_3[5] * x ** 7 + coefs_3[5] * x ** 8)
plt.show()
