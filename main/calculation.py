import math
import numpy as np

azimuth1 = math.radians(-110)
azimuth2 = math.radians(-110)

# Задаем внутренние параметры камеры (матрица K)
fx_1 = 35e-3
fy_1 = fx_1
cx_1 = 660
cy_1 = 760
cz_1 = 35

fx_2 = 35e-3
fy_2 = fx_2
cx_2 = 810
cy_2 = 760
cz_2 = 35

# Задаем 2D координаты проекций
u1 = 224
v1 = 674
x1 = np.array([u1, v1, 1])

u2 = 224
v2 = 674
x2 = np.array([u2, v2, 1])


def init_k_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


def init_r_matrix(azimuth):
    return np.array([[math.cos(azimuth), -math.sin(azimuth), 0],
                     [math.sin(azimuth), math.cos(azimuth), 0],
                     [0, 0, 1]])


K_1 = init_k_matrix(fx_1, fy_1, cx_1, cy_1)

K_2 = init_k_matrix(fx_2, fy_2, cx_2, cy_2)

# Задаем матрицу вращения R и вектор положения камеры c
R1 = init_r_matrix(azimuth1)
R2 = init_r_matrix(azimuth2)

c_1 = np.array([cx_1, cy_1, cz_1])
c_2 = np.array([cx_2, cy_2, cz_2])

# Вычисляем матрицы проекции P1 и P2
P1 = K_1 @ np.hstack((R1, -np.dot(R1, c_1)[:, None]))
P2 = K_2 @ np.hstack((R2, -np.dot(R2, c_2)[:, None]))


# Решаем систему линейных уравнений
A = np.vstack((P1, P2))
b = np.hstack((x1, x2))
X = np.linalg.lstsq(A, b, rcond=None)[0]

print("Координаты 3D точки:", X[0] / X[3], X[1] / X[3], X[2] / X[3])
