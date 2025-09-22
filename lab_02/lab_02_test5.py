import time
import multiprocessing
import numpy as np

# Параметры тора с неровностью
R = 3
r = 1.5
k = 0.6
n = 5

# Ограничения параллелепипеда
X_MIN, X_MAX = -5, 5
Y_MIN, Y_MAX = -5.5, 5.5
Z_MIN, Z_MAX = -2, 2

# Объем параллелепипеда (для оценки объема тела)
V_parallelepiped = (X_MAX - X_MIN) * (Y_MAX - Y_MIN) * (Z_MAX - Z_MIN)

# Количество точек
N = 1_000_000  # можно увеличить для точности


def is_dot_inside_shape_boundaries(x, y, z):
    """
    Проверка попадания точки в тор с неровностью через радиусные границы.
    """
    theta = np.arctan2(y, x)
    rho = np.sqrt(x ** 2 + y ** 2)
    rho_min = R - r + k * np.sin(n * theta)
    rho_max = R + r + k * np.sin(n * theta)
    return rho_min <= rho <= rho_max and abs(z) <= r


def worker_generate_points(point_amount):
    """
    Генерация случайных точек и проверка попадания в тело.
    """
    count_inside = 0
    start = time.perf_counter()
    for _ in range(point_amount):
        x = np.random.uniform(X_MIN, X_MAX)
        y = np.random.uniform(Y_MIN, Y_MAX)
        z = np.random.uniform(Z_MIN, Z_MAX)
        if is_dot_inside_shape_boundaries(x, y, z):
            count_inside += 1
    end = time.perf_counter()
    return count_inside, (end - start)


if __name__ == "__main__":
    print(f"Запуск Monte Carlo для {N} точек...")

    # Можно использовать один процесс
    single_result, single_time = worker_generate_points(N)
    estimated_volume = V_parallelepiped * (single_result / N)

    print(f"T1 (1 процесс) = {single_time:.6f} sec")
    print(f"Количество точек внутри: {single_result}")
    print(f"Оценка объема тора с неровностью: {estimated_volume:.6f}")
