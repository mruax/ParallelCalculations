import time
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

# Global variables
X_MIN, X_MAX = -4.95, 4.95
Y_MIN, Y_MAX = -5.13, 5.13
Z_MIN, Z_MAX = -3.34, 3.34


def is_dot_inside_shape(x, y, z):
    rho = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / rho) if rho != 0 else 0.0
    phi = np.arctan2(y, x)
    factor = 3 + 1.5 * np.cos(phi) + 0.6 * np.sin(5 * theta)
    r_max = factor

    # Если расстояние rho меньше или равно r_max — точка внутри фигуры
    return rho <= r_max


def worker_generate_points(point_amount):
    """
    Генерирует N точек в пределах заданного параллелепипеда и проверяет их на вхождение в фигуру.
    :param N: int, количество точек
    :return: int, количество точек внутри фигуры
    """
    count_inside = 0
    start = time.perf_counter()
    for _ in range(point_amount):
        x = np.round(np.random.uniform(X_MIN, X_MAX), 12)
        y = np.round(np.random.uniform(Y_MIN, Y_MAX), 12)
        z = np.round(np.random.uniform(Z_MIN, Z_MAX), 12)
        if is_dot_inside_shape(x, y, z):
            count_inside += 1
    end = time.perf_counter()
    return count_inside, (end - start)


if __name__ == "__main__":
    analytical_result = 148.157509543295
    V_parallelepiped = (abs(X_MAX) + abs(X_MIN)) * (abs(Y_MAX) + abs(Y_MIN)) * (abs(Z_MAX) + abs(Z_MIN))

    R = 3
    r = 1.5
    k = 0.6
    n = 5

    N = 1_000_000  # Количество точек

    # Сначала замерим время для одного потока (T1)
    print(f"Замер времени для одного процесса...")
    single_result, single_time = worker_generate_points(N)
    print(f"T1 (1 процесс) = {single_time:.6f} sec")
    print(V_parallelepiped * (single_result / N))

    M_values = []
    real_times = []
    ideal_times = []
    print(f"\nЗамер времени для M процессов от 1 до 24:")

    for M in range(1, 25):  # от 1 до 24 процессов
        if M == 1:
            M_values.append(1)
            real_times.append(single_time)  # берём время одного воркера
            ideal_times.append(single_time / M)
            print(f"M={M}, Real={single_time:.6f}, Ideal={single_time / M:.6f}, "
                  f"Result={V_parallelepiped * (single_result / N):.6f}")
            continue

        # делим количество точек на чанки
        chunk_size = N // M
        task_sizes = [chunk_size] * M
        task_sizes[-1] += N % M  # остаток в последний процесс

        with multiprocessing.Pool(processes=M) as pool:
            results = pool.map(worker_generate_points, task_sizes)

        total_inside = sum(r[0] for r in results)
        max_time = max(r[1] for r in results)  # берём самое долгое время

        M_values.append(M)
        real_times.append(max_time)
        ideal_times.append(single_time / M)

        print(f"M={M}, Real={max_time:.6f}, Ideal={single_time / M:.6f}, "
              f"Result={V_parallelepiped * (total_inside / N):.6f}")

    # ---------------- ГРАФИК №1 ----------------
    plt.figure(figsize=(10, 5))
    plt.plot(M_values, ideal_times, "bo-", label="Идеальное время (T1/M)")
    plt.plot(M_values, real_times, "ro-", label="Реальное время (max worker)")
    plt.xlabel("Число процессов (M)")
    plt.ylabel("Время, сек")
    plt.title(f"Сравнение идеального и реального времени (N={N})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------- ГРАФИК №2 ----------------
    print("\nРасчет ошибки для разных N при M=24:")
    N_values = list(range(100_000, N+1, 100))
    errors_24 = []

    for n_points in N_values:
        chunk_size = n_points // 24
        task_sizes = [chunk_size] * 24
        task_sizes[-1] += n_points % 24

        with multiprocessing.Pool(processes=24) as pool:
            results = pool.map(worker_generate_points, task_sizes)

        total_inside = sum(r[0] for r in results)
        volume_est = V_parallelepiped * (total_inside / n_points)
        error = abs(analytical_result - volume_est)
        errors_24.append(error)

        print(f"N={n_points}, Volume={volume_est:.6f}, Error={error:.6e}")

    plt.figure(figsize=(10, 5))
    plt.plot(N_values, errors_24, "go-", label="Ошибка при M=24")
    plt.xlabel("Количество точек (N)")
    plt.ylabel("Ошибка интеграла")
    plt.title("Зависимость ошибки от N при M=24")
    plt.legend()
    plt.grid(True)
    plt.show()
