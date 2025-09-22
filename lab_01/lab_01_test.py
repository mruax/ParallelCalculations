import math
import time
import multiprocessing
import matplotlib.pyplot as plt


def function(x):
    return math.sin(x / 2) + 0.25 * x


def get_plank_area(x, delta_x):
    y_n = function(x)
    y_n2 = function(x + delta_x)
    return delta_x * min(y_n, y_n2) + delta_x * 0.5 * (max(y_n, y_n2) - min(y_n, y_n2))


def worker(args):
    """Один процесс считает свою часть интеграла"""
    x, count, delta_x = args
    local_sum = 0
    for i in range(count):
        local_sum += get_plank_area(x, delta_x)
        x += delta_x
    return local_sum


if __name__ == "__main__":
    analytical_result = - (4 * math.cos(10) - 4 * math.cos(1) - 99) / 2  # ~52.2587
    LEFT_BORDER = 2
    RIGHT_BORDER = 20

    N = 32000000  # Количество "досочек"
    delta_x = (RIGHT_BORDER - LEFT_BORDER) / N

    # Сначала замерим время для одного потока (T1)
    start = time.perf_counter()
    single_result = worker((LEFT_BORDER, N, delta_x))
    end = time.perf_counter()
    T1 = end - start
    print(f"T1 (1 процесс) = {T1:.6f} sec")

    M_values = []
    real_times = []
    ideal_times = []

    for M in range(1, 24 + 1):  # от 1 до 24 потоков
        if M == 1:
            M_values.append(1)
            real_times.append(T1)
            ideal_times.append(T1 / M)
            print(f"M={M}, Real={T1:.6f}, Ideal={T1 / M:.6f}, "
                  f"Error={abs(analytical_result - single_result):.6e}")
            continue

        chunk_size = N // M
        tasks = []
        for i in range(M):
            start_idx = LEFT_BORDER + i * chunk_size * delta_x
            tasks.append((start_idx, chunk_size, delta_x))

        start = time.perf_counter()
        with multiprocessing.Pool(processes=M) as pool:
            results = pool.map(worker, tasks)
        end = time.perf_counter()

        total_result = sum(results)
        total_time = end - start  # чистое время работы пула

        M_values.append(M)
        real_times.append(total_time)
        ideal_times.append(T1 / M)

        print(f"M={M}, Real={total_time:.6f}, Ideal={T1/M:.6f}, "
              f"Error={abs(analytical_result - total_result):.6e}")

    # Рисуем график
    plt.figure(figsize=(10, 6))
    plt.plot(M_values, ideal_times, "bo-", label="Идеальное время (T1/M)")
    plt.plot(M_values, real_times, "ro-", label="Реальное время (pool)")
    plt.xlabel("Число процессов (M)")
    plt.ylabel("Время, сек")
    plt.title(f"Сравнение идеального и реального времени (N={N})")
    plt.legend()
    plt.grid(True)
    plt.show()
