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
    """Один поток - один воркер"""
    x, count, delta_x = args
    start = time.perf_counter()
    local_sum = 0
    for i in range(count):
        local_sum += get_plank_area(x, delta_x)
        x += delta_x
    end = time.perf_counter()
    return local_sum, (end - start)


if __name__ == "__main__":
    analytical_result = - (4 * math.cos(10) - 4 * math.cos(1) - 99) / 2  # ~52.2587
    LEFT_BORDER = 2
    RIGHT_BORDER = 20

    N = 32_000_000  # Количество "досочек"
    delta_x = (RIGHT_BORDER - LEFT_BORDER) / N

    # Сначала замерим время для одного потока (T1)
    print(f"Замер времени для одного процесса...")
    s_start = time.perf_counter()
    single_result, single_time = worker((LEFT_BORDER, N, delta_x))
    s_end = time.perf_counter()
    T1 = s_end - s_start
    print(f"T1 (1 процесс) = {T1:.6f} sec")

    M_values = []
    real_times = []
    ideal_times = []
    print(f"Замер времени для M потоков от 1 до 24:")
    for M in range(1, 24 + 1):  # от 1 до 24 потоков
        if M == 1:
            M_values.append(1)
            real_times.append(single_time)  # берём время одного воркера
            ideal_times.append(T1 / M)
            print(f"M={M}, Real={single_time:.6f}, Ideal={T1 / M:.6f}, "
                  f"Error={abs(analytical_result - single_result):.6e}")
            continue

        chunk_size = N // M
        tasks = []
        for i in range(M):
            start_idx = LEFT_BORDER + i * chunk_size * delta_x
            tasks.append((start_idx, chunk_size, delta_x))

        with multiprocessing.Pool(processes=M) as pool:
            results = pool.map(worker, tasks)

        total_result = sum(r[0] for r in results)
        max_time = max(r[1] for r in results)  # берём максимальное время воркера

        M_values.append(M)
        real_times.append(max_time)
        ideal_times.append(T1 / M)

        print(f"M={M}, Real={max_time:.6f}, Ideal={T1/M:.6f}, "
              f"Error={abs(analytical_result - total_result):.6e}")

    print("Расчет площадей для N с шагом 24:")
    N_values = list(range(24, 24 * 24, 24))
    errors_24 = []
    for n in N_values:
        delta_x = (RIGHT_BORDER - LEFT_BORDER) / n
        chunk_size = n // 24
        tasks = []
        for i in range(24):
            start_idx = LEFT_BORDER + i * chunk_size * delta_x
            tasks.append((start_idx, chunk_size, delta_x))

        with multiprocessing.Pool(processes=24) as pool:
            results = pool.map(worker, tasks)

        total_result = sum(r[0] for r in results)
        error = analytical_result - total_result
        errors_24.append(error)
        print(f"N={n}, Error={error:.6e}")

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Верхний график: идеальное vs реальное время
    axs[0].plot(M_values, ideal_times, "bo-", label="Идеальное время (T1/M)")
    axs[0].plot(M_values, real_times, "ro-", label="Условное реальное время")
    axs[0].set_xlabel("Число процессов (M)")
    axs[0].set_ylabel("Время, сек")
    axs[0].set_title(f"Сравнение идеального и условного времени (N={N})")
    axs[0].legend()
    axs[0].grid(True)

    # Нижний график: фиксируем M=24, меняем N, считаем погрешность
    axs[1].plot(N_values, errors_24, "go-", label="Погрешность (24 процесса)")
    axs[1].set_xlabel("Количество досочек (N)")
    axs[1].set_ylabel("Ошибка интеграла")
    axs[1].set_title("Зависимость ошибки от N при M=24")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
