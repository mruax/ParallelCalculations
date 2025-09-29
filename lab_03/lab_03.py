import heapq
import multiprocessing
import random
import time

import matplotlib.pyplot as plt
import numpy as np


def symmetric_random_matrix(n, seed=None) -> np.ndarray:
    """
    Возвращает n x n симметричную матрицу:
      - диагональ = 0
      - элементы a[i,j] = a[j,i] = случайный float в [-2, 2], округлённый до 12 знаков
    Параметр seed для воспроизводимости.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    # верхний треугольник без диагонали
    upper = rng.uniform(-2.0, 2.0, size=(n, n))
    # обнулим нижний треугольник и диагональ (сохраняем только strict upper)
    upper = np.triu(upper, k=1)
    # зеркалим в нижний треугольник
    mat = upper + upper.T
    # диагональ нули (на всякий случай)
    np.fill_diagonal(mat, 0.0)
    # округление до 12 знаков после запятой
    mat = np.round(mat, 12)
    return mat


def worker_range(start, end, N, Ad, B, k=4):
    """Обрабатывает диапазон голосований [start, end).
    Возвращает кучу из k минимальных (E, конфигурация) + время работы."""
    heap = []  # max-heap для k наименьших (через отрицательные E)
    t0 = time.perf_counter()

    for i in range(start, end):
        votes_list = list(map(int, bin(i)[2:].rjust(N, '0')))
        votes_list = list(map(lambda x: x if x == 1 else -1, votes_list))

        E = sum(Ad * Vd for Vd in votes_list)
        for n in range(N):
            for m in range(n + 1, N):
                E += B[n][m] * votes_list[n] * votes_list[m]

        if len(heap) < k:
            heapq.heappush(heap, (-E, votes_list))
        else:
            heapq.heappushpop(heap, (-E, votes_list))

    t1 = time.perf_counter()
    elapsed = t1 - t0

    return [(-x[0], x[1]) for x in heap], elapsed


if __name__ == "__main__":
    N = 22  # 22

    # Фиксированные коэффициенты
    Ad = -0.2                       # Воля депутата
    B = symmetric_random_matrix(N)  # Взаимоотношения депутатов

    total_votes = 2 ** N
    workers = multiprocessing.cpu_count()  # 24
    chunk_size = total_votes // workers
    ranges = [(i * chunk_size, (i + 1) * chunk_size if i < workers - 1 else total_votes, N, Ad, B)
              for i in range(workers)]

    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.starmap(worker_range, ranges)

        # объединяем минимумы
        combined_heap = []
        k = 4
        max_time = -1
        for part_min, t in results:
            max_time = max(t, max_time)
            for E, votes in part_min:
                if len(combined_heap) < k:
                    heapq.heappush(combined_heap, (-E, votes))
                else:
                    heapq.heappushpop(combined_heap, (-E, votes))

        min4 = sorted([(-x[0], x[1]) for x in combined_heap])

        print("4 минимальных E и конфигурации голосов:")
        for e, v in min4:
            print(f"E={e:.6f}, votes={v}")

        print(f"\nМаксимальное время среди воркеров: {max_time:.6f} sec")

    E1, E2, E3, E4 = min4[0][0], min4[1][0], min4[2][0], min4[3][0]
    V1, V2, V3, V4 = min4[0][1], min4[1][1], min4[2][1], min4[3][1]

    max_B_index = np.unravel_index(np.argmax(np.abs(B)), B.shape)  # matrix indices
    max_B_value = B[max_B_index]
    n, m = max_B_index[0], max_B_index[1]

    print("\nМатрица Bnm:")
    for row in B:
        print([f"{elem:.2f}".rjust(5, "+") for elem in row])

    T_values = np.arange(0.001, abs(E1), 0.005)
    mean_values = []

    for T in T_values:  # Хаос
        R1 = 1
        R2 = np.exp(-(E2 - E1) / T)
        R3 = np.exp(-(E3 - E1) / T)
        R4 = np.exp(-(E4 - E1) / T)

        Z = R1 + R2 + R3 + R4

        ro1 = R1 / Z
        ro2 = R2 / Z
        ro3 = R3 / Z
        ro4 = R4 / Z

        mean_value = (V1[n] * V1[m] * ro1 +
                      V2[n] * V2[m] * ro2 +
                      V3[n] * V3[m] * ro3 +
                      V4[n] * V4[m] * ro4)  # в долях
        mean_values.append(mean_value)

    # ---------------- ГРАФИК №1 ----------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(T_values, mean_values, color='blue', linewidth=2)

    ax.set_title("Зависимость mean_value от T", fontsize=20, fontweight='bold')
    ax.set_xlabel("T", fontsize=20, fontweight='bold')
    ax.set_ylabel("mean_value (%)", fontsize=20, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=14)  # подписи осей
    ax.grid(True)

    plt.show()

    # ---------------- ГРАФИК №2 ----------------
    print("\nЗамер времени для M процессов от 1 до 24:")

    N_test = 19
    Ad_test = -0.2
    B_test = symmetric_random_matrix(N_test)
    total_votes_test = 2 ** N_test

    M_values = []
    real_times = []
    ideal_times = []

    T1 = None  # время одного процесса, будем использовать для идеала

    for M in range(1, 25):
        chunk_size = total_votes_test // M
        ranges = [(i * chunk_size,
                   (i + 1) * chunk_size if i < M - 1 else total_votes_test,
                   N_test, Ad_test, B_test)
                  for i in range(M)]

        with multiprocessing.Pool(processes=M) as pool:
            results = pool.starmap(worker_range, ranges)

        max_time = max(r[1] for r in results)

        if M == 1:
            T1 = max_time  # эталонное время

        M_values.append(M)
        real_times.append(max_time)
        ideal_times.append(T1 / M)

        print(f"M={M}, Real={max_time:.6f}, Ideal={T1 / M:.6f}")

        # Строим график с жирным шрифтом
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(M_values, ideal_times, "bo-", label="Идеальное время (T1/M)")
    ax.plot(M_values, real_times, "ro-", label="Реальное время (max worker)")

    ax.set_title("Сравнение идеального и реального времени", fontsize=20, fontweight='bold')
    ax.set_xlabel("Число процессов (M)", fontsize=20, fontweight='bold')
    ax.set_ylabel("Время, сек", fontsize=20, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True)

    plt.show()
