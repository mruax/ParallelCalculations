"""
А блин с чего-то стартует, с какой-то конфигурации, которую потом мы будем потихоньку модифицировать, чтобы упасть в конфигурацию с значением Е в качестве некого локального минимума.
текущее положение = конф
Я беру некую близкую конфигурацию к исходной, да?
есть две близкие точки - возможность туда перейти
отличается на 1 ед
то есть только на 1
2 -> 1
-2 -> -1
1 -> 0 / 2
-1 -> 0 / -2
0 -> 1 / - 1

e1 - e2 = delta
< 0 - переход в пробную точку
e1 != e2
> 0 - можно перейти с вер-тью p=exp(-delta/T)
видимо выбираем е мин среди +- дельта

найти 10 локал мин стартуя с разных конфигураций (не 20)
1. T начальное = 200
2. Каждые 1000 итераций -> T = T * 0.99
3. Останавливаем вычисления когда T <= 0.01

Для кажд конф:
1. Сколько чисел в конф больше всего

Кол-во конфигураций 24 * 10, то есть 10 на поток.

emin - 10 лучш конф с 1 потока = иголка
10 иголок
параллельный алгоритм:
с произвольной конф найти ту минимальную конф, значение Е которой = или меньше
задаем а и б, ищем 10 лучш конф на каждом потоке
потом берем 10 лучш среди всех потоков
while не нашли минмальный ищем?
то есть после части 1 мы уже нашли минимальное е

1. Точно также генерируем случайную конфигурацию
2. Ad -> rand +2 ili -2 ili 0
3. Vd -> rand -2 -1 0 1 2
4. Bnm -> [-2;2]

"""
import multiprocessing
import time

import matplotlib.pyplot as plt
import numpy as np


def generate_deputies_will(deputies) -> np.ndarray:
    return np.random.choice([-2, 2, 0], size=deputies)


def generate_deputies_votes(deputies) -> np.ndarray:
    return np.random.choice([-2, -1, 0, 1, 2], size=deputies)


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


def calculate_E(N, A, B, V):
    E = np.sum(A * V)
    for n in range(N):
        for m in range(n + 1, N):
            E += B[n, m] * V[n] * V[m]
    return E


def worker(N, A, B, count, T):
    """Задача №1"""
    min_E = np.inf
    transitions = {
        2: [1],
        -2: [-1],
        1: [2, 0],
        -1: [-2, 0],
        0: [1, -1],
    }
    t0 = time.perf_counter()
    for i in range(count):
        V = generate_deputies_votes(N)
        V0 = V.copy()
        T_local = T
        iteration = 1
        E = 0
        while T_local > 0.01:
            E = calculate_E(N, A, B, V)
            random_deputy = np.random.randint(0, N)
            current = V[random_deputy]
            possible_moves = transitions[current]

            if len(possible_moves) == 2:
                best_V = V.copy()
                best_E = E
                for new_val in possible_moves:
                    V_test = best_V.copy()
                    V_test[random_deputy] = new_val
                    E_new = calculate_E(N, A, B, V_test)
                    if E_new < best_E:
                        best_E = E_new
                        best_V = V_test
            else:
                V_test = V.copy()
                V_test[random_deputy] = possible_moves[0]
                E_new = calculate_E(N, A, B, V_test)
                best_E = E_new
                best_V = V_test

            delta_E = best_E - E
            if delta_E < 0:
                V = best_V
            else:
                p = np.exp(-delta_E / T_local)
                R = np.random.random()  # (0, 1)
                if p >= R:
                    V = best_V
                else:
                    V = V0.copy()
            iteration += 1
            if iteration == 10:  # 1000
                iteration = 1
                T_local = T_local * 0.2  # 0.99
        # print(f"Высчитано {i + 1}-ое значение E = {E}")
        min_E = min(min_E, E)

    t1 = time.perf_counter()
    elapsed = t1 - t0
    return [min_E, elapsed]


def worker_task2(N, A, B, T, E_target, stop_event):
    """
    Задача №2: поиск конфигурации с E <= E_target.
    Цикл продолжается, пока не найдено E <= E_target.
    T_local не уменьшается ниже 0.01
    """
    min_E = np.inf
    elapsed = 0
    transitions = {
        2: [1],
        -2: [-1],
        1: [2, 0],
        -1: [-2, 0],
        0: [1, -1],
    }

    t0 = time.perf_counter()

    while min_E > E_target and not stop_event.is_set():
        V = generate_deputies_votes(N)
        V0 = V.copy()
        T_local = T
        iteration = 1

        while True:
            E = calculate_E(N, A, B, V)
            if E <= E_target:
                min_E = min(min_E, E)
                stop_event.set()
                break

            random_deputy = np.random.randint(0, N)
            current = V[random_deputy]
            possible_moves = transitions[current]

            best_V = V.copy()
            best_E = E

            for new_val in possible_moves:
                V_test = V.copy()
                V_test[random_deputy] = new_val
                E_new = calculate_E(N, A, B, V_test)
                if E_new < best_E:
                    best_E = E_new
                    best_V = V_test

            delta_E = best_E - E
            if delta_E < 0:
                V = best_V
            else:
                p = np.exp(-delta_E / T_local)
                R = np.random.random()  # (0, 1)
                if p >= R:
                    V = best_V
                else:
                    V = V0.copy()
            iteration += 1
            if iteration == 10:  # 1000
                iteration = 1
                T_local = T_local * 0.2  # 0.99
                T_local = max(T_local, 0.01)

    t1 = time.perf_counter()
    elapsed = t1 - t0
    return min_E, elapsed


if __name__ == "__main__":
    N = 100

    # Фиксированные коэффициенты
    A = generate_deputies_will(N)   # Воля депутата
    B = symmetric_random_matrix(N)  # Взаимоотношения депутатов

    # ---------------- PARALLEL TASK №1 ----------------
    workers = multiprocessing.cpu_count()  # 24
    chunk_size = 10
    total_tasks = workers * chunk_size
    T = 200

    real_times = []
    ideal_times = []
    result_min_E = np.inf

    T1 = 0

    print("Параллельная задача №1...")
    for workers in range(1, workers + 1):
        base_count = total_tasks // workers
        remainder = total_tasks % workers
        counts = [base_count + 1 if i < remainder else base_count for i in range(workers)]
        all_tasks = [(N, A, B, count, T) for count in counts]
        total_elapsed, min_E = 0, np.inf

        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.starmap(worker, all_tasks)

        for E, elapsed in results:
            min_E = min(min_E, E)
            total_elapsed = max(total_elapsed, elapsed)  # min != 0!!!!!!!!

        real_times.append(total_elapsed)
        if workers == 1:
            T1 = total_elapsed
        ideal_times.append(T1 / workers)
        result_min_E = min(result_min_E, min_E)
        print(f"Потоков: {workers:2d} | chunk_size: {base_count:3d} | "
              f"Время: {total_elapsed:.4f} сек | E_min: {min_E:.4f}")
    print(f"Минимальное E - {result_min_E:.4f}")

    # ---------------- ГРАФИК №1 ----------------
    M_values = list(range(1, workers + 1))
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

    # ---------------- PARALLEL TASK №2 ----------------
    real_times2 = []
    ideal_times2 = []
    total_elapsed2 = 0
    result_min_E2 = np.inf
    print("Параллельная задача №2...")
    for workers in range(1, workers + 1):
        stop_event = multiprocessing.Manager().Event()
        all_tasks2 = [(N, A, B, T, result_min_E, stop_event) for _ in range(workers)]

        with multiprocessing.Pool(processes=workers) as pool:
            results2 = pool.starmap(worker_task2, all_tasks2)

        total_elapsed2 = 0
        min_E2 = np.inf
        for E, elapsed in results2:
            min_E2 = min(min_E2, E)
            total_elapsed2 = max(total_elapsed2, elapsed)

        real_times2.append(total_elapsed2)
        if workers == 1:
            T1 = total_elapsed2
        ideal_times2.append(T1 / workers)
        result_min_E2 = min(result_min_E2, min_E2)
        print(f"Потоков: {workers:2d} | Время: {total_elapsed2:.4f} сек | E_min2: {min_E2:.4f}")
    print(f"Минимальное E2 - {result_min_E2:.4f}")

    # ---------------- ГРАФИК №2 ----------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(M_values, ideal_times2, "bo-", label="Идеальное время (T1/M)")
    ax.plot(M_values, real_times2, "ro-", label="Реальное время (max worker)")

    ax.set_title("Сравнение идеального и реального времени", fontsize=20, fontweight='bold')
    ax.set_xlabel("Число процессов (M)", fontsize=20, fontweight='bold')
    ax.set_ylabel("Время, сек", fontsize=20, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True)

    plt.show()
