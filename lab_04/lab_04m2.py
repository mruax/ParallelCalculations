import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import random
from collections import Counter
from operator import itemgetter
import matplotlib
matplotlib.use("TkAgg")


# Генерация случайных голосов депутатов
def generate_votes(N):
    # Голоса могут быть только -2, -1, 0, 1, 2
    return np.random.choice([-2, -1, 0, 1, 2], size=N)


# Генерация вектора воли депутатов
def generate_A(N):
    # Воля депутатов может быть только -10 или 10
    return np.random.choice([-10, 10], size=N)


# Генерация симметричной матрицы взаимодействий
def symmetric_random_matrix(N, seed=None):
    rng = np.random.default_rng(seed)  # Верхний треугольник со случайными числами
    upper_triangle = rng.uniform(-10.0, 10.0, size=(N, N))  # Верхний треугольник со случайными числами
    upper_triangle = np.triu(upper_triangle, k=1)  # Берём только верхний треугольник (без диагонали)
    matrix = upper_triangle + upper_triangle.T  # Симметризация
    np.fill_diagonal(matrix, 0.0)  # Диагональ обнуляем
    return matrix


# Подсчет конфигурации E
def calculate_E(A, B, V):
    # E = вклад воли депутатов + вклад взаимодействий
    E = np.sum(A * V)
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            E += B[i, j] * V[i] * V[j]
    return E


# Мутация: изменяем голос случайного депутата на +1 или -1 в пределах [-2,2]
def mutate_vote(V):
    new_V = V.copy()
    idx = np.random.randint(len(V))
    possible = []
    if new_V[idx] > -2:
        possible.append(-1)
    if new_V[idx] < 2:
        possible.append(1)
    if possible:
        new_V[idx] += random.choice(possible)
    return new_V


# Воркер для решения первой части задания
def worker_1(N, A, B, count, T):
    min_E = np.inf
    best_V = None  # хранение лучшей конфигурации

    t0 = time.perf_counter()  # Старт таймера

    for _ in range(count):
        # Начальная конфигурация голосов
        V = generate_votes(N)
        E = calculate_E(A, B, V)
        T_local = T  # Начальный коэффициент хаоса
        while T_local > 0.01:  # Цикл поиска, пока T не станет меньше 0.01
            V_new = mutate_vote(V)
            E_new = calculate_E(A, B, V_new)
            delta = E_new - E
            # Принимаем лучшее решение или с вероятностью exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T_local):
                V = V_new
                E = E_new
            T_local *= 0.99
        # сохраняем конфигурацию, если она дала новый минимум
        if E < min_E:
            min_E = E
            best_V = V.copy()

    t1 = time.perf_counter()  # Конец таймера

    return min_E, t1 - t0, best_V  # Возвращаем минимальное E, время и конфигурацию


# Воркер для решения второй части задания с использованием Process и stop_event
def worker_2_proc(N, A, B, T, E_target, stop_event, return_dict, idx):
    """
        Задача №2: поиск конфигурации с E <= E_target.
        Цикл продолжается, пока не найдено E <= E_target.
        T_local не уменьшается ниже 0.01
        """
    min_E = np.inf
    elapsed = 0
    V = 0
    transitions = {
        2: [1],
        -2: [-1],
        1: [2, 0],
        -1: [-2, 0],
        0: [1, -1],
    }

    t0 = time.perf_counter()

    while min_E > E_target and not stop_event.is_set():
        V = generate_votes(N)
        V0 = V.copy()
        T_local = T
        iteration = 1

        while True:
            E = calculate_E(A, B, V)
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
                E_new = calculate_E(A, B, V_test)
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

            if stop_event.is_set():
                break
        # print(E)
        if stop_event.is_set():
            break
    t1 = time.perf_counter()
    elapsed = t1 - t0
    # print(min_E, elapsed, V)
    return [min_E, elapsed, V]


if __name__ == "__main__":
    N = 100  # Число депутатов

    A = generate_A(N)  # Коэффициенты воли для депутатов
    B = symmetric_random_matrix(N)  # Матрица коэффициентов взаимодействий для пар депутатов

    processes = 24  # Число процессов
    chunk_size = 10  #
    total_tasks = processes * chunk_size  # Общее кол-во тасков
    T = 200  # Коэффициент хаоса

    # Тестируем масштабируемость Задачи 1
    print("Задача 1\n")
    time_real = []
    time_ideal = []
    result_min_E = np.inf
    T0 = 0
    for process in range(1, processes + 1):
        base_count = total_tasks // process
        remainder = total_tasks % process
        counts = [base_count + 1 if i < remainder else base_count for i in range(process)]
        tasks = [(N, A, B, c, T) for c in counts]

        with mp.Pool(processes=process) as pool:
            results = pool.starmap(worker_1, tasks)

        total_elapsed = max([r[1] for r in results])
        min_E = min([r[0] for r in results])
        conf_res = min(results, key=lambda x: x[0])[2]  # выбираем конфигурацию с минимальным E
        time_real.append(total_elapsed)
        T1 = total_elapsed if process == 1 else total_elapsed
        if process == 1:
            T0 = T1
        time_ideal.append(T0 / process)
        result_min_E = min(result_min_E, min_E)

        # выводим топ-5 голосов без np.int64
        top_votes = Counter(int(v) for v in conf_res).most_common(5)
        top_votes = sorted(top_votes, key=itemgetter(1))
        print(f"Потоков: {process:2d}, время выполнения: {total_elapsed:.4f} сек, "
              f"E мин.: {min_E:.4f}, Рейтинг голосов: {top_votes}")

    print(f"\nМинимальное E: {result_min_E:.4f}")

    # Выводим массивы
    print('\nРеальное время', time_real)
    print('Идеальное время', time_ideal)

    # График масштабируемости Задачи 1
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, processes + 1), time_ideal, "--k", label="Идеальная")
    plt.plot(range(1, processes + 1), time_real, "ro-", label="Реальная")
    plt.xlabel("Количество процессов", fontsize=16, fontweight='bold')
    plt.ylabel("Время работы (сек)", fontsize=16, fontweight='bold')
    plt.title("Сравнение скорости работы с использованием многопоточности (подзадача 1)", fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Тестируем масштабируемость Задачи 2
    print("\nЗадача 2\n")
    time_real_2 = []
    time_ideal_2 = []
    result_min_E_2 = np.inf

    # E_target можно взять из задачи 1
    E_target = result_min_E
    T0 = 0
    for process in range(1, processes + 1):
        stop_event = mp.Manager().Event()
        tasks = [(N, A, B, T, E_target, stop_event, 0, 0) for c in range(process)]

        with mp.Pool(processes=process) as pool:
            results = pool.starmap(worker_2_proc, tasks)

            # p = mp.Process(target=worker_2_proc, args=(N, A, B, T, E_target, stop_event, return_dict, i))

        #for res in results:
        #    print(res)

        if results:
            min_E2 = min(r[0] for r in results)
            conf_res2 = min(results, key=lambda x: x[0])[2]
            top_votes2 = Counter(int(v) for v in conf_res2).most_common(5)
            top_votes2 = sorted(top_votes2, key=itemgetter(1))
        else:
            min_E2 = None
            conf_res2 = None
            top_votes2 = []

        total_elapsed2 = max([r[1] for r in results] + [0])
        time_real_2.append(total_elapsed2)
        T1 = total_elapsed2 if process == 1 else total_elapsed2
        if process == 1:
            T0 = T1
        time_ideal_2.append(T0 / process)
        if min_E2 is not None:
            result_min_E_2 = min(result_min_E_2, min_E2)

        print(f"Потоков: {process:1d}, время выполнения: {total_elapsed2:.4f} сек, "
              f"E2 мин.: {min_E2 if min_E2 is not None else 'не найдено'}, "
              f"Рейтинг голосов: {top_votes2}")

    print(f"\nМинимальное E2 - {result_min_E_2 if result_min_E_2 != np.inf else 'не найдено'}")

    #  Выводим массивы
    print('\nРеальное время', time_real_2)
    print('Идеальное время', time_ideal_2)

    # График масштабируемости Задачи 2
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, processes + 1), time_ideal_2, "--k", label="Идеальная")
    plt.plot(range(1, processes + 1), time_real_2, "ro-", label="Реальная")
    plt.xlabel("Количество процессов", fontsize=16, fontweight='bold')
    plt.ylabel("Время работы (сек)", fontsize=16, fontweight='bold')
    plt.title("Сравнение скорости работы с использованием многопоточности (подзадача 2)", fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.legend()
    plt.show()
