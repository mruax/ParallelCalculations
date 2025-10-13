import heapq
import multiprocessing
import time

import matplotlib.pyplot as plt
import numpy as np


def mergesort(arr):
    """
    Классическая рекурсивная реализация сортировки слиянием.
    Возвращает отсортированный массив.
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])

    return merge(left, right)


def merge(left, right):
    """
    Слияние двух отсортированных массивов в один отсортированный.
    """
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


def worker_sort(start, end, data):
    """
    Воркер-функция для сортировки части массива.
    Принимает индексы start, end и общий массив data.
    Возвращает отсортированную часть массива и время работы.
    """
    t0 = time.perf_counter()

    # Извлекаем часть массива
    chunk = data[start:end]

    # Сортируем часть
    sorted_chunk = mergesort(chunk)

    t1 = time.perf_counter()
    elapsed = t1 - t0

    return sorted_chunk, elapsed


def k_way_merge(sorted_chunks):
    """
    Объединяет K отсортированных массивов в один отсортированный массив.
    Использует min-heap для эффективного слияния.
    """
    # Создаём кучу с первыми элементами каждого массива
    heap = []
    for chunk_idx, chunk in enumerate(sorted_chunks):
        if len(chunk) > 0:
            heapq.heappush(heap, (chunk[0], chunk_idx, 0))  # (значение, индекс массива, индекс в массиве)

    result = []

    while heap:
        value, chunk_idx, elem_idx = heapq.heappop(heap)
        result.append(value)

        # Добавляем следующий элемент из того же массива, если он есть
        if elem_idx + 1 < len(sorted_chunks[chunk_idx]):
            next_value = sorted_chunks[chunk_idx][elem_idx + 1]
            heapq.heappush(heap, (next_value, chunk_idx, elem_idx + 1))

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА: Параллельный Mergesort")
    print("=" * 60)

    # Параметры эксперимента
    N_test = 5_000_000  # Размер массива для основного эксперимента

    print(f"\nГенерация случайного массива размером N = {N_test:,}...")
    np.random.seed(42)  # Для воспроизводимости
    data_test = np.random.randint(-1_000_000, 1_000_000, size=N_test).tolist()

    print("\nЗамер времени для M процессов от 1 до 24:")
    print("-" * 60)

    M_values = []
    real_times = []
    ideal_times = []

    T1 = None  # Время одного процесса

    for M in range(1, 25):
        # Разделяем массив на M частей
        chunk_size = N_test // M
        ranges = []
        for i in range(M):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < M - 1 else N_test
            ranges.append((start, end, data_test))

        # Запускаем параллельную сортировку
        t_start = time.perf_counter()

        with multiprocessing.Pool(processes=M) as pool:
            results = pool.starmap(worker_sort, ranges)

        # Получаем отсортированные части и время работы каждого воркера
        sorted_chunks = [r[0] for r in results]
        worker_times = [r[1] for r in results]
        max_worker_time = max(worker_times)

        # Объединяем отсортированные части (k-way merge)
        t_merge_start = time.perf_counter()
        final_sorted = k_way_merge(sorted_chunks)
        t_merge_end = time.perf_counter()
        merge_time = t_merge_end - t_merge_start

        t_end = time.perf_counter()
        total_time = t_end - t_start

        # Время для одного процесса
        if M == 1:
            T1 = total_time

        M_values.append(M)
        real_times.append(total_time)
        ideal_times.append(T1 / M if T1 else 0)

        print(f"M={M:2d} | Total={total_time:7.4f}s | Workers={max_worker_time:7.4f}s | "
              f"Merge={merge_time:6.4f}s | Ideal={T1 / M if T1 else 0:7.4f}s")

    print("-" * 60)
    print(f"\nБазовое время T1 (1 процесс): {T1:.4f} секунд")
    print(f"Ускорение при 24 процессах: {T1 / real_times[-1]:.2f}x")

    # ---------------- ГРАФИК ----------------
    print("\nПостроение графика...")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(M_values, ideal_times, "bo-", label="Идеальное время (T1/M)",
            linewidth=2, markersize=8)
    ax.plot(M_values, real_times, "ro-", label="Реальное время (с merge)",
            linewidth=2, markersize=8)

    ax.set_title("Сравнение идеального и реального времени\nпри параллельной сортировке Mergesort",
                 fontsize=20, fontweight='bold')
    ax.set_xlabel("Число процессов (M)", fontsize=18, fontweight='bold')
    ax.set_ylabel("Время, сек", fontsize=18, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Добавляем аннотацию с ускорением
    ax.text(0.02, 0.98, f'Размер данных: {N_test:,}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

    print("\nГотово! График отображён.")
