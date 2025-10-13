import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt


# ============= РЕАЛИЗАЦИЯ HEAPSORT С НУЛЯ =============

def heapify(arr, n, i):
    """
    Процедура просеивания вниз для поддержания свойства max-heap
    arr - массив
    n - размер кучи
    i - индекс корня поддерева
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    # Если левый потомок больше корня
    if left < n and arr[left] > arr[largest]:
        largest = left

    # Если правый потомок больше текущего наибольшего
    if right < n and arr[right] > arr[largest]:
        largest = right

    # Если наибольший элемент не корень
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        # Рекурсивно просеиваем затронутое поддерево
        heapify(arr, n, largest)


def heapsort(arr):
    """
    Сортировка кучей (heapsort) - полная реализация с нуля
    """
    n = len(arr)
    arr = arr.copy()  # Не модифицируем исходный массив

    # Шаг 1: Построение max-heap (перегруппировка массива)
    # Начинаем с последнего не-листового узла и идем вверх
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Шаг 2: Извлечение элементов из кучи по одному
    for i in range(n - 1, 0, -1):
        # Перемещаем текущий корень (максимум) в конец
        arr[0], arr[i] = arr[i], arr[0]
        # Вызываем heapify на уменьшенной куче
        heapify(arr, i, 0)

    return arr


def merge_sorted_arrays(arrays):
    """
    Объединение k отсортированных массивов с использованием min-heap
    Реализация без heapq
    """

    # Простая min-heap реализация
    class MinHeap:
        def __init__(self):
            self.heap = []

        def push(self, item):
            """Добавить элемент в кучу"""
            self.heap.append(item)
            self._sift_up(len(self.heap) - 1)

        def pop(self):
            """Извлечь минимальный элемент"""
            if len(self.heap) == 0:
                raise IndexError("pop from empty heap")

            # Меняем первый и последний элементы
            self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
            result = self.heap.pop()

            if len(self.heap) > 0:
                self._sift_down(0)

            return result

        def _sift_up(self, idx):
            """Просеивание вверх"""
            parent = (idx - 1) // 2
            if idx > 0 and self.heap[idx] < self.heap[parent]:
                self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
                self._sift_up(parent)

        def _sift_down(self, idx):
            """Просеивание вниз"""
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2

            if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
                smallest = left

            if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest != idx:
                self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
                self._sift_down(smallest)

        def __len__(self):
            return len(self.heap)

    # K-way merge
    heap = MinHeap()

    # Инициализация: (значение, индекс_массива, индекс_элемента)
    for i, arr in enumerate(arrays):
        if len(arr) > 0:
            heap.push((arr[0], i, 0))

    result = []
    while len(heap) > 0:
        val, arr_idx, elem_idx = heap.pop()
        result.append(val)

        # Если в массиве есть еще элементы
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heap.push((next_val, arr_idx, elem_idx + 1))

    return result


def worker_sort(start, end, shared_array):
    """Сортирует часть массива с индексами [start, end)"""
    t0 = time.perf_counter()
    portion = shared_array[start:end]
    sorted_portion = heapsort(portion)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    return sorted_portion, elapsed


def parallel_heapsort(data, M):
    """
    Параллельная сортировка массива data на M процессах
    Возвращает: отсортированный массив, общее время, время сортировки, время merge
    """
    N = len(data)
    chunk_size = N // M
    ranges = [(i * chunk_size,
               (i + 1) * chunk_size if i < M - 1 else N,
               data)
              for i in range(M)]

    # Параллельная сортировка частей
    with multiprocessing.Pool(processes=M) as pool:
        results = pool.starmap(worker_sort, ranges)

    sorted_chunks = [r[0] for r in results]
    worker_times = [r[1] for r in results]
    max_worker_time = max(worker_times)

    # Последовательное слияние отсортированных частей
    t0 = time.perf_counter()
    final_sorted = merge_sorted_arrays(sorted_chunks)
    t1 = time.perf_counter()
    merge_time = t1 - t0

    total_time = max_worker_time + merge_time

    return final_sorted, total_time, max_worker_time, merge_time


if __name__ == "__main__":
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА: ПАРАЛЛЕЛЬНЫЙ HEAPSORT")
    print("Реализация полностью с нуля (без heapq)")
    print("=" * 80)

    # Проверка корректности heapsort
    print("\n[ТЕСТ] Проверка корректности сортировки...")
    test_arr = [12, 11, 13, 5, 6, 7, 3, 15, 2, 8]
    sorted_arr = heapsort(test_arr)
    expected = sorted(test_arr)
    print(f"Исходный массив:      {test_arr}")
    print(f"После heapsort:       {sorted_arr}")
    print(f"Ожидаемый результат:  {expected}")
    print(f"Корректность: {'✓ PASSED' if sorted_arr == expected else '✗ FAILED'}")

    # ================ ЭКСПЕРИМЕНТ 1: Время для M процессов от 1 до 24 ================
    print("\n[ЭКСПЕРИМЕНТ 1] Замер времени для M процессов от 1 до 24")
    print("-" * 80)

    # Подбираем N для ~20 секунд на 1 процессе
    N_exp1 = 6_000_000  # Собственная реализация медленнее, уменьшаем N
    print(f"Размер данных N = {N_exp1:,}")

    # Генерируем случайный массив
    np.random.seed(42)
    test_data = np.random.rand(N_exp1).tolist()

    M_values = []
    real_times = []
    worker_times = []
    merge_times = []
    ideal_times = []

    T1 = None  # Время для 1 процесса

    for M in range(1, 25):
        print(f"M = {M:2d} процессов... ", end="", flush=True)

        _, total_time, max_worker, merge_time = parallel_heapsort(test_data, M)

        if M == 1:
            T1 = total_time

        M_values.append(M)
        real_times.append(total_time)
        worker_times.append(max_worker)
        merge_times.append(merge_time)
        ideal_times.append(T1 / M)

        print(f"Total={total_time:.3f}s, Worker={max_worker:.3f}s, "
              f"Merge={merge_time:.3f}s, Ideal={T1 / M:.3f}s")

    # График 1: Идеальное vs реальное время
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(M_values, ideal_times, "bo-", linewidth=2.5, markersize=8,
            label="Идеальное время (T₁/M)")
    ax.plot(M_values, real_times, "ro-", linewidth=2.5, markersize=8,
            label="Реальное время (с merge)")
    ax.plot(M_values, worker_times, "go--", linewidth=2, markersize=6,
            label="Время сортировки (max worker)")

    ax.set_title("Сравнение идеального и реального времени параллельного heapsort",
                 fontsize=20, fontweight='bold')
    ax.set_xlabel("Число процессов (M)", fontsize=20, fontweight='bold')
    ax.set_ylabel("Время, сек", fontsize=20, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ================ ЭКСПЕРИМЕНТ 2: Зависимость времени от N для 24 потоков ================
    print("\n[ЭКСПЕРИМЕНТ 2] Зависимость времени от размера данных N (M = 24)")
    print("-" * 80)

    M_fixed = 24
    N_values = []
    times_parallel = []
    times_sequential = []
    speedup_values = []
    efficiency_values = []

    # Тестируем разные размеры
    test_sizes = [500_000, 1_000_000, 2_000_000, 3_000_000]

    for N in test_sizes:
        print(f"N = {N:>10,} ... ", end="", flush=True)

        # Генерируем данные
        np.random.seed(42)
        data = np.random.rand(N).tolist()

        # Последовательная сортировка (M=1)
        _, t_seq, _, _ = parallel_heapsort(data, 1)

        # Параллельная сортировка (M=24)
        _, t_par, _, _ = parallel_heapsort(data, M_fixed)

        speedup = t_seq / t_par
        efficiency = speedup / M_fixed * 100  # в процентах

        N_values.append(N)
        times_sequential.append(t_seq)
        times_parallel.append(t_par)
        speedup_values.append(speedup)
        efficiency_values.append(efficiency)

        print(f"T_seq={t_seq:.3f}s, T_par={t_par:.3f}s, "
              f"Speedup={speedup:.2f}x, Efficiency={efficiency:.1f}%")

    # График 2: Зависимость времени от N
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Подграфик 1: Время выполнения
    N_millions = [n / 1_000_000 for n in N_values]
    ax1.plot(N_millions, times_sequential, "bs-", linewidth=2.5, markersize=8,
             label="Последовательный (M=1)")
    ax1.plot(N_millions, times_parallel, "rs-", linewidth=2.5, markersize=8,
             label="Параллельный (M=24)")

    ax1.set_title("Зависимость времени выполнения от размера данных",
                  fontsize=18, fontweight='bold')
    ax1.set_xlabel("Размер данных N (млн элементов)", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Время, сек", fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Подграфик 2: Ускорение и эффективность
    ax2_twin = ax2.twinx()

    line1 = ax2.plot(N_millions, speedup_values, "go-", linewidth=2.5, markersize=8,
                     label="Ускорение (Speedup)")
    ax2.axhline(y=M_fixed, color='gray', linestyle='--', linewidth=2,
                label=f"Идеальное ускорение ({M_fixed}x)")

    line2 = ax2_twin.plot(N_millions, efficiency_values, "mo-", linewidth=2.5,
                          markersize=8, label="Эффективность (%)")

    ax2.set_title("Ускорение и эффективность параллелизации",
                  fontsize=18, fontweight='bold')
    ax2.set_xlabel("Размер данных N (млн элементов)", fontsize=16, fontweight='bold')
    ax2.set_ylabel("Ускорение (раз)", fontsize=16, fontweight='bold', color='green')
    ax2_twin.set_ylabel("Эффективность (%)", fontsize=16, fontweight='bold', color='purple')

    ax2.tick_params(axis='y', labelcolor='green', labelsize=13)
    ax2.tick_params(axis='x', labelsize=13)
    ax2_twin.tick_params(axis='y', labelcolor='purple', labelsize=13)

    # Объединяем легенды
    lines = line1 + line2 + [plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2)]
    labels = [l.get_label() for l in line1] + [l.get_label() for l in line2] + \
             [f"Идеальное ускорение ({M_fixed}x)"]
    ax2.legend(lines, labels, fontsize=12, loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print("=" * 80)