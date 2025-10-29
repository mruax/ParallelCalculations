import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


# Генерация массива случайных чисел (float)
def generate_array(N, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1000.0, 1000.0, size=N)


# Разделение массива с помощью опорного элемента
def partition(arr, low, high):
    pivot = arr[high]  # Выбираем последний элемент как опорный
    i = low - 1  # Индекс для элементов меньше pivot

    # Проходим по всем элементам от low до high-1
    for j in range(low, high):
        if arr[j] <= pivot:  # Если элемент меньше или равен pivot
            i += 1  # Увеличиваем индекс меньшей части
            arr[i], arr[j] = arr[j], arr[i]  # Меняем элементы местами

    # Ставим pivot на его правильное место
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1  # Возвращаем позицию pivot


# Быстрая сортировка
def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    # Базовый случай: если в части массива больше 1 элемента
    if low < high:
        # Получаем индекс опорного элемента после разделения
        pivot_index = partition(arr, low, high)

        # Рекурсивно сортируем левую часть (элементы < pivot)
        quicksort(arr, low, pivot_index - 1)

        # Рекурсивно сортируем правую часть (элементы > pivot)
        quicksort(arr, pivot_index + 1, high)


# Worker для параллельной обработки
def worker(args):
    sub_array, chunk_id = args

    # Засекаем время начала сортировки этого chunk
    start_time = time.perf_counter()

    # Создаём копию массива для сортировки
    arr_copy = list(sub_array)

    # Выполняем быструю сортировку
    quicksort(arr_copy, 0, len(arr_copy) - 1)

    # Вычисляем время выполнения
    elapsed = time.perf_counter() - start_time

    return arr_copy, elapsed, chunk_id


# Итеративное попарное слияние
def iterative_merge(sorted_chunks):
    chunks = list(sorted_chunks)

    # Продолжаем слияние, пока есть больше одного chunk
    while len(chunks) > 1:
        merged_chunks = []

        # Попарно сливаем chunks
        for i in range(0, len(chunks), 2):
            if i + 1 < len(chunks):
                # Если есть пара - сливаем два chunk
                merged = merge_two_arrays(chunks[i], chunks[i + 1])
                merged_chunks.append(merged)
            else:
                # Если chunk остался без пары - просто добавляем его
                merged_chunks.append(chunks[i])

        # Обновляем список chunks для следующей итерации
        chunks = merged_chunks

    # Возвращаем единственный оставшийся chunk (полностью отсортированный массив)
    return chunks[0] if chunks else []


# Слияние двух отсортированных массивов
def merge_two_arrays(arr1, arr2):
    result = []
    i = j = 0  # Указатели для arr1 и arr2

    # Пока не дошли до конца ни одного из массивов
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    # Добавляем оставшиеся элементы (если есть)
    result.extend(arr1[i:])
    result.extend(arr2[j:])

    return result


# Главная функция - Параллельная быстрая сортировка
def parallel_quicksort(arr, processes):
    N = len(arr)
    chunk_size = N // processes  # Размер каждой части

    # Создаём chunks с индексами для сохранения порядка
    # Последний chunk может быть больше, если N не делится нацело на processes
    chunks = [(arr[i * chunk_size:(i + 1) * chunk_size] if i < processes - 1
               else arr[i * chunk_size:], i)
              for i in range(processes)]

    # НАЧАЛО ОТСЧЕТА ВРЕМЕНИ СОРТИРОВКИ
    sort_start_time = time.perf_counter()

    # Создаём пул процессов и распределяем задачи
    with mp.Pool(processes=processes) as pool:
        results = pool.map(worker, chunks)

    # Сортируем результаты по chunk_id для сохранения правильного порядка
    results.sort(key=lambda x: x[2])

    # Извлекаем только отсортированные части (без времени и id)
    sorted_chunks = [res[0] for res in results]

    # Выполняем итеративное попарное слияние всех частей
    merged = iterative_merge(sorted_chunks)

    # КОНЕЦ ОТСЧЕТА ВРЕМЕНИ СОРТИРОВКИ
    sort_end_time = time.perf_counter()
    total_sort_time = sort_end_time - sort_start_time

    return merged, total_sort_time


if __name__ == "__main__":
    processes = 6  # Количество процессов
    N = 5000000  # Размер массива для тестирования

    print(f"Параллельная быстрая сортировка массива из {N:,} элементов")

    random_arr = generate_array(N, seed=42)  # Генерация рандомного массива чисел float

    time_ideal = []  # Идеальное время
    time_real = []  # Реальное время
    base_time = None  # Время при 1 процессе

    # Тестирование с разным количеством процессов
    for process in range(1, processes + 1):
        # Сортируем копию массива (чтобы каждый раз сортировать один и тот же массив)
        sorted_arr, elapsed = parallel_quicksort(random_arr.copy(), processes=process)

        # Сохраняем время работы на 1 процессе как базовое
        if process == 1:
            base_time = elapsed

        # Сохраняем результаты
        time_real.append(elapsed)
        # Идеальное время = базовое_время / количество_процессов
        time_ideal.append(base_time / process)

        print(f"{process} процесс(ов): время {elapsed:.6f} сек")

    # Проверка корректности
    if sorted_arr == sorted(random_arr.tolist()):
        print("\nМассив отсортирован правильно!")
    else:
        print("\nОшибка сортировки!")

    # Вывод времени
    print("\nРеальное время:", time_real)
    print("Идеальное время:", time_ideal)

    # Построение графика производительности
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, processes + 1), time_ideal, label="Идеальная производительность", linestyle="--", color="black")
    plt.plot(range(1, processes + 1), time_real, label="Реальная производительность", marker="o", color="red")
    plt.xlabel("Количество процессов", fontsize=16, fontweight='bold')
    plt.ylabel("Время работы (с)", fontsize=16, fontweight='bold')
    plt.title("Масштабируемость быстрой сортировки", fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()
