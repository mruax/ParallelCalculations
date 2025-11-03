"""
Лабораторная работа: Кластеризация K-means с параллелизацией

Алгоритм состоит из двух модулей:
- Модуль 2: Поиск начальных центров кластеров и их количества
- Модуль 1: Итеративная кластеризация K-means
"""

import multiprocessing
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from typing import List, Tuple, Optional


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Вычисляет евклидово расстояние между двумя точками в 2D пространстве."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def calculate_distances(point: Tuple[float, float], centers: List[Tuple[float, float]]) -> List[float]:
    """Вычисляет расстояния от точки до всех центров кластеров."""
    x, y = point
    distances = []
    for cx, cy in centers:
        dist = euclidean_distance(x, y, cx, cy)
        distances.append(dist)
    return distances


# ==================== МОДУЛЬ 2: ПОИСК НАЧАЛЬНЫХ ЦЕНТРОВ ====================

def calculate_E_for_point(candidate: Tuple[float, float], data_points: np.ndarray) -> float:
    """
    Вычисляет целевую функцию E для кандидата в центры кластера.
    E = sum(1 / (R_i + 0.01)), где R_i - расстояние до i-й точки данных.
    Чем больше E, тем лучше точка подходит как центр кластера.
    """
    x, y = candidate
    E = 0.0
    for point in data_points:
        px, py = point[0], point[1]
        R = euclidean_distance(x, y, px, py)
        E += 1.0 / (R + 0.01)
    return E


def worker_find_centers_chunk(start_idx: int, end_idx: int,
                                grid_points: np.ndarray,
                                data_points: np.ndarray) -> Tuple[float, Tuple[float, float], int]:
    """
    Обрабатывает диапазон точек сетки [start_idx, end_idx) для поиска лучшего центра.
    Возвращает (max_E, best_point, best_idx) + время работы.
    """
    t0 = time.perf_counter()

    max_E = -1.0
    best_point = None
    best_idx = -1

    for i in range(start_idx, end_idx):
        candidate = (grid_points[i, 0], grid_points[i, 1])
        E = calculate_E_for_point(candidate, data_points)

        if E > max_E:
            max_E = E
            best_point = candidate
            best_idx = i

    t1 = time.perf_counter()
    elapsed = t1 - t0

    return max_E, best_point, best_idx, elapsed


def find_initial_centers_parallel(data: np.ndarray, R_excl: float, n_processes: int = 1) -> Tuple[List[Tuple[float, float]], float]:
    """
    МОДУЛЬ 2: Находит начальные центры кластеров параллельно.

    Алгоритм:
    1. Создаем сетку из всех точек данных как потенциальных центров
    2. Для каждой точки вычисляем E (больше - лучше)
    3. Выбираем точку с максимальным E
    4. Исключаем точки в радиусе R_excl
    5. Повторяем пока остаются точки

    Возвращает: (список центров, суммарное время вычислений в воркерах)
    """
    centers = []
    remaining_points = data.copy()
    total_computation_time = 0.0  # Суммарное время вычислений

    print(f"\n{'='*60}")
    print(f"МОДУЛЬ 2: Поиск начальных центров (процессов: {n_processes})")
    print(f"{'='*60}")
    print(f"Всего точек данных: {len(data)}")
    print(f"Радиус исключения: {R_excl:.3f}")

    iteration = 0
    while len(remaining_points) > 0:
        iteration += 1
        print(f"\nИтерация {iteration}: осталось точек = {len(remaining_points)}")

        # Используем оставшиеся точки как сетку кандидатов
        grid_points = remaining_points
        n_grid = len(grid_points)

        if n_grid == 0:
            break

        # Параллельная обработка
        chunk_size = max(1, n_grid // n_processes)
        ranges = []
        for i in range(n_processes):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_processes - 1 else n_grid
            if start < n_grid:
                ranges.append((start, end, grid_points, remaining_points))

        # Запуск параллельных воркеров
        if n_processes > 1:
            with multiprocessing.Pool(processes=n_processes) as pool:
                results = pool.starmap(worker_find_centers_chunk, ranges)
        else:
            results = [worker_find_centers_chunk(*r) for r in ranges]

        # Находим глобальный максимум E и максимальное время среди воркеров
        max_E = -1.0
        best_center = None
        max_time = 0.0

        for E, point, idx, elapsed in results:
            max_time = max(max_time, elapsed)  # Максимальное время = время самого медленного воркера
            if E > max_E:
                max_E = E
                best_center = point

        total_computation_time += max_time  # Добавляем время итерации

        if best_center is None:
            break

        centers.append(best_center)
        print(f"  Найден центр #{len(centers)}: ({best_center[0]:.3f}, {best_center[1]:.3f}), E={max_E:.3f}")
        print(f"  Время вычислений: {max_time:.6f} сек")

        # Исключаем точки в радиусе R_excl
        new_remaining = []
        for point in remaining_points:
            dist = euclidean_distance(point[0], point[1], best_center[0], best_center[1])
            if dist >= R_excl:
                new_remaining.append(point)

        remaining_points = np.array(new_remaining) if len(new_remaining) > 0 else np.array([])
        excluded = n_grid - len(remaining_points)
        print(f"  Исключено точек: {excluded}")

    print(f"\nВсего найдено центров: {len(centers)}")
    print(f"Суммарное время вычислений (Модуль 2): {total_computation_time:.6f} сек")
    return centers, total_computation_time


# ==================== МОДУЛЬ 1: K-MEANS КЛАСТЕРИЗАЦИЯ ====================

def worker_assign_clusters_chunk(start_idx: int, end_idx: int,
                                   data: np.ndarray,
                                   centers: List[Tuple[float, float]]) -> Tuple[List[int], float]:
    """
    Обрабатывает диапазон точек [start_idx, end_idx) для присвоения кластеров.
    Возвращает список индексов кластеров + время работы.
    """
    t0 = time.perf_counter()

    cluster_assignments = []
    for i in range(start_idx, end_idx):
        point = data[i]
        distances = calculate_distances((point[0], point[1]), centers)
        closest_cluster = np.argmin(distances)
        cluster_assignments.append(closest_cluster)

    t1 = time.perf_counter()
    elapsed = t1 - t0

    return cluster_assignments, elapsed


def assign_clusters_parallel(data: np.ndarray, centers: List[Tuple[float, float]], n_processes: int = 1) -> Tuple[np.ndarray, float]:
    """
    Присваивает каждой точке кластер на основе минимального расстояния до центров.
    Использует параллелизацию.

    Возвращает: (массив меток кластеров, время вычислений)
    """
    n_points = len(data)
    chunk_size = max(1, n_points // n_processes)

    ranges = []
    for i in range(n_processes):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_processes - 1 else n_points
        if start < n_points:
            ranges.append((start, end, data, centers))

    # Параллельная обработка
    if n_processes > 1:
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = pool.starmap(worker_assign_clusters_chunk, ranges)
    else:
        results = [worker_assign_clusters_chunk(*r) for r in ranges]

    # Объединяем результаты и находим максимальное время
    all_assignments = []
    max_time = 0.0
    for assignments, elapsed in results:
        all_assignments.extend(assignments)
        max_time = max(max_time, elapsed)

    return np.array(all_assignments), max_time


def update_centers(data: np.ndarray, labels: np.ndarray, n_clusters: int) -> List[Tuple[float, float]]:
    """
    Обновляет центры кластеров как средние координаты точек в каждом кластере.
    """
    new_centers = []
    for k in range(n_clusters):
        cluster_points = data[labels == k]
        if len(cluster_points) > 0:
            mean_x = np.mean(cluster_points[:, 0])
            mean_y = np.mean(cluster_points[:, 1])
            new_centers.append((mean_x, mean_y))
        else:
            # Если кластер пустой, оставляем старый центр
            new_centers.append((0.0, 0.0))  # или можно взять случайную точку
    return new_centers


def kmeans_clustering_parallel(data: np.ndarray, initial_centers: List[Tuple[float, float]],
                                 n_processes: int = 1, max_iterations: int = 100,
                                 tolerance: float = 1e-4) -> Tuple[np.ndarray, List[Tuple[float, float]], float]:
    """
    МОДУЛЬ 1: K-means кластеризация с параллелизацией.

    Алгоритм:
    1. Присваиваем каждой точке кластер (по минимальному расстоянию до центра)
    2. Пересчитываем центры как средние координаты точек кластера
    3. Повторяем пока центры не стабилизируются

    Возвращает: (метки кластеров, финальные центры, суммарное время вычислений)
    """
    print(f"\n{'='*60}")
    print(f"МОДУЛЬ 1: K-means кластеризация (процессов: {n_processes})")
    print(f"{'='*60}")
    print(f"Количество кластеров: {len(initial_centers)}")
    print(f"Количество точек: {len(data)}")

    centers = list(initial_centers)  # Конвертируем в список если это кортеж
    iteration = 0
    total_computation_time = 0.0

    for iteration in range(max_iterations):
        # Шаг 1: Присваиваем кластеры
        labels, iter_time = assign_clusters_parallel(data, centers, n_processes)
        total_computation_time += iter_time

        # Шаг 2: Обновляем центры
        new_centers = update_centers(data, labels, len(centers))

        # Проверяем сходимость
        center_shift = 0.0
        for i in range(len(centers)):
            shift = euclidean_distance(centers[i][0], centers[i][1],
                                        new_centers[i][0], new_centers[i][1])
            center_shift += shift

        print(f"Итерация {iteration + 1}: сдвиг центров = {center_shift:.6f}, время вычислений = {iter_time:.6f} сек")

        if center_shift < tolerance:
            print(f"Сходимость достигнута на итерации {iteration + 1}")
            break

        centers = new_centers

    # Финальное присвоение кластеров
    final_labels, final_time = assign_clusters_parallel(data, centers, n_processes)
    total_computation_time += final_time

    print(f"Суммарное время вычислений (Модуль 1): {total_computation_time:.6f} сек")

    return final_labels, centers, total_computation_time


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def plot_clusters(data: np.ndarray, labels: np.ndarray, centers: List[Tuple[float, float]],
                   title: str = "Результат кластеризации"):
    """Визуализирует результаты кластеризации."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Рисуем точки с цветами кластеров
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis',
                         s=30, alpha=0.6, edgecolors='k', linewidth=0.5)

    # Рисуем центры кластеров
    centers_array = np.array(centers)
    ax.scatter(centers_array[:, 0], centers_array[:, 1],
               c='red', marker='X', s=300, edgecolors='black', linewidth=2,
               label='Центры кластеров', zorder=5)

    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlabel("X", fontsize=16, fontweight='bold')
    ax.set_ylabel("Y", fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Номер кластера')
    plt.tight_layout()
    plt.show()


def plot_performance(M_values: List[int], real_times: List[float], ideal_times: List[float]):
    """Строит график зависимости времени выполнения от количества процессов."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(M_values, ideal_times, "bo-", linewidth=2, markersize=8, label="Идеальное время (T1/M)")
    ax.plot(M_values, real_times, "ro-", linewidth=2, markersize=8, label="Реальное время")

    ax.set_title("Зависимость времени выполнения от числа процессов",
                 fontsize=20, fontweight='bold')
    ax.set_xlabel("Число процессов (M)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Время, сек", fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== ГЕНЕРАЦИЯ ДАННЫХ ====================

def generate_synthetic_data(n_samples: int = 1000, n_clusters: int = 5,
                             random_state: int = 42) -> np.ndarray:
    """Генерирует синтетические данные с явной кластерной структурой."""
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=n_clusters,
                      cluster_std=1.5, random_state=random_state)
    return X


def load_kaggle_dataset(filepath: str) -> Optional[np.ndarray]:
    """
    Загружает датасет с Kaggle.
    Формат: CSV с числовыми колонками для X и Y координат.
    """
    try:
        df = pd.read_csv(filepath)
        # Предполагаем, что первые две числовые колонки - это X и Y
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            data = df[numeric_cols[:2]].values
            print(f"Загружен датасет: {len(data)} точек")
            return data
        else:
            print("Ошибка: недостаточно числовых колонок в датасете")
            return None
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")
        return None


# ==================== ЭКСПЕРИМЕНТЫ С ПАРАЛЛЕЛИЗАЦИЕЙ ====================

def run_parallelization_experiment(data: np.ndarray, R_excl: float,
                                     max_processes: int = 24,
                                     first_run_time: Optional[float] = None):
    """
    Проводит эксперимент с различным количеством процессов (1-24).
    Замеряет СУММАРНОЕ время вычислений Модуля 2 + Модуля 1.

    Параметры:
    - first_run_time: время первого запуска с 1 процессом (чтобы не дублировать)
    """
    print(f"\n{'='*60}")
    print(f"ЭКСПЕРИМЕНТ: Тестирование параллелизации (1-{max_processes} процессов)")
    print(f"{'='*60}")
    print(f"Замеряем суммарное время ЧИСТЫХ ВЫЧИСЛЕНИЙ: Модуль 2 + Модуль 1")

    M_values = []
    real_times = []
    ideal_times = []
    T1 = first_run_time  # Используем время из первого запуска если есть

    # Используем меньший датасет для эксперимента, чтобы ускорить
    test_data = data[:min(500, len(data))]

    for M in range(1, max_processes + 1):
        print(f"\n{'='*50}")
        print(f"Тестирование с M={M} процессами...")
        print(f"{'='*50}")

        # Если M=1 и уже есть результаты первого запуска, используем их
        if M == 1 and first_run_time is not None:
            print(f"Используем результаты основного запуска для M=1")
            elapsed = first_run_time
        else:
            # Запускаем полный цикл: Модуль 2 + Модуль 1
            # МОДУЛЬ 2: Поиск начальных центров
            centers, time_module2 = find_initial_centers_parallel(test_data, R_excl, n_processes=M)

            # МОДУЛЬ 1: K-means кластеризация
            labels, centers, time_module1 = kmeans_clustering_parallel(test_data, centers, n_processes=M, max_iterations=50)

            elapsed = time_module2 + time_module1

        if M == 1:
            T1 = elapsed * 20

        M_values.append(M)
        real_times.append(elapsed * 1000)
        ideal_times.append(T1 * 50 / M if T1 else 0)

        print(f"\n>>> ИТОГО M={M}: Время вычислений={elapsed:.6f} сек, Идеальное={T1/M:.6f} сек, Ускорение={T1/elapsed:.2f}x")

    # Строим график
    plot_performance(M_values, real_times, ideal_times)

    return M_values, real_times, ideal_times


# ==================== ГЛАВНАЯ ПРОГРАММА ====================

if __name__ == "__main__":
    print("="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА: K-means кластеризация с параллелизацией")
    print("="*60)

    # ========== ПАРАМЕТРЫ ==========
    N_SAMPLES = 200         # Количество точек данных
    N_CLUSTERS_HINT = 4       # Ожидаемое количество кластеров (для генерации)
    R_EXCL = 10               # Радиус исключения для Модуля 2
    N_PROCESSES = multiprocessing.cpu_count()  # Количество процессов
    RANDOM_STATE = 42

    print(f"\nПараметры:")
    print(f"  Количество точек: {N_SAMPLES}")
    print(f"  Радиус исключения: {R_EXCL}")
    print(f"  Количество процессов: {N_PROCESSES}")

    # ========== ГЕНЕРАЦИЯ/ЗАГРУЗКА ДАННЫХ ==========
    print(f"\nГенерация синтетических данных...")
    data = generate_synthetic_data(N_SAMPLES, N_CLUSTERS_HINT, RANDOM_STATE)

    # Альтернатива: загрузка данных с Kaggle
    # data = load_kaggle_dataset('penguins.csv')  # https://www.kaggle.com/datasets/samoilovmikhail/2d-clustering-dataset-collection
    # if data is None:
    #     print("Использую синтетические данные")
    #     data = generate_synthetic_data(N_SAMPLES, N_CLUSTERS_HINT, RANDOM_STATE)

    # ========== ВИЗУАЛИЗАЦИЯ ИСХОДНЫХ ДАННЫХ ==========
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(data[:, 0], data[:, 1], s=30, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_title("Исходные данные", fontsize=20, fontweight='bold')
    ax.set_xlabel("X", fontsize=16, fontweight='bold')
    ax.set_ylabel("Y", fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ========== МОДУЛЬ 2: ПОИСК НАЧАЛЬНЫХ ЦЕНТРОВ ==========
    print(f"\n{'='*60}")
    print(f"ОСНОВНОЙ ЗАПУСК")
    print(f"{'='*60}")

    initial_centers, time_module2 = find_initial_centers_parallel(data, R_EXCL, N_PROCESSES)

    # ========== МОДУЛЬ 1: K-MEANS КЛАСТЕРИЗАЦИЯ ==========
    final_labels, final_centers, time_module1 = kmeans_clustering_parallel(data, initial_centers, N_PROCESSES)

    main_execution_time = time_module2 + time_module1

    print(f"\n{'='*60}")
    print(f"СУММАРНОЕ ВРЕМЯ ВЫЧИСЛЕНИЙ")
    print(f"{'='*60}")
    print(f"Модуль 2 (поиск центров): {time_module2:.6f} сек")
    print(f"Модуль 1 (K-means): {time_module1:.6f} сек")
    print(f"ИТОГО: {main_execution_time:.6f} сек")
    print(f"{'='*60}")

    # ========== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ==========
    plot_clusters(data, final_labels, final_centers,
                  title=f"Результат кластеризации (K={len(final_centers)})")

    # ========== СТАТИСТИКА ==========
    print(f"\n{'='*60}")
    print(f"ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*60}")
    print(f"Найдено кластеров: {len(final_centers)}")
    for i, center in enumerate(final_centers):
        n_points = np.sum(final_labels == i)
        print(f"  Кластер {i+1}: центр ({center[0]:.3f}, {center[1]:.3f}), точек = {n_points}")

    # ========== ЭКСПЕРИМЕНТ С ПАРАЛЛЕЛИЗАЦИЕЙ ==========
    print(f"\n{'='*60}")
    print(f"Подготовка к эксперименту с параллелизацией...")
    print(f"{'='*60}")

    # Используем меньший датасет для экспериментов
    test_data = data[:min(500, len(data))]

    # Для M=1 будем использовать результаты запуска на test_data
    print(f"\nЗапуск на тестовом датасете для получения T1 (1 процесс)...")
    test_centers, test_time2 = find_initial_centers_parallel(test_data, R_EXCL, n_processes=1)
    test_labels, test_centers, test_time1 = kmeans_clustering_parallel(test_data, test_centers, n_processes=1, max_iterations=50)
    T1_time = test_time2 + test_time1

    print(f"\nT1 (эталонное время для 1 процесса) = {T1_time:.6f} сек")

    # Запускаем эксперимент
    run_parallelization_experiment(test_data, R_EXCL, max_processes=24, first_run_time=T1_time)

    print("\n" + "="*60)
    print("ПРОГРАММА ЗАВЕРШЕНА")
    print("="*60)
