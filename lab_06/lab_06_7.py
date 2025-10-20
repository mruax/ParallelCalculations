import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, RawArray
import time
from typing import Tuple, List
import os
import ctypes


class DynamicsWithPreferredDistribution:
    """
    Алгоритм динамики с предпочтительным распределением
    для воспроизведения изображений
    """

    def __init__(self, image_path: str, n_agents: int = 100, seed: int = 42):
        """
        Инициализация алгоритма

        Args:
            image_path: путь к изображению
            n_agents: количество агентов
            seed: seed для воспроизводимости
        """
        np.random.seed(seed)

        # Загрузка и подготовка эталонного распределения
        self.load_image(image_path)

        # Параметры
        self.n_agents = n_agents
        self.height, self.width = self.target_distribution.shape

        # Инициализация динамического распределения
        self.dynamic_distribution = np.zeros((self.height, self.width), dtype=np.float64)
        self.M = 0  # норма динамического распределения

        # Инициализация агентов в случайных позициях
        self.agents_x = np.random.randint(0, self.width, n_agents)
        self.agents_y = np.random.randint(0, self.height, n_agents)

        # Размещаем агентов на динамическом распределении
        for x, y in zip(self.agents_x, self.agents_y):
            self.dynamic_distribution[y, x] += 1
            self.M += 1

        # Подготовка случайных порядков проверки направлений
        self.prepare_random_orders(10000)
        self.current_order_idx = 0

        # 8 направлений: (dx, dy)
        self.directions = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1)
        ]

        # История для визуализации
        self.history = []

        # История метрик для графиков
        self.metric_history = []

    def load_image(self, image_path: str):
        """Загрузка изображения и создание эталонного распределения"""
        img = Image.open(image_path).convert('L')  # в grayscale

        # Изменяем размер если нужно
        max_size = 400
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            # Совместимость с разными версиями Pillow
            try:
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            except AttributeError:
                img = img.resize(new_size, Image.LANCZOS)

        # Преобразуем в numpy массив
        img_array = np.array(img, dtype=np.float64)

        # Нормализуем яркость к диапазону [1, 256]
        self.target_distribution = img_array + 1.0

        # Вычисляем норму эталонного распределения
        self.N_target = np.sum(self.target_distribution)

        print(f"Изображение загружено: {self.target_distribution.shape}")
        print(f"Норма эталонного распределения: {self.N_target:.2f}")

    def prepare_random_orders(self, n_orders: int = 10000):
        """Подготовка случайных порядков проверки направлений"""
        self.random_orders = np.zeros((n_orders, 8), dtype=np.int32)
        for i in range(n_orders):
            self.random_orders[i] = np.random.permutation(8)

    def get_next_order(self) -> np.ndarray:
        """Получить следующий случайный порядок проверки"""
        order = self.random_orders[self.current_order_idx]
        self.current_order_idx = (self.current_order_idx + 1) % len(self.random_orders)
        return order

    def calculate_k(self, x: int, y: int) -> float:
        """
        Вычисление коэффициента K для точки (x, y)
        K = n_target(x,y) - (N_target/M) * m(x,y)
        """
        if self.M == 0:
            return self.target_distribution[y, x]

        normalized_dynamic = (self.N_target / self.M) * self.dynamic_distribution[y, x]
        return self.target_distribution[y, x] - normalized_dynamic

    def move_agent(self, agent_idx: int) -> Tuple[int, int]:
        """
        Перемещение одного агента

        Returns:
            новые координаты (x, y)
        """
        x, y = self.agents_x[agent_idx], self.agents_y[agent_idx]

        # Получаем случайный порядок проверки направлений
        order = self.get_next_order()

        best_k = float('-inf')
        best_x, best_y = x, y

        # Проверяем все направления в случайном порядке
        for dir_idx in order:
            dx, dy = self.directions[dir_idx]
            new_x, new_y = x + dx, y + dy

            # Проверка границ
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                k = self.calculate_k(new_x, new_y)
                if k > best_k:
                    best_k = k
                    best_x, best_y = new_x, new_y

        return best_x, best_y

    def step(self, n_steps: int = 1):
        """Выполнить n_steps шагов алгоритма"""
        for _ in range(n_steps):
            # Перемещаем всех агентов последовательно
            for agent_idx in range(self.n_agents):
                new_x, new_y = self.move_agent(agent_idx)

                # Обновляем позицию агента
                self.agents_x[agent_idx] = new_x
                self.agents_y[agent_idx] = new_y

                # Обновляем динамическое распределение
                self.dynamic_distribution[new_y, new_x] += 1
                self.M += 1

    def calculate_metric(self) -> float:
        """
        Вычисление метрики различия между распределениями
        Используем нормированную среднеквадратичную ошибку
        """
        if self.M == 0:
            return float('inf')

        # Нормируем динамическое распределение
        normalized_dynamic = (self.N_target / self.M) * self.dynamic_distribution

        # Вычисляем относительную ошибку
        diff = np.abs(self.target_distribution - normalized_dynamic)
        relative_error = np.mean(diff / (self.target_distribution + 1e-10))

        return relative_error * 1000  # Масштабируем для читаемости

    def get_current_image(self) -> np.ndarray:
        """Получить текущее изображение из динамического распределения"""
        if self.M == 0:
            return np.zeros_like(self.target_distribution)

        # Нормируем динамическое распределение
        normalized = (self.N_target / self.M) * self.dynamic_distribution

        # Возвращаем как есть (без инверсии)
        result = np.clip(normalized - 1.0, 0, 255)

        return result

    def save_checkpoint(self, iteration: int, save_to_history: bool = True):
        """Сохранить контрольную точку"""
        current_img = self.get_current_image()
        metric = self.calculate_metric()

        if save_to_history:
            self.history.append({
                'iteration': iteration,
                'image': current_img.copy(),
                'metric': metric,
                'M': self.M
            })

        # Всегда сохраняем в metric_history для графиков
        self.metric_history.append({
            'iteration': iteration,
            'metric': metric
        })

        print(f"Checkpoint - Итерация {iteration:,}: метрика = {metric:.4f}, M = {self.M:,}")

    def visualize_progress(self, save_path: str = 'progress.png'):
        """Визуализация прогресса воспроизведения"""
        n_checkpoints = len(self.history)
        if n_checkpoints == 0:
            print("Нет данных для визуализации")
            return

        # Создаем сетку для отображения (добавляем +1 для оригинала)
        n_cols = min(4, n_checkpoints + 1)
        n_rows = (n_checkpoints + 1 + n_cols - 1) // n_cols  # Округление вверх

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        axes = axes.flatten()

        # Показываем оригинал
        original_img = self.target_distribution - 1.0
        axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Оригинал', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Показываем промежуточные результаты
        for idx, checkpoint in enumerate(self.history):
            axes[idx + 1].imshow(checkpoint['image'], cmap='gray', vmin=0, vmax=255)
            axes[idx + 1].set_title(f"Iter: {checkpoint['iteration']:,}\nMetric: {checkpoint['metric']:.4f}",
                                    fontsize=14, fontweight='bold')
            axes[idx + 1].axis('off')

        # Скрываем неиспользуемые оси
        for idx in range(n_checkpoints + 1, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация сохранена в {save_path}")
        plt.close()


# Глобальные переменные для shared memory и Pool
shared_dynamic = None
shared_target = None
shared_random_orders = None
shared_directions = None
shared_M_counter = None
global_pool = None


def init_worker_shared(dynamic_base, target_base, orders_base, M_counter_base, shape, n_orders, directions_tuple):
    """Инициализация воркера с доступом к shared memory"""
    global shared_dynamic, shared_target, shared_random_orders, shared_directions, shared_M_counter

    # Создаем numpy массивы из shared memory
    shared_dynamic = np.frombuffer(dynamic_base, dtype=np.float64).reshape(shape)
    shared_target = np.frombuffer(target_base, dtype=np.float64).reshape(shape)
    shared_random_orders = np.frombuffer(orders_base, dtype=np.int32).reshape((n_orders, 8))
    shared_M_counter = np.frombuffer(M_counter_base, dtype=np.int64)
    shared_directions = directions_tuple


def worker_step_batch(args):
    """
    Воркер обрабатывает батч агентов
    """
    agent_start, agent_end, agents_x, agents_y, n_steps, order_start_idx, N_target, height, width = args

    # Локальные копии позиций агентов
    local_agents_x = agents_x[agent_start:agent_end].copy()
    local_agents_y = agents_y[agent_start:agent_end].copy()

    # Локальное динамическое распределение для накопления изменений
    local_dynamic_updates = {}

    # Читаем текущее M
    current_M = int(shared_M_counter[0])

    order_idx = order_start_idx

    # Обрабатываем n_steps для наших агентов
    for step in range(n_steps):
        for local_idx in range(len(local_agents_x)):
            x = local_agents_x[local_idx]
            y = local_agents_y[local_idx]

            # Получаем порядок проверки
            order = shared_random_orders[order_idx % len(shared_random_orders)]
            order_idx += 1

            best_k = float('-inf')
            best_x, best_y = x, y

            # Проверяем направления
            for dir_idx in order:
                dx, dy = shared_directions[dir_idx]
                new_x, new_y = x + dx, y + dy

                if 0 <= new_x < width and 0 <= new_y < height:
                    # Вычисляем K
                    if current_M == 0:
                        k = shared_target[new_y, new_x]
                    else:
                        dynamic_val = shared_dynamic[new_y, new_x]
                        if (new_y, new_x) in local_dynamic_updates:
                            dynamic_val += local_dynamic_updates[(new_y, new_x)]

                        normalized_dynamic = (N_target / current_M) * dynamic_val
                        k = shared_target[new_y, new_x] - normalized_dynamic

                    if k > best_k:
                        best_k = k
                        best_x, best_y = new_x, new_y

            # Обновляем локальную позицию
            local_agents_x[local_idx] = best_x
            local_agents_y[local_idx] = best_y

            # Накапливаем изменения локально
            key = (best_y, best_x)
            local_dynamic_updates[key] = local_dynamic_updates.get(key, 0) + 1
            current_M += 1

    return agent_start, agent_end, local_agents_x, local_agents_y, local_dynamic_updates


class OptimizedParallelDPD(DynamicsWithPreferredDistribution):
    """
    Оптимизированная параллельная версия
    """

    def __init__(self, image_path: str, n_agents: int = 100, n_processes: int = None, seed: int = 42):
        super().__init__(image_path, n_agents, seed)
        self.n_processes = n_processes or cpu_count()
        print(f"Используется {self.n_processes} процессов (оптимизированная версия)")

        # Создаем shared memory
        self._init_shared_memory()

        # Создаем Pool один раз
        self.pool = None
        self._init_pool()

    def _init_shared_memory(self):
        """Инициализация shared memory"""
        # Динамическое распределение
        self.shared_dynamic_base = RawArray(ctypes.c_double, int(self.height * self.width))
        self.dynamic_distribution = np.frombuffer(
            self.shared_dynamic_base, dtype=np.float64
        ).reshape((self.height, self.width))

        # Целевое распределение
        self.shared_target_base = RawArray(ctypes.c_double, self.target_distribution.flatten().tolist())

        # Случайные порядки
        self.shared_orders_base = RawArray(ctypes.c_int32, self.random_orders.flatten().tolist())

        # Счетчик M
        self.shared_M_base = RawArray(ctypes.c_int64, 1)
        self.shared_M_base[0] = self.M

    def _init_pool(self):
        """Инициализация Pool (создается один раз)"""
        self.pool = Pool(
            processes=self.n_processes,
            initializer=init_worker_shared,
            initargs=(
                self.shared_dynamic_base,
                self.shared_target_base,
                self.shared_orders_base,
                self.shared_M_base,
                (self.height, self.width),
                len(self.random_orders),
                self.directions
            )
        )

    def parallel_step(self, n_steps: int = 100):
        """
        Параллельное выполнение n_steps шагов
        """
        # Обновляем M в shared memory
        self.shared_M_base[0] = self.M

        # Разделяем агентов между процессами
        agents_per_process = self.n_agents // self.n_processes
        tasks = []

        for i in range(self.n_processes):
            agent_start = i * agents_per_process
            if i == self.n_processes - 1:
                agent_end = self.n_agents
            else:
                agent_end = (i + 1) * agents_per_process

            order_start = (self.current_order_idx + i * 1000) % len(self.random_orders)

            tasks.append((
                agent_start,
                agent_end,
                self.agents_x,
                self.agents_y,
                n_steps,
                order_start,
                self.N_target,
                self.height,
                self.width
            ))

        # Выполняем параллельно с переиспользованием Pool
        results = self.pool.map(worker_step_batch, tasks)

        # Применяем обновления из всех процессов
        for agent_start, agent_end, new_agents_x, new_agents_y, dynamic_updates in results:
            # Обновляем позиции агентов
            self.agents_x[agent_start:agent_end] = new_agents_x
            self.agents_y[agent_start:agent_end] = new_agents_y

            # Применяем обновления динамического распределения
            for (y, x), count in dynamic_updates.items():
                self.dynamic_distribution[y, x] += count
                self.M += count

        # Обновляем индекс порядка
        self.current_order_idx = (self.current_order_idx + n_steps * self.n_agents) % len(self.random_orders)

    def __del__(self):
        """Закрываем Pool при удалении объекта"""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()


def run_single_experiment(image_path: str, n_processes: int, max_iterations: int,
                          checkpoint_iterations: list = None, verbose: bool = True):
    """
    Запуск одного эксперимента

    Args:
        checkpoint_iterations: список итераций для чекпоинтов

    Returns:
        tuple: (execution_time, algo_instance)
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Эксперимент: {n_processes} процесс(ов), {max_iterations:,} итераций")
        print(f"{'=' * 60}")

    # Создаем алгоритм
    if n_processes == 1:
        algo = DynamicsWithPreferredDistribution(image_path, n_agents=100)
        step_func = algo.step
        batch_size = 1000
    else:
        algo = OptimizedParallelDPD(image_path, n_agents=100, n_processes=n_processes)
        step_func = algo.parallel_step
        batch_size = 5000

    # Начальный чекпоинт
    algo.save_checkpoint(0, save_to_history=True)

    # Определяем чекпоинты
    if checkpoint_iterations is None:
        checkpoint_iterations = [max_iterations]

    checkpoint_set = set(checkpoint_iterations)

    start_time = time.time()
    current_iteration = 0
    last_print_time = start_time

    # Основной цикл
    while current_iteration < max_iterations:
        # Определяем размер шага
        remaining = max_iterations - current_iteration
        step_size = min(batch_size, remaining)

        step_func(step_size)
        current_iteration += step_size

        # Сохраняем чекпоинт если достигли нужной итерации
        if current_iteration in checkpoint_set:
            algo.save_checkpoint(current_iteration, save_to_history=True)

        # Вывод прогресса
        if verbose and time.time() - last_print_time >= 3.0:
            elapsed = time.time() - start_time
            speed = current_iteration / elapsed if elapsed > 0 else 0
            eta = (max_iterations - current_iteration) / speed if speed > 0 else 0
            print(f"  Прогресс: {current_iteration:,}/{max_iterations:,} | "
                  f"Скорость: {speed:.0f} шаг/с | ETA: {eta:.0f}с")
            last_print_time = time.time()

    total_time = time.time() - start_time

    if verbose:
        final_metric = algo.calculate_metric()
        print(f"\nЗавершено за {total_time:.2f}с")
        print(f"Финальная метрика: {final_metric:.4f}")
        print(f"Средняя скорость: {max_iterations / total_time:.0f} шаг/с")

    # Закрываем Pool если это параллельная версия
    if n_processes > 1:
        algo.pool.close()
        algo.pool.join()

    return total_time, algo


def experiment_1_scalability(image_path: str, max_processes: int = 24):
    """
    Эксперимент 1: Масштабируемость
    Чекпоинты: 0, 1000, 2500, 5000, 10000, 25000, 50000, 100000
    """
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 1: МАСШТАБИРУЕМОСТЬ (100,000 итераций)")
    print("=" * 70)

    iterations = 100000
    checkpoints = [0, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
    results = {}

    # Запускаем для каждого числа процессов
    for n_proc in range(1, max_processes + 1):
        print(f"\n--- Запуск на {n_proc} процессе(ах) ---")
        exec_time, algo = run_single_experiment(
            image_path,
            n_proc,
            iterations,
            checkpoint_iterations=checkpoints,
            verbose=True
        )
        results[n_proc] = exec_time

        # Сохраняем визуализацию только для некоторых конфигураций
        if n_proc in [1, 4, 8, 16, max_processes]:
            algo.visualize_progress(f'exp1_progress_{n_proc}proc.png')

    # Строим график
    plot_scalability(results, iterations)

    return results


def experiment_2_convergence(image_path: str, n_processes: int = 24):
    """
    Эксперимент 2: Сходимость
    Чекпоинты: каждые 500 до 50000, затем каждые 50000 до 1000000
    """
    print("\n" + "=" * 70)
    print(f"ЭКСПЕРИМЕНТ 2: СХОДИМОСТЬ (1,000,000 итераций на {n_processes} процессах)")
    print("=" * 70)

    max_iterations = 1000000

    # Чекпоинты: каждые 500 до 50000, затем каждые 50000
    checkpoints = list(range(0, 50001, 500)) + list(range(100000, 1000001, 50000))
    checkpoints = sorted(set(checkpoints))  # Убираем дубликаты и сортируем

    exec_time, algo = run_single_experiment(
        image_path,
        n_processes,
        max_iterations,
        checkpoint_iterations=checkpoints,
        verbose=True
    )

    # Строим график сходимости
    plot_convergence(algo.metric_history)

    # Сохраняем визуализацию
    algo.visualize_progress('exp2_progress_convergence.png')

    return exec_time, algo


def plot_scalability(results: dict, iterations: int):
    """
    График масштабируемости (только время)
    """
    n_processes = sorted(results.keys())
    times = [results[n] for n in n_processes]

    # Идеальное ускорение
    t1 = times[0]
    ideal_times = [t1 / n for n in n_processes]

    # Создаем фигуру с одним графиком
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # График: Время выполнения
    ax.plot(n_processes, times, 'bo-', linewidth=3, markersize=10, label='Реальное время')
    ax.plot(n_processes, ideal_times, 'r--', linewidth=3, label='Идеальное время (T₁/M)')
    ax.set_xlabel('Количество процессов', fontsize=14, fontweight='bold')
    ax.set_ylabel('Время выполнения (секунды)', fontsize=14, fontweight='bold')
    ax.set_title(f'Зависимость времени от числа процессов\n({iterations:,} итераций)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, prop={'weight': 'bold'})

    # Делаем метки осей жирными
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Добавляем значения на график
    for i, (n, t) in enumerate(zip(n_processes, times)):
        if i % 3 == 0 or n == n_processes[-1]:
            ax.annotate(f'{t:.1f}с',
                        xy=(n, t),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=12,
                        fontweight='bold')

    plt.tight_layout()
    plt.savefig('experiment1_scalability.png', dpi=150, bbox_inches='tight')
    print(f"\nГрафик масштабируемости сохранен: experiment1_scalability.png")
    plt.close()

    # Выводим таблицу результатов
    speedups = [t1 / t for t in times]
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА 1")
    print("=" * 70)
    print(f"{'Процессы':<12} {'Время (с)':<12} {'Ускорение':<12} {'Эффективность':<15}")
    print("-" * 70)
    for n, t, s in zip(n_processes, times, speedups):
        efficiency = (s / n) * 100
        print(f"{n:<12} {t:>10.2f}   {s:>10.2f}x   {efficiency:>12.1f}%")
    print("=" * 70)


def plot_convergence(metric_history: list):
    """
    График сходимости
    """
    iterations = [item['iteration'] for item in metric_history]
    metrics = [item['metric'] for item in metric_history]

    # Создаем график
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(iterations, metrics, 'b-', linewidth=3, marker='o', markersize=4)
    ax.set_xlabel('Итерации', fontsize=14, fontweight='bold')
    ax.set_ylabel('Метрика ошибки', fontsize=14, fontweight='bold')
    ax.set_title('Сходимость алгоритма: зависимость метрики от итераций',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Логарифмическая шкала для итераций
    ax.set_xscale('log')

    # Делаем метки осей жирными
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Добавляем значения на ключевых точках
    key_indices = [0, len(iterations) // 4, len(iterations) // 2, 3 * len(iterations) // 4, len(iterations) - 1]
    for idx in key_indices:
        if idx < len(iterations):
            ax.annotate(f'{metrics[idx]:.2f}',
                        xy=(iterations[idx], metrics[idx]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=12,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('experiment2_convergence.png', dpi=150, bbox_inches='tight')
    print(f"\nГрафик сходимости сохранен: experiment2_convergence.png")
    plt.close()

    # Выводим
    # Выводим таблицу результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА 2")
    print("=" * 70)
    print(f"{'Итерация':<15} {'Метрика':<12} {'Улучшение от начала':<20}")
    print("-" * 70)

    initial_metric = metrics[0]
    for i, m in zip(iterations, metrics):
        improvement = ((initial_metric - m) / initial_metric) * 100
        print(f"{i:<15,} {m:>10.4f}   {improvement:>18.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    image_path = "obama2.jpg"

    # Проверяем наличие изображения
    if not os.path.exists(image_path):
        print("Создаем тестовое изображение...")
        img = Image.new('L', (300, 300), color=255)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.ellipse([75, 75, 225, 225], fill=50, outline=0)
        draw.rectangle([120, 120, 180, 180], fill=200)
        img.save(image_path)
        print(f"Тестовое изображение сохранено в {image_path}")

    # ЭКСПЕРИМЕНТ 1: Масштабируемость
    print("\n" + "#" * 70)
    print("# ЗАПУСК ЭКСПЕРИМЕНТА 1: МАСШТАБИРУЕМОСТЬ")
    print("#" * 70)

    scalability_results = experiment_1_scalability(image_path, max_processes=24)

    # ЭКСПЕРИМЕНТ 2: Сходимость
    print("\n\n" + "#" * 70)
    print("# ЗАПУСК ЭКСПЕРИМЕНТА 2: СХОДИМОСТЬ")
    print("#" * 70)

    convergence_time, convergence_algo = experiment_2_convergence(image_path, n_processes=24)

    # Итоговая сводка
    print("\n\n" + "=" * 70)
    print("ИТОГОВАЯ СВОДКА ЭКСПЕРИМЕНТОВ")
    print("=" * 70)

    print("\nЭксперимент 1 (Масштабируемость):")
    print(f"  - Итерации: 100,000")
    print(f"  - Диапазон процессов: 1-24")
    t1 = scalability_results[1]
    t24 = scalability_results[24]
    speedup = t1 / t24
    efficiency = (speedup / 24) * 100
    print(f"  - Время на 1 процессе: {t1:.2f}с")
    print(f"  - Время на 24 процессах: {t24:.2f}с")
    print(f"  - Ускорение: {speedup:.2f}x")
    print(f"  - Эффективность: {efficiency:.1f}%")

    # Находим оптимальное количество процессов
    best_speedup = max(t1 / scalability_results[n] for n in scalability_results.keys())
    best_n_proc = [n for n in scalability_results.keys() if abs((t1 / scalability_results[n]) - best_speedup) < 0.01][0]
    print(f"  - Оптимальное число процессов: {best_n_proc} (ускорение {best_speedup:.2f}x)")

    print("\nЭксперимент 2 (Сходимость):")
    print(f"  - Итерации: 1,000,000")
    print(f"  - Процессы: 24")
    print(f"  - Время выполнения: {convergence_time:.2f}с")
    print(f"  - Средняя скорость: {1000000 / convergence_time:.0f} итераций/с")

    initial_metric = convergence_algo.metric_history[0]['metric']
    final_metric = convergence_algo.metric_history[-1]['metric']
    improvement = ((initial_metric - final_metric) / initial_metric) * 100
    print(f"  - Начальная метрика: {initial_metric:.4f}")
    print(f"  - Финальная метрика: {final_metric:.4f}")
    print(f"  - Улучшение: {improvement:.1f}%")

    print("\n" + "=" * 70)
    print("Все эксперименты завершены!")
    print("=" * 70)

    print("\nСохраненные файлы:")
    print("  - experiment1_scalability.png - график масштабируемости")
    print("  - experiment2_convergence.png - график сходимости")
    print("  - exp1_progress_*proc.png - визуализация прогресса (эксперимент 1)")
    print("  - exp2_progress_convergence.png - визуализация прогресса (эксперимент 2)")

    print("\n" + "=" * 70)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 70)

    # Анализ масштабируемости
    print("\n1. Масштабируемость:")
    speedups = [t1 / scalability_results[n] for n in range(1, 25)]
    efficiencies = [(speedups[n - 1] / n) * 100 for n in range(1, 25)]

    # Находим когда эффективность падает ниже 50%
    threshold_50 = next((n for n in range(1, 25) if efficiencies[n - 1] < 50), 24)
    print(f"   - Эффективность >50% до {threshold_50} процессов")

    # Средняя эффективность на разных диапазонах
    avg_eff_small = np.mean(efficiencies[:4])  # 1-4 процесса
    avg_eff_medium = np.mean(efficiencies[4:8])  # 5-8 процессов
    avg_eff_large = np.mean(efficiencies[8:])  # 9-24 процесса

    print(f"   - Средняя эффективность (1-4 процесса): {avg_eff_small:.1f}%")
    print(f"   - Средняя эффективность (5-8 процессов): {avg_eff_medium:.1f}%")
    print(f"   - Средняя эффективность (9-24 процесса): {avg_eff_large:.1f}%")

    # Анализ сходимости
    print("\n2. Сходимость:")
    metrics = [item['metric'] for item in convergence_algo.metric_history]
    iterations_conv = [item['iteration'] for item in convergence_algo.metric_history]

    # Скорость сходимости (изменение метрики на 100k итераций)
    convergence_rates = []
    for i in range(1, len(metrics)):
        iter_diff = iterations_conv[i] - iterations_conv[i - 1]
        metric_diff = metrics[i - 1] - metrics[i]
        rate = (metric_diff / iter_diff) * 100000  # на 100k итераций
        convergence_rates.append(rate)

    print(f"   - Начальная скорость сходимости: {convergence_rates[0]:.4f} на 100k итераций")
    print(f"   - Конечная скорость сходимости: {convergence_rates[-1]:.4f} на 100k итераций")

    # Когда достигается 80% улучшения
    target_metric = initial_metric - 0.8 * (initial_metric - final_metric)
    iter_80 = next((iterations_conv[i] for i, m in enumerate(metrics) if m <= target_metric), iterations_conv[-1])
    print(f"   - 80% улучшения достигается на итерации: {iter_80:,}")

    print("\n" + "=" * 70)