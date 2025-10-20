import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, RawArray, Lock
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

    def save_checkpoint(self, iteration: int):
        """Сохранить контрольную точку"""
        current_img = self.get_current_image()
        metric = self.calculate_metric()
        self.history.append({
            'iteration': iteration,
            'image': current_img.copy(),
            'metric': metric,
            'M': self.M
        })
        print(f"Checkpoint - Итерация {iteration:,}: метрика = {metric:.4f}, M = {self.M:,}")

    def visualize_progress(self, save_path: str = 'progress.png'):
        """Визуализация прогресса воспроизведения"""
        n_checkpoints = len(self.history)
        if n_checkpoints == 0:
            print("Нет данных для визуализации")
            return

        # Создаем сетку для отображения
        n_cols = min(4, n_checkpoints + 1)
        n_rows = (n_checkpoints + 2) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        # Показываем оригинал
        original_img = self.target_distribution - 1.0
        axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Оригинал')
        axes[0].axis('off')

        # Показываем промежуточные результаты
        for idx, checkpoint in enumerate(self.history):
            axes[idx + 1].imshow(checkpoint['image'], cmap='gray', vmin=0, vmax=255)
            axes[idx + 1].set_title(f"Iter: {checkpoint['iteration']:,}\nMetric: {checkpoint['metric']:.4f}")
            axes[idx + 1].axis('off')

        # Скрываем неиспользуемые оси
        for idx in range(n_checkpoints + 1, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация сохранена в {save_path}")
        plt.close()


# Глобальные переменные для shared memory
shared_dynamic = None
shared_target = None
shared_random_orders = None
shared_directions = None
shared_M_counter = None


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
    Использует локальные копии для вычислений, затем атомарно обновляет shared memory
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
                    # Вычисляем K на основе текущего состояния
                    if current_M == 0:
                        k = shared_target[new_y, new_x]
                    else:
                        # Читаем из shared memory + учитываем локальные обновления
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

    # Возвращаем обновления
    return agent_start, agent_end, local_agents_x, local_agents_y, local_dynamic_updates


class OptimizedParallelDPD(DynamicsWithPreferredDistribution):
    """
    Оптимизированная параллельная версия
    Использует shared memory + батчевую обработку для минимизации overhead
    """

    def __init__(self, image_path: str, n_agents: int = 100, n_processes: int = None, seed: int = 42):
        super().__init__(image_path, n_agents, seed)
        self.n_processes = n_processes or cpu_count()
        print(f"Используется {self.n_processes} процессов (оптимизированная версия)")

        # Создаем shared memory
        self._init_shared_memory()

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

    def parallel_step(self, n_steps: int = 100):
        """
        Параллельное выполнение n_steps шагов
        Разделяет агентов между процессами и обрабатывает батчами
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

        # Выполняем параллельно
        with Pool(
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
        ) as pool:
            results = pool.map(worker_step_batch, tasks)

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


def run_experiment(image_path: str, use_parallel: bool = True, n_processes: int = None,
                   max_iterations: int = 1000000):
    """
    Запуск эксперимента

    Args:
        image_path: путь к изображению
        use_parallel: использовать ли параллельную версию
        n_processes: количество процессов
        max_iterations: максимальное количество итераций
    """
    print("=" * 60)
    print("Алгоритм динамики с предпочтительным распределением")
    print("=" * 60)

    # Создаем экземпляр алгоритма
    if use_parallel:
        algo = OptimizedParallelDPD(image_path, n_agents=100, n_processes=n_processes)
        step_func = algo.parallel_step
        mode_name = "Параллельный (оптимизированный)"
    else:
        algo = DynamicsWithPreferredDistribution(image_path, n_agents=100)
        step_func = algo.step
        mode_name = "Последовательный"

    # Контрольные точки для визуализации
    checkpoints = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    checkpoints = [cp for cp in checkpoints if cp <= max_iterations]
    if max_iterations not in checkpoints and max_iterations > 1000:
        checkpoints.append(max_iterations)

    print(f"\nРежим: {mode_name}")
    print(f"Количество агентов: {algo.n_agents}")
    if use_parallel:
        print(f"Количество процессов: {algo.n_processes}")

    # Начальная метрика
    initial_metric = algo.calculate_metric()
    print(f"\nНачальная метрика: {initial_metric:.4f}")

    # Сохраняем начальное состояние
    algo.save_checkpoint(0)

    # Основной цикл
    start_time = time.time()
    current_iteration = 0
    last_print_time = start_time
    last_checkpoint_time = start_time

    print("\nНачало обработки...")
    print("-" * 60)

    for checkpoint_idx, checkpoint in enumerate(checkpoints):
        steps_to_do = checkpoint - current_iteration

        # Адаптивный размер батча в зависимости от режима
        if use_parallel:
            # Для параллельной версии используем большие батчи
            batch_size = min(1000, steps_to_do)
        else:
            # Для последовательной - меньшие батчи для частого обновления прогресса
            batch_size = min(1000, steps_to_do)

        n_batches = steps_to_do // batch_size

        for batch in range(n_batches):
            step_func(batch_size)
            current_iteration += batch_size

            # Показываем прогресс каждые 2 секунды
            current_time = time.time()
            if current_time - last_print_time >= 2.0:
                elapsed = current_time - start_time
                if elapsed > 0:
                    speed = current_iteration / elapsed
                    eta_checkpoint = (checkpoint - current_iteration) / speed if speed > 0 else 0
                    eta_total = (max_iterations - current_iteration) / speed if speed > 0 else 0

                    # Периодически вычисляем метрику
                    if batch % 10 == 0:
                        current_metric = algo.calculate_metric()
                        print(f"  Итерация {current_iteration:,}/{checkpoint:,} | "
                              f"Метрика: {current_metric:.4f} | "
                              f"Скорость: {speed:.0f} шаг/с | "
                              f"ETA: {eta_total:.0f}с")
                    else:
                        print(f"  Итерация {current_iteration:,}/{checkpoint:,} | "
                              f"Скорость: {speed:.0f} шаг/с | "
                              f"ETA: {eta_total:.0f}с")

                    last_print_time = current_time

        # Оставшиеся шаги
        remaining = steps_to_do % batch_size
        if remaining > 0:
            step_func(remaining)
            current_iteration += remaining

        # Сохраняем чекпоинт
        print()
        algo.save_checkpoint(current_iteration)

        # Статистика
        elapsed = time.time() - start_time
        checkpoint_time = time.time() - last_checkpoint_time
        steps_per_sec = current_iteration / elapsed if elapsed > 0 else 0

        print(f"  └─ Время с начала: {elapsed:.1f}с | "
              f"Время чекпоинта: {checkpoint_time:.1f}с | "
              f"Средняя скорость: {steps_per_sec:.0f} шаг/с")
        print("-" * 60)

        last_checkpoint_time = time.time()

    total_time = time.time() - start_time
    final_metric = algo.calculate_metric()

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Завершено за: {total_time:.2f} секунд")
    print(f"Всего итераций: {current_iteration:,}")
    print(f"Средняя скорость: {current_iteration / total_time:.0f} шагов/сек")
    print(f"Начальная метрика: {initial_metric:.4f}")
    print(f"Финальная метрика: {final_metric:.4f}")
    print(f"Улучшение: {(1 - final_metric / initial_metric) * 100:.1f}%")
    print("=" * 60)

    # Визуализация
    if use_parallel:
        suffix = f"_parallel_optimized_{algo.n_processes}proc"
    else:
        suffix = "_sequential"
    algo.visualize_progress(f'progress{suffix}.png')

    return algo, total_time


def compare_performance(image_path: str, max_iter: int = 100000):
    """Сравнение производительности различных конфигураций"""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)

    results = {}

    # Последовательная версия
    print("\n" + "=" * 60)
    print("1. Последовательная версия")
    print("=" * 60)
    _, time_seq = run_experiment(image_path, use_parallel=False, max_iterations=max_iter)
    results['sequential'] = time_seq

    # Параллельная версия с разным числом процессов
    available_cores = cpu_count()
    test_processes = [2, 4, 8, 16, available_cores]

    for n_proc in test_processes:
        if n_proc > available_cores:
            continue
        print("\n" + "=" * 60)
        print(f"{len(results) + 1}. Параллельная версия ({n_proc} процесса)")
        print("=" * 60)
        _, time_par = run_experiment(image_path, use_parallel=True, n_processes=n_proc,
                                     max_iterations=max_iter)
        results[f'parallel_{n_proc}'] = time_par

    # Итоговое сравнение
    print("\n" + "=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 60)
    print(f"{'Конфигурация':<25} {'Время (с)':<12} {'Ускорение':<10} {'Эффективность':<15}")
    print("-" * 70)

    for name, time_val in results.items():
        speedup = results['sequential'] / time_val if time_val > 0 else 0

        # Вычисляем эффективность (для параллельных версий)
        if name.startswith('parallel_'):
            n_cores = int(name.split('_')[1])
            efficiency = (speedup / n_cores) * 100
            efficiency_str = f"{efficiency:.1f}%"
        else:
            efficiency_str = "N/A"

        print(f"{name:<25} {time_val:>10.2f}   {speedup:>8.2f}x   {efficiency_str:>12}")

    print("=" * 70)

    # Анализ масштабируемости
    print("\nАНАЛИЗ МАСШТАБИРУЕМОСТИ:")
    parallel_results = {k: v for k, v in results.items() if k.startswith('parallel_')}
    if len(parallel_results) > 1:
        speedups = [(int(k.split('_')[1]), results['sequential'] / v)
                    for k, v in parallel_results.items()]
        speedups.sort()

        print(f"  Лучшее ускорение: {max(s[1] for s in speedups):.2f}x на {max(s, key=lambda x: x[1])[0]} процессах")
        print(f"  Слабое масштабирование: {'Да' if speedups[-1][1] > speedups[0][1] * 1.5 else 'Нет'}")

    print("=" * 60)


if __name__ == "__main__":
    image_path = "obama2.jpg"

    if not os.path.exists(image_path):
        print("Создаем тестовое изображение...")
        img = Image.new('L', (300, 300), color=255)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.ellipse([75, 75, 225, 225], fill=50, outline=0)
        draw.rectangle([120, 120, 180, 180], fill=200)
        img.save(image_path)
        print(f"Тестовое изображение сохранено в {image_path}")

    # Основной эксперимент с 1 миллионом итераций
    print("\nЗапуск основного эксперимента (1,000,000 итераций)...")
    run_experiment(image_path, use_parallel=False, max_iterations=1000000)

    # Сравнение производительности
    print("\n\nСравнение производительности (100,000 итераций для быстроты)...")
    compare_performance(image_path, max_iter=100000)
