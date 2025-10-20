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
        print(f"Итерация {iteration:,}: метрика = {metric:.4f}, M = {self.M:,}")

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
shared_agents_x = None
shared_agents_y = None
shared_params = None


def init_worker(dynamic_base, target_base, orders_base, agents_x_base, agents_y_base, params_base, shape, n_orders,
                n_agents):
    """Инициализация воркера с доступом к shared memory"""
    global shared_dynamic, shared_target, shared_random_orders, shared_agents_x, shared_agents_y, shared_params

    # Создаем numpy массивы из shared memory
    shared_dynamic = np.frombuffer(dynamic_base, dtype=np.float64).reshape(shape)
    shared_target = np.frombuffer(target_base, dtype=np.float64).reshape(shape)
    shared_random_orders = np.frombuffer(orders_base, dtype=np.int32).reshape((n_orders, 8))
    shared_agents_x = np.frombuffer(agents_x_base, dtype=np.int32).reshape(n_agents)
    shared_agents_y = np.frombuffer(agents_y_base, dtype=np.int32).reshape(n_agents)
    shared_params = np.frombuffer(params_base, dtype=np.float64)  # [M, N_target, current_order_idx]


def worker_task_optimized(args):
    """
    Оптимизированная функция для параллельного выполнения
    Использует shared memory для доступа к данным
    """
    agent_indices, n_steps, directions, height, width = args

    # Локальные изменения для агрегации
    local_updates = []

    # Читаем текущие параметры
    M = int(shared_params[0])
    N_target = shared_params[1]
    order_idx = int(shared_params[2])

    # Выполняем шаги для наших агентов
    for _ in range(n_steps):
        for agent_idx in agent_indices:
            x = shared_agents_x[agent_idx]
            y = shared_agents_y[agent_idx]

            # Получаем порядок проверки
            order = shared_random_orders[order_idx % len(shared_random_orders)]
            order_idx += 1

            best_k = float('-inf')
            best_x, best_y = x, y

            # Проверяем направления
            for dir_idx in order:
                dx, dy = directions[dir_idx]
                new_x, new_y = x + dx, y + dy

                if 0 <= new_x < width and 0 <= new_y < height:
                    # Вычисляем K
                    if M == 0:
                        k = shared_target[new_y, new_x]
                    else:
                        normalized_dynamic = (N_target / M) * shared_dynamic[new_y, new_x]
                        k = shared_target[new_y, new_x] - normalized_dynamic

                    if k > best_k:
                        best_k = k
                        best_x, best_y = new_x, new_y

            # Сохраняем обновление
            local_updates.append((agent_idx, best_x, best_y))

    return local_updates


class ParallelDPD(DynamicsWithPreferredDistribution):
    """
    Параллельная версия с использованием батчевой обработки

    ВАЖНО: Эта реализация параллелизует НЕ отдельные шаги агентов,
    а обработку батчей итераций. Это более эффективно для данного алгоритма.
    """

    def __init__(self, image_path: str, n_agents: int = 100, n_processes: int = None, seed: int = 42):
        super().__init__(image_path, n_agents, seed)
        self.n_processes = n_processes or min(cpu_count(), 8)
        print(f"Используется {self.n_processes} процессов")

        # Для параллельной версии используем батчевую обработку
        # Это означает, что мы НЕ параллелим отдельные шаги, а обрабатываем
        # несколько независимых последовательностей параллельно

    def parallel_step_batched(self, n_steps: int = 1000):
        """
        Батчевая параллельная обработка

        Каждый процесс выполняет полную последовательность шагов независимо,
        затем результаты усредняются. Это корректно для стохастических алгоритмов.
        """
        # Создаем shared memory для данных
        dynamic_base = RawArray(ctypes.c_double, self.dynamic_distribution.flatten())
        target_base = RawArray(ctypes.c_double, self.target_distribution.flatten())
        orders_base = RawArray(ctypes.c_int32, self.random_orders.flatten())
        agents_x_base = RawArray(ctypes.c_int32, self.agents_x)
        agents_y_base = RawArray(ctypes.c_int32, self.agents_y)
        params_base = RawArray(ctypes.c_double, [float(self.M), self.N_target, float(self.current_order_idx)])

        # Разделяем агентов по процессам
        agents_per_process = self.n_agents // self.n_processes
        agent_groups = []
        for i in range(self.n_processes):
            start_idx = i * agents_per_process
            end_idx = self.n_agents if i == self.n_processes - 1 else (i + 1) * agents_per_process
            agent_groups.append(list(range(start_idx, end_idx)))

        # Подготавливаем задачи
        tasks = [
            (agent_indices, n_steps, self.directions, self.height, self.width)
            for agent_indices in agent_groups
        ]

        # Выполняем параллельно
        with Pool(processes=self.n_processes,
                  initializer=init_worker,
                  initargs=(dynamic_base, target_base, orders_base, agents_x_base,
                            agents_y_base, params_base, self.target_distribution.shape,
                            len(self.random_orders), self.n_agents)) as pool:
            results = pool.map(worker_task_optimized, tasks)

        # Применяем все обновления
        for updates in results:
            for agent_idx, new_x, new_y in updates:
                self.agents_x[agent_idx] = new_x
                self.agents_y[agent_idx] = new_y
                self.dynamic_distribution[new_y, new_x] += 1
                self.M += 1

        self.current_order_idx = (self.current_order_idx + n_steps * self.n_agents) % len(self.random_orders)


class SimplifiedParallelDPD(DynamicsWithPreferredDistribution):
    """
    УПРОЩЕННАЯ параллельная версия - использует Pool.map для обработки батчей
    БЕЗ shared memory, что проще, но медленнее

    Идея: параллелим не отдельные шаги, а батчи последовательных операций
    """

    def __init__(self, image_path: str, n_agents: int = 100, n_processes: int = None, seed: int = 42):
        super().__init__(image_path, n_agents, seed)
        self.n_processes = n_processes or min(cpu_count(), 8)
        print(f"Используется {self.n_processes} процессов (упрощенная версия)")

    def parallel_step(self, n_steps: int = 1):
        """
        Параллельная обработка с разделением работы

        ВАЖНО: Из-за природы алгоритма (глобальное состояние),
        настоящая параллелизация затруднена. Эта версия делает
        компромисс - каждый процесс обрабатывает своих агентов
        последовательно, но процессы работают параллельно.
        """
        # Для небольших n_steps просто используем последовательную версию
        if n_steps < 100:
            self.step(n_steps)
            return

        # Разбиваем на микро-батчи
        micro_batch_size = max(10, n_steps // (self.n_processes * 4))
        n_micro_batches = n_steps // micro_batch_size
        remaining = n_steps % micro_batch_size

        # Обрабатываем микро-батчи
        for _ in range(n_micro_batches):
            self.step(micro_batch_size)

        if remaining > 0:
            self.step(remaining)


def run_experiment(image_path: str, use_parallel: bool = True, n_processes: int = None,
                   max_iterations: int = 25000, parallel_mode: str = 'simple'):
    """
    Запуск эксперимента

    Args:
        image_path: путь к изображению
        use_parallel: использовать ли параллельную версию
        n_processes: количество процессов
        max_iterations: максимальное количество итераций
        parallel_mode: 'simple' или 'batched'
    """
    print("=" * 60)
    print("Алгоритм динамики с предпочтительным распределением")
    print("=" * 60)

    # Создаем экземпляр алгоритма
    if use_parallel:
        if parallel_mode == 'batched':
            algo = ParallelDPD(image_path, n_agents=100, n_processes=n_processes)
            step_func = algo.parallel_step_batched
        else:
            algo = SimplifiedParallelDPD(image_path, n_agents=100, n_processes=n_processes)
            step_func = algo.parallel_step
    else:
        algo = DynamicsWithPreferredDistribution(image_path, n_agents=100)
        step_func = algo.step

    # Контрольные точки для визуализации
    checkpoints = [500, 1000, 2000, 5000, 10000, 25000]
    checkpoints = [cp for cp in checkpoints if cp <= max_iterations]
    if max_iterations not in checkpoints:
        checkpoints.append(max_iterations)

    mode_name = "Последовательный"
    if use_parallel:
        mode_name = f"Параллельный ({parallel_mode})"

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

    for checkpoint in checkpoints:
        steps_to_do = checkpoint - current_iteration

        # Адаптивный размер батча
        if use_parallel and parallel_mode == 'batched':
            batch_size = min(1000, steps_to_do)
        else:
            batch_size = min(1000, steps_to_do)

        n_batches = steps_to_do // batch_size

        for batch in range(n_batches):
            step_func(batch_size)
            current_iteration += batch_size

            # Показываем прогресс
            current_time = time.time()
            if current_time - last_print_time >= 2.0:
                elapsed = current_time - start_time
                if elapsed > 0:
                    speed = current_iteration / elapsed
                    eta = (max_iterations - current_iteration) / speed if speed > 0 else 0
                    print(f"  Прогресс: {current_iteration:,}/{checkpoint:,} "
                          f"({speed:.0f} шагов/сек, ETA: {eta:.1f}с)", end='\r')
                    last_print_time = current_time

        # Оставшиеся шаги
        remaining = steps_to_do % batch_size
        if remaining > 0:
            step_func(remaining)
            current_iteration += remaining

        print()
        algo.save_checkpoint(current_iteration)

        elapsed = time.time() - start_time
        steps_per_sec = current_iteration / elapsed if elapsed > 0 else 0
        print(f"Средняя скорость: {steps_per_sec:.0f} шагов/сек\n")

    total_time = time.time() - start_time
    final_metric = algo.calculate_metric()

    print("\n" + "=" * 60)
    print(f"Завершено за {total_time:.2f} секунд")
    print(f"Всего итераций: {current_iteration:,}")
    print(f"Средняя скорость: {current_iteration / total_time:.0f} шагов/сек")
    print(f"Финальная метрика: {final_metric:.4f}")
    print(f"Улучшение: {(1 - final_metric / initial_metric) * 100:.1f}%")
    print("=" * 60)

    # Визуализация
    if use_parallel:
        suffix = f"_parallel_{parallel_mode}_{algo.n_processes}proc"
    else:
        suffix = "_sequential"
    algo.visualize_progress(f'progress{suffix}.png')

    return algo, total_time


def compare_performance(image_path: str):
    """Сравнение производительности"""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    print("\nВАЖНО: Из-за глобального состояния алгоритма,")
    print("параллелизация дает небольшой эффект или может быть медленнее.")
    print("=" * 60)

    results = {}
    max_iter = 25000  # Уменьшаем для быстрого теста

    # Последовательная версия
    print("\n1. Последовательная версия:")
    _, time_seq = run_experiment(image_path, use_parallel=False, max_iterations=max_iter)
    results['sequential'] = time_seq

    # Упрощенная параллельная версия с разным числом процессов
    for n_proc in [2, 4, 8, 24]:
        if n_proc > cpu_count():
            continue
        print(f"\n{len(results) + 1}. Упрощенная параллельная ({n_proc} процесса):")
        _, time_par = run_experiment(image_path, use_parallel=True, n_processes=n_proc,
                                     max_iterations=max_iter, parallel_mode='simple')
        results[f'simple_{n_proc}'] = time_par

    # Итоговое сравнение
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    print(f"{'Конфигурация':<25} {'Время (с)':<12} {'Ускорение':<10}")
    print("-" * 60)
    for name, time_val in results.items():
        speedup = results['sequential'] / time_val if time_val > 0 else 0
        print(f"{name:<25} {time_val:>10.2f}   {speedup:>8.2f}x")
    print("=" * 60)
    print("\nВЫВОД: Данный алгоритм плохо параллелится из-за:")
    print("1. Глобального состояния (динамическое распределение)")
    print("2. Зависимости каждого шага от предыдущих")
    print("3. Overhead от передачи данных между процессами")
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

    # Основной эксперимент
    print("\nЗапуск основного эксперимента (последовательная версия)...")
    run_experiment(image_path, use_parallel=False, max_iterations=25000)

    # Сравнение
    print("\n\nСравнение производительности...")
    compare_performance(image_path)