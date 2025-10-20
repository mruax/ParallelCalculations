import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
from typing import Tuple, List
import os
import math


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

        # Ограничиваем количество отображаемых чекпоинтов
        max_display = 11  # Оригинал + 10 чекпоинтов максимум
        if n_checkpoints + 1 > max_display:
            # Выбираем равномерно распределенные чекпоинты
            indices = np.linspace(0, n_checkpoints - 1, max_display - 1, dtype=int)
            display_history = [self.history[i] for i in indices]
            n_checkpoints = len(display_history)
        else:
            display_history = self.history

        # Создаем сетку для отображения
        n_cols = min(4, n_checkpoints + 1)
        n_rows = (n_checkpoints + 1 + n_cols - 1) // n_cols  # Округление вверх

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

        # Обрабатываем случай одной строки
        if n_rows == 1:
            if n_cols == 1:
                axes = np.array([[axes]])
            else:
                axes = axes.reshape(1, -1)

        axes = axes.flatten()

        # Показываем оригинал
        original_img = self.target_distribution - 1.0
        axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Оригинал')
        axes[0].axis('off')

        # Показываем промежуточные результаты
        for idx, checkpoint in enumerate(display_history):
            if idx + 1 < len(axes):  # Проверка границ
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


def process_region(args):
    """
    Обработка одного региона изображения в отдельном процессе

    Args:
        args: (region_id, target_region, n_agents_region, n_steps, random_orders, seed)

    Returns:
        (region_id, dynamic_region, agents_x, agents_y, M_region)
    """
    region_id, target_region, n_agents_region, n_steps, random_orders, seed = args

    # Устанавливаем seed для воспроизводимости
    np.random.seed(seed + region_id)

    height, width = target_region.shape
    N_target = np.sum(target_region)

    # Инициализация динамического распределения для региона
    dynamic_region = np.zeros((height, width), dtype=np.float64)
    M = 0

    # Инициализация агентов в случайных позициях внутри региона
    agents_x = np.random.randint(0, width, n_agents_region)
    agents_y = np.random.randint(0, height, n_agents_region)

    # Размещаем агентов
    for x, y in zip(agents_x, agents_y):
        dynamic_region[y, x] += 1
        M += 1

    # 8 направлений
    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1)
    ]

    order_idx = 0

    # Выполняем шаги
    for step in range(n_steps):
        for agent_idx in range(n_agents_region):
            x, y = agents_x[agent_idx], agents_y[agent_idx]

            # Получаем случайный порядок проверки
            order = random_orders[order_idx % len(random_orders)]
            order_idx += 1

            best_k = float('-inf')
            best_x, best_y = x, y

            # Проверяем направления
            for dir_idx in order:
                dx, dy = directions[dir_idx]
                new_x, new_y = x + dx, y + dy

                # Проверка границ региона
                if 0 <= new_x < width and 0 <= new_y < height:
                    # Вычисляем K
                    if M == 0:
                        k = target_region[new_y, new_x]
                    else:
                        normalized_dynamic = (N_target / M) * dynamic_region[new_y, new_x]
                        k = target_region[new_y, new_x] - normalized_dynamic

                    if k > best_k:
                        best_k = k
                        best_x, best_y = new_x, new_y

            # Обновляем позицию
            agents_x[agent_idx] = best_x
            agents_y[agent_idx] = best_y
            dynamic_region[best_y, best_x] += 1
            M += 1

    return region_id, dynamic_region, agents_x, agents_y, M


class RegionParallelDPD:
    """
    Параллельная версия с разделением изображения на регионы
    """

    def __init__(self, image_path: str, n_agents: int = 100, n_processes: int = 4, seed: int = 42):
        """
        Args:
            image_path: путь к изображению
            n_agents: общее количество агентов (будет распределено по регионам)
            n_processes: количество процессов (должно быть квадратом: 4, 9, 16, ...)
            seed: seed для воспроизводимости
        """
        self.seed = seed
        np.random.seed(seed)

        # Проверяем, что n_processes является квадратом
        self.grid_size = int(math.sqrt(n_processes))
        if self.grid_size * self.grid_size != n_processes:
            # Если не квадрат, находим ближайший квадрат
            self.grid_size = int(math.sqrt(n_processes))
            n_processes = self.grid_size * self.grid_size
            print(f"⚠️  Скорректировано количество процессов до {n_processes} ({self.grid_size}x{self.grid_size})")

        self.n_processes = n_processes

        # Загрузка изображения
        self.load_image(image_path)

        # Параметры
        self.n_agents = n_agents
        self.height, self.width = self.target_distribution.shape

        # Подготовка случайных порядков
        self.random_orders = np.zeros((10000, 8), dtype=np.int32)
        for i in range(10000):
            self.random_orders[i] = np.random.permutation(8)

        # Разделяем изображение на регионы
        self.setup_regions()

        # История
        self.history = []

        print(f"✓ Инициализация завершена:")
        print(f"  - Изображение: {self.height}x{self.width}")
        print(f"  - Сетка регионов: {self.grid_size}x{self.grid_size} = {self.n_processes} регионов")
        print(f"  - Размер региона: {self.region_height}x{self.region_width}")
        print(f"  - Агентов на регион: {self.agents_per_region}")
        print(f"  - Всего агентов: {self.n_agents}")

    def load_image(self, image_path: str):
        """Загрузка изображения"""
        img = Image.open(image_path).convert('L')

        max_size = 400
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            try:
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            except AttributeError:
                img = img.resize(new_size, Image.LANCZOS)

        img_array = np.array(img, dtype=np.float64)
        self.target_distribution = img_array + 1.0
        self.N_target = np.sum(self.target_distribution)

    def setup_regions(self):
        """Разделение изображения на регионы"""
        # Вычисляем размеры региона
        self.region_height = self.height // self.grid_size
        self.region_width = self.width // self.grid_size

        # Агентов на регион
        self.agents_per_region = self.n_agents // self.n_processes

        # Создаем регионы
        self.regions = []
        region_id = 0

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Границы региона
                y_start = row * self.region_height
                y_end = (row + 1) * self.region_height if row < self.grid_size - 1 else self.height
                x_start = col * self.region_width
                x_end = (col + 1) * self.region_width if col < self.grid_size - 1 else self.width

                # Извлекаем регион
                target_region = self.target_distribution[y_start:y_end, x_start:x_end].copy()

                self.regions.append({
                    'id': region_id,
                    'row': row,
                    'col': col,
                    'y_start': y_start,
                    'y_end': y_end,
                    'x_start': x_start,
                    'x_end': x_end,
                    'target': target_region,
                    'dynamic': None,
                    'M': 0
                })

                region_id += 1

    def parallel_step(self, n_steps: int = 1000):
        """
        Параллельное выполнение шагов
        Каждый регион обрабатывается независимо
        """
        # Подготавливаем задачи для каждого региона
        tasks = []
        for region in self.regions:
            task = (
                region['id'],
                region['target'],
                self.agents_per_region,
                n_steps,
                self.random_orders,
                self.seed
            )
            tasks.append(task)

        # Выполняем параллельно
        with Pool(processes=self.n_processes) as pool:
            results = pool.map(process_region, tasks)

        # Собираем результаты
        for region_id, dynamic_region, agents_x, agents_y, M_region in results:
            self.regions[region_id]['dynamic'] = dynamic_region
            self.regions[region_id]['M'] = M_region

    def get_full_dynamic_distribution(self) -> np.ndarray:
        """Склеивание динамического распределения из регионов"""
        full_dynamic = np.zeros((self.height, self.width), dtype=np.float64)

        for region in self.regions:
            if region['dynamic'] is not None:
                y_start, y_end = region['y_start'], region['y_end']
                x_start, x_end = region['x_start'], region['x_end']
                full_dynamic[y_start:y_end, x_start:x_end] = region['dynamic']

        return full_dynamic

    def get_total_M(self) -> int:
        """Общая норма динамического распределения"""
        return sum(region['M'] for region in self.regions)

    def calculate_metric(self) -> float:
        """Вычисление метрики"""
        full_dynamic = self.get_full_dynamic_distribution()
        M_total = self.get_total_M()

        if M_total == 0:
            return float('inf')

        normalized_dynamic = (self.N_target / M_total) * full_dynamic
        diff = np.abs(self.target_distribution - normalized_dynamic)
        relative_error = np.mean(diff / (self.target_distribution + 1e-10))

        return relative_error * 1000

    def get_current_image(self) -> np.ndarray:
        """Получить текущее изображение"""
        full_dynamic = self.get_full_dynamic_distribution()
        M_total = self.get_total_M()

        if M_total == 0:
            return np.zeros_like(self.target_distribution)

        normalized = (self.N_target / M_total) * full_dynamic
        result = np.clip(normalized - 1.0, 0, 255)

        return result

    def save_checkpoint(self, iteration: int):
        """Сохранить контрольную точку"""
        current_img = self.get_current_image()
        metric = self.calculate_metric()
        M_total = self.get_total_M()

        self.history.append({
            'iteration': iteration,
            'image': current_img.copy(),
            'metric': metric,
            'M': M_total
        })
        print(f"Checkpoint - Итерация {iteration:,}: метрика = {metric:.4f}, M = {M_total:,}")

    def visualize_progress(self, save_path: str = 'progress.png'):
        """Визуализация прогресса"""
        n_checkpoints = len(self.history)
        if n_checkpoints == 0:
            print("Нет данных для визуализации")
            return

        # Ограничиваем количество отображаемых чекпоинтов
        max_display = 11  # Оригинал + 10 чекпоинтов максимум
        if n_checkpoints + 1 > max_display:
            # Выбираем равномерно распределенные чекпоинты
            indices = np.linspace(0, n_checkpoints - 1, max_display - 1, dtype=int)
            display_history = [self.history[i] for i in indices]
            n_checkpoints = len(display_history)
        else:
            display_history = self.history

        # Создаем сетку для отображения
        n_cols = min(4, n_checkpoints + 1)
        n_rows = (n_checkpoints + 1 + n_cols - 1) // n_cols  # Округление вверх

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

        # Обрабатываем случай одной строки
        if n_rows == 1:
            if n_cols == 1:
                axes = np.array([[axes]])
            else:
                axes = axes.reshape(1, -1)

        axes = axes.flatten()

        # Оригинал
        original_img = self.target_distribution - 1.0
        axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Оригинал')
        axes[0].axis('off')

        # Промежуточные результаты
        for idx, checkpoint in enumerate(display_history):
            if idx + 1 < len(axes):  # Проверка границ
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

    def visualize_regions(self, save_path: str = 'regions.png'):
        """Визуализация разделения на регионы"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Оригинал
        axes[0].imshow(self.target_distribution - 1.0, cmap='gray')
        axes[0].set_title('Оригинальное изображение')
        axes[0].axis('off')

        # С границами регионов
        img_with_grid = (self.target_distribution - 1.0).copy()
        axes[1].imshow(img_with_grid, cmap='gray')

        # Рисуем сетку
        for i in range(1, self.grid_size):
            y = i * self.region_height
            axes[1].axhline(y=y, color='red', linewidth=2, alpha=0.7)
            x = i * self.region_width
            axes[1].axvline(x=x, color='red', linewidth=2, alpha=0.7)

        axes[1].set_title(f'Разделение на {self.n_processes} регионов ({self.grid_size}x{self.grid_size})')
        axes[1].axis('off')

        # Текущее динамическое распределение
        current_img = self.get_current_image()
        axes[2].imshow(current_img, cmap='gray')
        axes[2].set_title('Текущее состояние')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация регионов сохранена в {save_path}")
        plt.close()


def run_region_experiment(image_path: str, n_processes: int = 4, max_iterations: int = 1000000):
    """
    Запуск эксперимента с региональной параллелизацией
    """
    print("=" * 60)
    print("РЕГИОНАЛЬНАЯ ПАРАЛЛЕЛИЗАЦИЯ")
    print("=" * 60)

    # Создаем экземпляр
    algo = RegionParallelDPD(image_path, n_agents=100, n_processes=n_processes)

    # Визуализируем разделение
    algo.visualize_regions('regions_initial.png')

    # Контрольные точки
    checkpoints = [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 500000, 1000000]
    checkpoints = [cp for cp in checkpoints if cp <= max_iterations]
    if max_iterations not in checkpoints:
        checkpoints.append(max_iterations)

    # Начальная метрика
    algo.parallel_step(1)  # Один шаг для инициализации
    initial_metric = algo.calculate_metric()
    print(f"\nНачальная метрика: {initial_metric:.4f}")

    algo.save_checkpoint(0)

    # Основной цикл
    start_time = time.time()
    current_iteration = 0
    last_print_time = start_time

    print("\nНачало обработки...")
    print("-" * 60)

    for checkpoint in checkpoints:
        steps_to_do = checkpoint - current_iteration

        # Размер батча (больше для параллельной версии)
        batch_size = 5000
        n_batches = steps_to_do // batch_size

        for batch in range(n_batches):
            algo.parallel_step(batch_size)
            current_iteration += batch_size

            # Прогресс
            current_time = time.time()
            if current_time - last_print_time >= 2.0:
                elapsed = current_time - start_time
                if elapsed > 0:
                    speed = current_iteration / elapsed
                    eta = (checkpoint - current_iteration) / speed if speed > 0 else 0
                    current_metric = algo.calculate_metric()

                    print(f"  Итерация {current_iteration:,}/{checkpoint:,} | "
                          f"Метрика: {current_metric:.4f} | "
                          f"Скорость: {speed:.0f} шаг/с | "
                          f"ETA: {eta:.1f}с")

                    last_print_time = current_time

        # Оставшиеся шаги
        remaining = steps_to_do % batch_size
        if remaining > 0:
            algo.parallel_step(remaining)
            current_iteration += remaining

        algo.save_checkpoint(current_iteration)

        elapsed = time.time() - start_time
        steps_per_sec = current_iteration / elapsed if elapsed > 0 else 0
        print(f"  └─ Средняя скорость: {steps_per_sec:.0f} шаг/с")
        print("-" * 60)

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
    algo.visualize_progress(f'progress_regions_{n_processes}proc.png')
    algo.visualize_regions(f'regions_final_{n_processes}proc.png')

    return algo, total_time


def compare_region_parallelization(image_path: str):
    """Сравнение разных конфигураций региональной параллелизации"""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ РЕГИОНАЛЬНОЙ ПАРАЛЛЕЛИЗАЦИИ")
    print("="*60)

    results = {}
    max_iter = 100000

    # Тестируем разные конфигурации
    configs = [
        (1, "Последовательно (1 регион)"),
        (4, "Параллельно (2x2 = 4 региона)"),
        (9, "Параллельно (3x3 = 9 регионов)"),
        (16, "Параллельно (4x4 = 16 регионов)"),
    ]

    for n_proc, description in configs:
        if n_proc > cpu_count() and n_proc > 1:
            print(f"\n⚠️  Пропускаем {description} - недостаточно ядер")
            continue

        print("\n" + "="*60)
        print(f"{len(results)+1}. {description}")
        print("="*60)

        _, time_val = run_region_experiment(image_path, n_processes=n_proc, max_iterations=max_iter)
        results[f'{n_proc}_regions'] = time_val

    # Итоговое сравнение
    print("\n" + "="*60)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*60)
    print(f"{'Конфигурация':<30} {'Время (с)':<12} {'Ускорение':<10}")
    print("-" * 60)

    baseline_time = results.get('1_regions', 0)
    for name, time_val in results.items():
        speedup = baseline_time / time_val if time_val > 0 and baseline_time > 0 else 0
        n_regions = name.split('_')[0]
        grid_size = int(math.sqrt(int(n_regions)))
        display_name = f"{n_regions} регионов ({grid_size}x{grid_size})"
        print(f"{display_name:<30} {time_val:>10.2f}   {speedup:>8.2f}x")

    print("="*60)

    # Визуализация сравнения
    visualize_performance_comparison(results)


def visualize_performance_comparison(results: dict):
    """Визуализация сравнения производительности"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Извлекаем данные
    n_regions_list = []
    times = []
    speedups = []

    baseline_time = results.get('1_regions', None)

    for name, time_val in sorted(results.items(), key=lambda x: int(x[0].split('_')[0])):
        n_regions = int(name.split('_')[0])
        n_regions_list.append(n_regions)
        times.append(time_val)

        if baseline_time and baseline_time > 0:
            speedup = baseline_time / time_val
            speedups.append(speedup)
        else:
            speedups.append(1.0)

    # График 1: Время выполнения
    ax1.plot(n_regions_list, times, 'bo-', linewidth=2, markersize=8, label='Фактическое время')
    ax1.set_xlabel('Количество регионов', fontsize=12)
    ax1.set_ylabel('Время выполнения (секунды)', fontsize=12)
    ax1.set_title('Время выполнения vs Количество регионов', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Добавляем значения на точки
    for x, y in zip(n_regions_list, times):
        ax1.annotate(f'{y:.1f}с', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    # График 2: Ускорение
    ax2.plot(n_regions_list, speedups, 'go-', linewidth=2, markersize=8, label='Фактическое ускорение')

    # Идеальное ускорение (линейное)
    if n_regions_list:
        ideal_speedups = [n / n_regions_list[0] for n in n_regions_list]
        ax2.plot(n_regions_list, ideal_speedups, 'r--', linewidth=2, alpha=0.5, label='Идеальное ускорение')

    ax2.set_xlabel('Количество регионов', fontsize=12)
    ax2.set_ylabel('Ускорение (раз)', fontsize=12)
    ax2.set_title('Ускорение vs Количество регионов', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=1, color='k', linestyle=':', alpha=0.3)

    # Добавляем значения на точки
    for x, y in zip(n_regions_list, speedups):
        ax2.annotate(f'{y:.2f}x', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nГрафик сравнения сохранен в performance_comparison.png")
    plt.close()


def demonstrate_region_splitting(image_path: str):
    """
    Демонстрация того, как изображение разделяется на регионы
    """
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ РАЗДЕЛЕНИЯ НА РЕГИОНЫ")
    print("="*60)

    # Загружаем изображение
    img = Image.open(image_path).convert('L')
    max_size = 400
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        try:
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        except AttributeError:
            img = img.resize(new_size, Image.LANCZOS)

    img_array = np.array(img, dtype=np.float64)

    # Показываем разные варианты разделения
    configs = [4, 9, 16]
    fig, axes = plt.subplots(1, len(configs) + 1, figsize=(5 * (len(configs) + 1), 5))

    # Оригинал
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Оригинал', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Разные сетки
    for idx, n_proc in enumerate(configs):
        grid_size = int(math.sqrt(n_proc))
        height, width = img_array.shape
        region_height = height // grid_size
        region_width = width // grid_size

        axes[idx + 1].imshow(img_array, cmap='gray')

        # Рисуем сетку
        for i in range(1, grid_size):
            y = i * region_height
            axes[idx + 1].axhline(y=y, color='red', linewidth=2, alpha=0.8)
            x = i * region_width
            axes[idx + 1].axvline(x=x, color='red', linewidth=2, alpha=0.8)

        # Нумеруем регионы
        region_id = 0
        for row in range(grid_size):
            for col in range(grid_size):
                y_center = row * region_height + region_height // 2
                x_center = col * region_width + region_width // 2
                axes[idx + 1].text(x_center, y_center, str(region_id),
                                  fontsize=20, fontweight='bold', color='yellow',
                                  ha='center', va='center',
                                  bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
                region_id += 1

        axes[idx + 1].set_title(f'{n_proc} регионов ({grid_size}x{grid_size})',
                               fontsize=14, fontweight='bold')
        axes[idx + 1].axis('off')

    plt.tight_layout()
    plt.savefig('region_splitting_demo.png', dpi=150, bbox_inches='tight')
    print("Демонстрация разделения сохранена в region_splitting_demo.png")
    plt.close()


def analyze_region_quality(image_path: str, n_processes: int = 4, n_iterations: int = 50000):
    """
    Анализ качества воспроизведения для каждого региона отдельно
    """
    print("\n" + "="*60)
    print(f"АНАЛИЗ КАЧЕСТВА ПО РЕГИОНАМ ({n_processes} регионов)")
    print("="*60)

    algo = RegionParallelDPD(image_path, n_agents=100, n_processes=n_processes)

    print(f"\nВыполнение {n_iterations:,} итераций...")
    algo.parallel_step(n_iterations)

    # Вычисляем метрику для каждого региона
    print("\nМетрика качества по регионам:")
    print("-" * 60)

    grid_size = algo.grid_size
    region_metrics = np.zeros((grid_size, grid_size))

    for region in algo.regions:
        if region['dynamic'] is not None:
            # Вычисляем метрику для региона
            target = region['target']
            dynamic = region['dynamic']
            M = region['M']
            N_target = np.sum(target)

            if M > 0:
                normalized = (N_target / M) * dynamic
                diff = np.abs(target - normalized)
                metric = np.mean(diff / (target + 1e-10)) * 1000
            else:
                metric = float('inf')

            region_metrics[region['row'], region['col']] = metric

            print(f"Регион {region['id']:2d} (позиция [{region['row']},{region['col']}]): "
                  f"метрика = {metric:8.4f}, M = {M:,}")

    print("-" * 60)
    print(f"Средняя метрика: {np.mean(region_metrics):.4f}")
    print(f"Мин метрика: {np.min(region_metrics):.4f}")
    print(f"Макс метрика: {np.max(region_metrics):.4f}")
    print(f"Стд. отклонение: {np.std(region_metrics):.4f}")

    # Визуализация метрик по регионам
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Оригинал
    ax1.imshow(algo.target_distribution - 1.0, cmap='gray')
    ax1.set_title('Оригинальное изображение', fontsize=14)
    ax1.axis('off')

    # Текущий результат
    current_img = algo.get_current_image()
    ax2.imshow(current_img, cmap='gray')
    ax2.set_title(f'Результат ({n_iterations:,} итераций)', fontsize=14)
    ax2.axis('off')

    # Тепловая карта метрик
    im = ax3.imshow(region_metrics, cmap='RdYlGn_r', aspect='auto')
    ax3.set_title('Метрика качества по регионам\n(меньше = лучше)', fontsize=14)
    ax3.set_xlabel('Колонка региона')
    ax3.set_ylabel('Строка региона')

    # Добавляем значения на карту
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax3.text(j, i, f'{region_metrics[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax3, label='Метрика качества')

    plt.tight_layout()
    plt.savefig(f'region_quality_analysis_{n_processes}proc.png', dpi=150, bbox_inches='tight')
    print(f"\nАнализ качества сохранен в region_quality_analysis_{n_processes}proc.png")
    plt.close()


if __name__ == "__main__":
    image_path = "obama2.jpg"

    # Создаем тестовое изображение если нужно
    if not os.path.exists(image_path):
        print("Создаем тестовое изображение...")
        img = Image.new('L', (400, 400), color=255)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)

        # Рисуем несколько фигур
        draw.ellipse([100, 100, 300, 300], fill=50, outline=0)
        draw.rectangle([150, 150, 250, 250], fill=200)
        draw.ellipse([180, 50, 220, 90], fill=100)
        draw.rectangle([50, 180, 90, 220], fill=150)

        img.save(image_path)
        print(f"Тестовое изображение сохранено в {image_path}")

    # Выбор режима работы
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        # Полный набор тестов
        print("\n🚀 ЗАПУСК ПОЛНОГО НАБОРА ТЕСТОВ\n")

        # 1. Демонстрация разделения на регионы
        print("\n" + "=" * 60)
        print("1. ДЕМОНСТРАЦИЯ РАЗДЕЛЕНИЯ НА РЕГИОНЫ")
        print("=" * 60)
        demonstrate_region_splitting(image_path)

        # 2. Основной эксперимент с 4 регионами
        print("\n" + "=" * 60)
        print("2. ОСНОВНОЙ ЭКСПЕРИМЕНТ (4 региона)")
        print("=" * 60)
        run_region_experiment(image_path, n_processes=4, max_iterations=100000)

        # 3. Анализ качества по регионам
        print("\n" + "=" * 60)
        print("3. АНАЛИЗ КАЧЕСТВА ПО РЕГИОНАМ")
        print("=" * 60)
        analyze_region_quality(image_path, n_processes=4, n_iterations=50000)

        # 4. Сравнение производительности
        print("\n" + "=" * 60)
        print("4. СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 60)
        compare_region_parallelization(image_path)

    else:
        # Быстрый тест
        print("\n🚀 БЫСТРЫЙ ТЕСТ (для полного теста запустите: python script.py full)\n")

        # 1. Демонстрация разделения
        demonstrate_region_splitting(image_path)

        # 2. Один быстрый эксперимент
        print("\n" + "=" * 60)
        print("БЫСТРЫЙ ЭКСПЕРИМЕНТ (4 региона, 50k итераций)")
        print("=" * 60)
        run_region_experiment(image_path, n_processes=4, max_iterations=50000)

        # 3. Анализ качества
        print("\n" + "=" * 60)
        print("АНАЛИЗ КАЧЕСТВА ПО РЕГИОНАМ")
        print("=" * 60)
        analyze_region_quality(image_path, n_processes=4, n_iterations=25000)

        # 4. Мини-сравнение (только 1 и 4 региона)
        print("\n" + "=" * 60)
        print("МИНИ-СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 60)

        results = {}
        max_iter = 50000

        for n_proc in [1, 4]:
            print("\n" + "=" * 60)
            print(f"Тест с {n_proc} регионом(ами)")
            print("=" * 60)
            _, time_val = run_region_experiment(image_path, n_processes=n_proc, max_iterations=max_iter)
            results[f'{n_proc}_regions'] = time_val

        # Итоговое сравнение
        print("\n" + "=" * 60)
        print("ИТОГОВОЕ СРАВНЕНИЕ")
        print("=" * 60)
        print(f"{'Конфигурация':<30} {'Время (с)':<12} {'Ускорение':<10}")
        print("-" * 60)

        baseline_time = results.get('1_regions', 0)
        for name, time_val in results.items():
            speedup = baseline_time / time_val if time_val > 0 and baseline_time > 0 else 0
            n_regions = name.split('_')[0]
            grid_size = int(math.sqrt(int(n_regions)))
            display_name = f"{n_regions} регионов ({grid_size}x{grid_size})"
            print(f"{display_name:<30} {time_val:>10.2f}   {speedup:>8.2f}x")

        print("=" * 60)

    print("\n" + "=" * 60)
    print("ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 60)
    print("\nСозданные файлы:")
    print("  - region_splitting_demo.png - демонстрация разделения")
    print("  - regions_initial_4proc.png - начальное состояние")
    print("  - progress_regions_4proc.png - прогресс воспроизведения")
    print("  - regions_final_4proc.png - финальное состояние")
    print("  - region_quality_analysis_4proc.png - анализ качества")
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        print("  - performance_comparison.png - полное сравнение")
    print("=" * 60)