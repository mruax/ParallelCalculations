import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, Manager
import time
from typing import Tuple, List
import os


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
        # Темные области должны иметь МЕНЬШИЕ значения (больше агентов)
        # Светлые области должны иметь БОЛЬШИЕ значения (меньше агентов)
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
        # Клиппируем значения в допустимый диапазон
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


def worker_task(args):
    """
    Функция для параллельного выполнения шагов
    Работает с локальной копией данных
    """
    worker_id, agent_indices, agents_x, agents_y, dynamic_dist, M, target_dist, N_target, \
        directions, random_orders, start_order_idx, n_steps, height, width = args

    # Локальные копии
    local_agents_x = agents_x.copy()
    local_agents_y = agents_y.copy()
    local_dynamic = np.zeros_like(dynamic_dist)  # Начинаем с нуля для корректного суммирования
    local_M = 0
    order_idx = start_order_idx

    # Выполняем n_steps шагов для своих агентов
    for _ in range(n_steps):
        for agent_idx in agent_indices:
            x, y = local_agents_x[agent_idx], local_agents_y[agent_idx]

            # Получаем случайный порядок
            order = random_orders[order_idx % len(random_orders)]
            order_idx += 1

            best_k = float('-inf')
            best_x, best_y = x, y

            # Проверяем направления
            for dir_idx in order:
                dx, dy = directions[dir_idx]
                new_x, new_y = x + dx, y + dy

                if 0 <= new_x < width and 0 <= new_y < height:
                    # Вычисляем K на основе ГЛОБАЛЬНОГО состояния
                    global_M = M + local_M
                    if global_M == 0:
                        k = target_dist[new_y, new_x]
                    else:
                        # Используем глобальное + локальное распределение
                        total_dynamic = dynamic_dist[new_y, new_x] + local_dynamic[new_y, new_x]
                        normalized_dynamic = (N_target / global_M) * total_dynamic
                        k = target_dist[new_y, new_x] - normalized_dynamic

                    if k > best_k:
                        best_k = k
                        best_x, best_y = new_x, new_y

            # Обновляем локальное состояние
            local_agents_x[agent_idx] = best_x
            local_agents_y[agent_idx] = best_y
            local_dynamic[best_y, best_x] += 1
            local_M += 1

    return worker_id, local_agents_x, local_agents_y, local_dynamic, local_M


class ParallelDPD(DynamicsWithPreferredDistribution):
    """Параллельная версия алгоритма"""

    def __init__(self, image_path: str, n_agents: int = 100, n_processes: int = None, seed: int = 42):
        super().__init__(image_path, n_agents, seed)
        self.n_processes = n_processes or min(cpu_count(), 8)  # Ограничиваем максимум 8 процессами
        print(f"Используется {self.n_processes} процессов")

    def parallel_step(self, n_steps: int = 1):
        """Параллельное выполнение шагов"""
        # Разделяем агентов между процессами
        agents_per_process = self.n_agents // self.n_processes
        agent_groups = []

        for i in range(self.n_processes):
            start_idx = i * agents_per_process
            if i == self.n_processes - 1:
                end_idx = self.n_agents
            else:
                end_idx = (i + 1) * agents_per_process
            agent_groups.append(list(range(start_idx, end_idx)))

        # Подготавливаем аргументы для каждого процесса
        tasks = []
        for i, agent_indices in enumerate(agent_groups):
            start_order_idx = (self.current_order_idx + i * 100) % len(self.random_orders)
            tasks.append((
                i,  # worker_id для идентификации
                agent_indices,
                self.agents_x,
                self.agents_y,
                self.dynamic_distribution,
                self.M,
                self.target_distribution,
                self.N_target,
                self.directions,
                self.random_orders,
                start_order_idx,
                n_steps,
                self.height,
                self.width
            ))

        # Выполняем параллельно
        with Pool(processes=self.n_processes) as pool:
            results = pool.map(worker_task, tasks)

        # Объединяем результаты
        for worker_id, agents_x, agents_y, local_dynamic, local_M in results:
            agent_indices = agent_groups[worker_id]

            # Обновляем позиции агентов
            for agent_idx in agent_indices:
                self.agents_x[agent_idx] = agents_x[agent_idx]
                self.agents_y[agent_idx] = agents_y[agent_idx]

            # Суммируем динамические распределения
            self.dynamic_distribution += local_dynamic
            self.M += local_M

        self.current_order_idx = (self.current_order_idx + n_steps * self.n_agents) % len(self.random_orders)


def run_experiment(image_path: str, use_parallel: bool = True, n_processes: int = None, max_iterations: int = 10000000):
    """
    Запуск эксперимента

    Args:
        image_path: путь к изображению
        use_parallel: использовать ли параллельную версию
        n_processes: количество процессов (если None, используются все доступные)
        max_iterations: максимальное количество итераций
    """
    print("=" * 60)
    print("Алгоритм динамики с предпочтительным распределением")
    print("=" * 60)

    # Создаем экземпляр алгоритма
    if use_parallel:
        algo = ParallelDPD(image_path, n_agents=100, n_processes=n_processes)
        step_func = algo.parallel_step
    else:
        algo = DynamicsWithPreferredDistribution(image_path, n_agents=100)
        step_func = algo.step

    # Контрольные точки для визуализации
    checkpoints = [500, 1000, 2000, 5000, 10000, 25000]
    checkpoints = [cp for cp in checkpoints if cp <= max_iterations]

    print(f"\nРежим: {'Параллельный' if use_parallel else 'Последовательный'}")
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

    for checkpoint_idx, checkpoint in enumerate(checkpoints):
        steps_to_do = checkpoint - current_iteration

        # Делаем шаги батчами для лучшей производительности
        batch_size = min(1000, steps_to_do)
        n_batches = steps_to_do // batch_size

        for batch in range(n_batches):
            step_func(batch_size)
            current_iteration += batch_size

            # Показываем прогресс каждую секунду
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
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

        print()  # Новая строка после прогресса
        algo.save_checkpoint(current_iteration)

        # Промежуточное время
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
    suffix = f"_parallel_{algo.n_processes}proc" if use_parallel else "_sequential"
    algo.visualize_progress(f'progress{suffix}.png')

    return algo, total_time


def compare_performance(image_path: str):
    """Сравнение производительности последовательной и параллельной версий"""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)

    results = {}
    max_iter = 1000000  # Уменьшаем для более быстрого сравнения

    # Последовательная версия
    print("\n1. Последовательная версия:")
    _, time_seq = run_experiment(image_path, use_parallel=False, max_iterations=max_iter)
    results['sequential'] = time_seq

    # Параллельные версии с разным числом процессов
    available_cores = cpu_count()
    test_processes = [2, 4, 8]

    for n_proc in test_processes:
        if n_proc > available_cores:
            continue
        print(f"\n{len(results) + 1}. Параллельная версия ({n_proc} процесса):")
        _, time_par = run_experiment(image_path, use_parallel=True, n_processes=n_proc, max_iterations=max_iter)
        results[f'parallel_{n_proc}'] = time_par

    # Итоговое сравнение
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    print(f"{'Конфигурация':<20} {'Время (с)':<12} {'Ускорение':<10}")
    print("-" * 60)
    for name, time_val in results.items():
        speedup = results['sequential'] / time_val if time_val > 0 else 0
        print(f"{name:<20} {time_val:>10.2f}   {speedup:>8.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    # Путь к изображению (замените на свой)
    image_path = "obama2.jpg"

    # Если файл не существует, создаем тестовое изображение
    if not os.path.exists(image_path):
        print("Создаем тестовое изображение...")
        # Создаем простое тестовое изображение
        img = Image.new('L', (300, 300), color=255)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        # Рисуем круг
        draw.ellipse([75, 75, 225, 225], fill=50, outline=0)
        # Рисуем прямоугольник
        draw.rectangle([120, 120, 180, 180], fill=200)
        img.save(image_path)
        print(f"Тестовое изображение сохранено в {image_path}")

    # Запуск одного эксперимента
    print("\nЗапуск основного эксперимента...")
    run_experiment(image_path, use_parallel=True, max_iterations=10000000)

    # Раскомментируйте для полного сравнения производительности
    print("\n\nЗапуск сравнения производительности...")
    compare_performance(image_path)