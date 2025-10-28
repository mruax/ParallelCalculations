import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, RawArray
import time
from typing import Tuple, List
from collections import deque
import ctypes
import atexit


class MazeNavigationAgent:
    """
    Алгоритм навигации агентов по лабиринту с использованием градиента альфа
    """

    def __init__(self, maze_path: str, n_agents: int = 100, seed: int = 42, verbose: bool = True):
        """
        Инициализация алгоритма

        Args:
            maze_path: путь к изображению лабиринта
            n_agents: количество агентов
            seed: seed для воспроизводимости
            verbose: выводить ли информацию о лабиринте
        """
        np.random.seed(seed)
        self.verbose = verbose

        # Загрузка лабиринта
        self.load_maze(maze_path)

        # Параметры
        self.n_agents = n_agents
        self.height, self.width = self.maze.shape

        # Вычисляем метрику альфа (градиент к выходу)
        self.compute_alpha_metric()

        # Инициализация агентов у входа
        self.initialize_agents()

        # 8 направлений: (dx, dy)
        self.directions = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1)
        ]

        # Подготовка случайных порядков проверки направлений
        self.prepare_random_orders(100000)
        self.current_order_idx = 0

        # История для визуализации
        self.history = []

        # Статистика
        self.successful_agents = 0
        self.total_steps = 0

    def load_maze(self, maze_path: str):
        """Загрузка лабиринта из изображения"""
        img = Image.open(maze_path).convert('L')

        # Преобразуем в numpy массив
        img_array = np.array(img, dtype=np.uint8)

        # Бинаризация: темные пиксели (< 128) = стены (0), светлые = проходы (1)
        self.maze = (img_array > 128).astype(np.int32)

        self.height, self.width = self.maze.shape

        if self.verbose:
            print(f"\nЗагружен лабиринт размером {self.maze.shape}")
            print(f"Клетки, по которым можно ходить: {np.sum(self.maze)}")
            print(f"Стены: {np.sum(1 - self.maze)}")

    def find_entry_exit(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Поиск входа (слева) и выхода (справа) в лабиринте

        Returns:
            ((entry_x, entry_y), (exit_x, exit_y))
        """
        # Ищем вход в левой колонке
        entry = None
        for y in range(self.height):
            if self.maze[y, 0] == 1:
                entry = (0, y)
                break

        # Ищем выход в правой колонке
        exit_point = None
        for y in range(self.height):
            if self.maze[y, self.width - 1] == 1:
                exit_point = (self.width - 1, y)
                break

        if entry is None:
            # Если нет прохода на границе, ищем ближайший к левому краю
            passable_left = np.where(self.maze[:, :self.width // 4] == 1)
            if len(passable_left[0]) > 0:
                idx = np.argmin(passable_left[1])
                entry = (passable_left[1][idx], passable_left[0][idx])

        if exit_point is None:
            # Если нет прохода на границе, ищем ближайший к правому краю
            passable_right = np.where(self.maze[:, 3 * self.width // 4:] == 1)
            if len(passable_right[0]) > 0:
                idx = np.argmax(passable_right[1])
                exit_point = (3 * self.width // 4 + passable_right[1][idx], passable_right[0][idx])

        if entry is None or exit_point is None:
            raise ValueError("Не удалось найти вход или выход в лабиринте")

        if self.verbose:
            print(f"Вход на координатах: {entry}")
            print(f"Выход на координатах: {exit_point}")

        return entry, exit_point

    def compute_alpha_metric(self):
        """
        Вычисление метрики альфа через волновой алгоритм (BFS) от выхода.
        Альфа = -расстояние_до_выхода, то есть чем ближе к выходу, тем больше альфа.
        """
        entry, exit_point = self.find_entry_exit()
        self.entry = entry
        self.exit = exit_point

        # Инициализация: все клетки с -inf, кроме выхода
        self.alpha = np.full((self.height, self.width), -np.inf, dtype=np.float64)

        # BFS от выхода
        queue = deque([exit_point])
        self.alpha[exit_point[1], exit_point[0]] = 0
        visited = set([exit_point])

        # 8 направлений для BFS
        directions_8 = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1)
        ]

        while queue:
            x, y = queue.popleft()
            current_dist = self.alpha[y, x]

            for dx, dy in directions_8:
                nx, ny = x + dx, y + dy

                # Проверка границ и проходимости
                if (0 <= nx < self.width and 0 <= ny < self.height and
                        self.maze[ny, nx] == 1 and (nx, ny) not in visited):
                    # Расстояние с учетом диагоналей
                    if dx != 0 and dy != 0:
                        dist = np.sqrt(2)  # диагональ
                    else:
                        dist = 1  # прямое движение

                    self.alpha[ny, nx] = current_dist - dist  # отрицательное расстояние
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        # Нормализация альфа к диапазону [0, 1] для наглядности
        valid_alpha = self.alpha[self.alpha > -np.inf]
        if len(valid_alpha) > 0:
            min_alpha = np.min(valid_alpha)
            max_alpha = np.max(valid_alpha)
            self.alpha_normalized = np.where(
                self.alpha > -np.inf,
                (self.alpha - min_alpha) / (max_alpha - min_alpha + 1e-10),
                0
            )
        else:
            self.alpha_normalized = np.zeros_like(self.alpha)

        if self.verbose:
            print(f"Диапазон альфа: [{np.min(valid_alpha):.2f}, {np.max(valid_alpha):.2f}]")

    def initialize_agents(self):
        """Инициализация агентов у входа в лабиринт"""
        entry_x, entry_y = self.entry

        # ВСЕ агенты стартуют строго в точке входа
        self.agents_x = np.full(self.n_agents, entry_x, dtype=np.int32)
        self.agents_y = np.full(self.n_agents, entry_y, dtype=np.int32)
        self.agents_finished = np.zeros(self.n_agents, dtype=bool)

        # Вычисляем альфа для стартовой позиции
        start_alpha = self.alpha[entry_y, entry_x]

    def prepare_random_orders(self, n_orders: int = 100000):
        """Подготовка случайных порядков проверки направлений"""
        self.random_orders = np.zeros((n_orders, 8), dtype=np.int32)
        for i in range(n_orders):
            self.random_orders[i] = np.random.permutation(8)

    def get_next_order(self) -> np.ndarray:
        """Получить следующий случайный порядок проверки"""
        order = self.random_orders[self.current_order_idx]
        self.current_order_idx = (self.current_order_idx + 1) % len(self.random_orders)
        return order

    def move_agent(self, agent_idx: int) -> Tuple[int, int]:
        """
        Перемещение одного агента

        Returns:
            новые координаты (x, y)
        """
        if self.agents_finished[agent_idx]:
            return self.agents_x[agent_idx], self.agents_y[agent_idx]

        x, y = self.agents_x[agent_idx], self.agents_y[agent_idx]

        # Проверяем, достиг ли агент выхода (в радиусе 2 клеток)
        exit_x, exit_y = self.exit
        if abs(x - exit_x) <= 2 and abs(y - exit_y) <= 2:
            self.agents_finished[agent_idx] = True
            return x, y

        current_alpha = self.alpha[y, x]
        best_alpha = current_alpha
        best_x, best_y = x, y

        # Случайный порядок проверки направлений
        check_order = self.get_next_order()

        for dir_idx in check_order:
            dx, dy = self.directions[dir_idx]
            new_x, new_y = x + dx, y + dy

            # Проверка границ и проходимости
            if (0 <= new_x < self.width and 0 <= new_y < self.height and
                    self.maze[new_y, new_x] == 1):
                new_alpha = self.alpha[new_y, new_x]

                # Движемся в сторону увеличения альфа (приближения к выходу)
                if new_alpha > best_alpha:
                    best_alpha = new_alpha
                    best_x, best_y = new_x, new_y

        return best_x, best_y

    def step(self):
        """Один шаг симуляции: перемещаем всех агентов"""
        for i in range(self.n_agents):
            if not self.agents_finished[i]:
                new_x, new_y = self.move_agent(i)
                self.agents_x[i] = new_x
                self.agents_y[i] = new_y

                # Проверяем достижение выхода
                exit_x, exit_y = self.exit
                if abs(new_x - exit_x) <= 2 and abs(new_y - exit_y) <= 2:
                    if not self.agents_finished[i]:
                        self.agents_finished[i] = True
                        self.successful_agents += 1

        self.total_steps += 1

    def get_statistics(self) -> dict:
        """Получить текущую статистику"""
        return {
            'total_steps': self.total_steps,
            'successful_agents': self.successful_agents,
            'remaining_agents': self.n_agents - self.successful_agents
        }

    def visualize_alpha_metric(self, save_path: str = 'alpha_metric.png'):
        """Визуализация метрики альфа на лабиринте"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Оригинальный лабиринт
        ax1.imshow(self.maze, cmap='gray')
        ax1.plot(self.entry[0], self.entry[1], 'go', markersize=15, label='Вход')
        ax1.plot(self.exit[0], self.exit[1], 'ro', markersize=15, label='Выход')
        ax1.set_title('Лабиринт', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.axis('off')

        # Метрика альфа (градиент к выходу)
        alpha_display = np.where(self.maze == 1, self.alpha_normalized, np.nan)
        im = ax2.imshow(alpha_display, cmap='hot', interpolation='nearest')
        ax2.plot(self.entry[0], self.entry[1], 'go', markersize=15, label='Вход')
        ax2.plot(self.exit[0], self.exit[1], 'ro', markersize=15, label='Выход')
        ax2.set_title('Метрика альфа (градиент к выходу)', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.axis('off')

        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='Альфа (нормализованная)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"Визуализация метрики альфа сохранена: {save_path}")


# Глобальные переменные для многопроцессорной обработки
shared_maze_base = None
shared_alpha_base = None
shared_random_orders_base = None
shared_agents_x_base = None
shared_agents_y_base = None
shared_agents_finished_base = None

maze_shape = None
alpha_shape = None
n_random_orders = None
directions_global = None
exit_point_global = None


def init_worker_maze(maze_base, alpha_base, orders_base, agents_x_base, agents_y_base,
                     agents_finished_base, maze_shp, alpha_shp, n_orders, dirs, exit_pt):
    """Инициализация worker процесса"""
    global shared_maze_base, shared_alpha_base, shared_random_orders_base
    global shared_agents_x_base, shared_agents_y_base, shared_agents_finished_base
    global maze_shape, alpha_shape, n_random_orders, directions_global, exit_point_global

    shared_maze_base = maze_base
    shared_alpha_base = alpha_base
    shared_random_orders_base = orders_base
    shared_agents_x_base = agents_x_base
    shared_agents_y_base = agents_y_base
    shared_agents_finished_base = agents_finished_base

    maze_shape = maze_shp
    alpha_shape = alpha_shp
    n_random_orders = n_orders
    directions_global = dirs
    exit_point_global = exit_pt


def process_agent_batch_maze(batch_info):
    """Обработка батча агентов в отдельном процессе"""
    agent_start, agent_end, order_start, n_steps = batch_info

    # Получаем доступ к shared memory
    maze_np = np.frombuffer(shared_maze_base, dtype=np.int32).reshape(maze_shape)
    alpha_np = np.frombuffer(shared_alpha_base, dtype=np.float64).reshape(alpha_shape)
    orders_np = np.frombuffer(shared_random_orders_base, dtype=np.int32).reshape((n_random_orders, 8))
    agents_x_np = np.frombuffer(shared_agents_x_base, dtype=np.int32)
    agents_y_np = np.frombuffer(shared_agents_y_base, dtype=np.int32)
    agents_finished_np = np.frombuffer(shared_agents_finished_base, dtype=ctypes.c_bool)

    height, width = maze_shape
    exit_x, exit_y = exit_point_global

    # Обработка батча
    new_positions = []
    finished_flags = []
    order_idx = order_start

    for agent_idx in range(agent_start, agent_end):
        if agents_finished_np[agent_idx]:
            new_positions.append((agents_x_np[agent_idx], agents_y_np[agent_idx]))
            finished_flags.append(True)
            continue

        x, y = agents_x_np[agent_idx], agents_y_np[agent_idx]

        # Проверка достижения выхода
        if abs(x - exit_x) <= 2 and abs(y - exit_y) <= 2:
            new_positions.append((x, y))
            finished_flags.append(True)
            continue

        current_alpha = alpha_np[y, x]
        best_alpha = current_alpha
        best_x, best_y = x, y

        check_order = orders_np[order_idx % n_random_orders]
        order_idx += 1

        for dir_idx in check_order:
            dx, dy = directions_global[dir_idx]
            new_x, new_y = x + dx, y + dy

            if (0 <= new_x < width and 0 <= new_y < height and
                    maze_np[new_y, new_x] == 1):
                new_alpha = alpha_np[new_y, new_x]

                if new_alpha > best_alpha:
                    best_alpha = new_alpha
                    best_x, best_y = new_x, new_y

        # Проверка на достижение выхода после движения
        is_finished = (abs(best_x - exit_x) <= 2 and abs(best_y - exit_y) <= 2)
        new_positions.append((best_x, best_y))
        finished_flags.append(is_finished)

    return new_positions, finished_flags


class ParallelMazeNavigator(MazeNavigationAgent):
    """Параллельная версия навигатора по лабиринту"""

    def __init__(self, maze_path: str, n_agents: int = 100, n_processes: int = 4, seed: int = 42, verbose: bool = True):
        self.n_processes = n_processes
        super().__init__(maze_path, n_agents, seed, verbose)

        # Создаем shared memory для параллельной обработки
        self.create_shared_memory()

        # Создаем пул процессов
        self.pool = Pool(
            processes=n_processes,
            initializer=init_worker_maze,
            initargs=(
                self.shared_maze,
                self.shared_alpha,
                self.shared_random_orders,
                self.shared_agents_x,
                self.shared_agents_y,
                self.shared_agents_finished,
                self.maze.shape,
                self.alpha.shape,
                len(self.random_orders),
                self.directions,
                self.exit
            )
        )

        # Регистрируем cleanup при выходе
        atexit.register(self.close_pool)

    def create_shared_memory(self):
        """Создание shared memory для всех необходимых данных"""
        # Лабиринт
        self.shared_maze = RawArray(ctypes.c_int32, self.maze.size)
        maze_np = np.frombuffer(self.shared_maze, dtype=np.int32).reshape(self.maze.shape)
        np.copyto(maze_np, self.maze)

        # Альфа
        self.shared_alpha = RawArray(ctypes.c_double, self.alpha.size)
        alpha_np = np.frombuffer(self.shared_alpha, dtype=np.float64).reshape(self.alpha.shape)
        np.copyto(alpha_np, self.alpha)

        # Случайные порядки
        self.shared_random_orders = RawArray(ctypes.c_int32, self.random_orders.size)
        orders_np = np.frombuffer(self.shared_random_orders, dtype=np.int32).reshape(self.random_orders.shape)
        np.copyto(orders_np, self.random_orders)

        # Агенты
        self.shared_agents_x = RawArray(ctypes.c_int32, self.n_agents)
        self.shared_agents_y = RawArray(ctypes.c_int32, self.n_agents)
        self.shared_agents_finished = RawArray(ctypes.c_bool, self.n_agents)

        agents_x_np = np.frombuffer(self.shared_agents_x, dtype=np.int32)
        agents_y_np = np.frombuffer(self.shared_agents_y, dtype=np.int32)
        agents_finished_np = np.frombuffer(self.shared_agents_finished, dtype=ctypes.c_bool)

        np.copyto(agents_x_np, self.agents_x)
        np.copyto(agents_y_np, self.agents_y)
        np.copyto(agents_finished_np, self.agents_finished)

    def close_pool(self):
        """Закрытие пула процессов"""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def step_parallel(self, n_steps: int = 1):
        """Параллельное выполнение шагов"""
        for _ in range(n_steps):
            # Создаем батчи для параллельной обработки
            batch_size = max(1, self.n_agents // self.n_processes)
            batches = []

            for i in range(self.n_processes):
                start = i * batch_size
                end = min((i + 1) * batch_size, self.n_agents) if i < self.n_processes - 1 else self.n_agents
                if start < end:
                    batches.append((start, end, self.current_order_idx, 1))
                    self.current_order_idx = (self.current_order_idx + (end - start)) % len(self.random_orders)

            # Параллельная обработка - используем существующий pool!
            results = self.pool.map(process_agent_batch_maze, batches)

            # Обновляем позиции - используем view вместо копирования
            agents_x_np = np.frombuffer(self.shared_agents_x, dtype=np.int32)
            agents_y_np = np.frombuffer(self.shared_agents_y, dtype=np.int32)
            agents_finished_np = np.frombuffer(self.shared_agents_finished, dtype=ctypes.c_bool)

            batch_idx = 0
            for start, end, _, _ in batches:
                positions, finished_flags = results[batch_idx]
                for i, (new_x, new_y) in enumerate(positions):
                    agent_idx = start + i
                    agents_x_np[agent_idx] = new_x
                    agents_y_np[agent_idx] = new_y

                    # Обновляем флаг завершения
                    if finished_flags[i] and not agents_finished_np[agent_idx]:
                        agents_finished_np[agent_idx] = True
                        self.successful_agents += 1

                batch_idx += 1

            self.total_steps += 1


def plot_maze_scalability(results: dict, save_path: str = 'experiment2_scalability.png'):
    """График масштабируемости"""
    n_processes = sorted(results.keys())
    times = [results[n] for n in n_processes]

    t1 = times[0]
    ideal_times = [t1 / n for n in n_processes]

    plt.figure(figsize=(8, 5))
    plt.plot(n_processes, ideal_times, label="Идеальная производительность",
             linestyle="--", color="black", linewidth=2)
    plt.plot(n_processes, times, label="Реальная производительность",
             marker="o", color="red", linewidth=2)
    plt.xlabel("Количество процессов", fontsize=16, fontweight='bold')
    plt.ylabel("Время работы (с)", fontsize=16, fontweight='bold')
    plt.title("Масштабируемость алгоритма", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nРезультаты масштабируемости:")
    print(f"{'Процессы':<12} {'Время (с)':<14} {'Ускорение':<14} {'Эффективность':<15}")

    speedups = [t1 / t for t in times]
    for n, t, s in zip(n_processes, times, speedups):
        efficiency = (s / n) * 100
        print(f"{n:<12} {t:>12.2f}   {s:>12.2f}x   {efficiency:>13.1f}%")


def experiment_2_scalability(maze_path: str, n_agents: int = 100, max_processes: int = 24):
    """
    Эксперимент: Масштабируемость от 1 до max_processes
    Задача - первый агент достигает выхода
    """
    print(f"\nЭксперимент: Масштабируемость", end="")

    results = {}
    iterations_per_run = {}

    for n_proc in range(1, max_processes + 1):
        print(f"\nТест с {n_proc} {'процессом' if n_proc == 1 else 'процессами'}:")

        # verbose=False чтобы не дублировать вывод лабиринта
        algo = ParallelMazeNavigator(
            maze_path,
            n_agents=n_agents,
            n_processes=n_proc,
            seed=42,
            verbose=False
        )

        iteration = 0
        start_time = time.time()
        last_print = 0

        while algo.successful_agents < 1:
            algo.step_parallel(1)
            iteration += 1

            # Печать прогресса каждые 1000 итераций
            if iteration - last_print >= 1000:
                agents_x = np.frombuffer(algo.shared_agents_x, dtype=np.int32)
                agents_y = np.frombuffer(algo.shared_agents_y, dtype=np.int32)
                agents_finished = np.frombuffer(algo.shared_agents_finished, dtype=ctypes.c_bool)

                active_mask = ~agents_finished
                if np.any(active_mask):
                    active_alphas = algo.alpha[agents_y[active_mask], agents_x[active_mask]]
                    max_alpha = np.max(active_alphas)
                    avg_alpha = np.mean(active_alphas)
                    print(f"  Итерация {iteration}: max_α={max_alpha:.2f}, avg_α={avg_alpha:.2f}")
                else:
                    print(f"  Итерация {iteration}: все агенты завершили")
                last_print = iteration

            # Защита от бесконечного цикла
            if iteration > 100000:
                print(f"Достигнут лимит итераций: {iteration}")
                break

        elapsed = time.time() - start_time

        results[n_proc] = elapsed
        iterations_per_run[n_proc] = iteration

        stats = algo.get_statistics()
        print(f"Время - {elapsed:.2f}с, Итераций - {iteration}, "
              f"Успешных агентов на момент остановки - {stats['successful_agents']}")

        # Закрываем пул процессов
        algo.close_pool()

    return results, iterations_per_run


if __name__ == "__main__":
    import sys
    import os

    maze_path = "maze.png"

    # Проверка существования файла
    if not os.path.exists(maze_path):
        print(f"Файл {maze_path} не найден!")
        sys.exit(1)

    print(f"Лабораторная работа №6. Прохождение лабиринта")

    # Загрузка лабиринта и визуализация метрики альфа
    algo_init = MazeNavigationAgent(maze_path, n_agents=100, seed=42, verbose=True)
    algo_init.visualize_alpha_metric('alpha_metric.png')

    # ЭКСПЕРИМЕНТ: Масштабируемость
    max_proc = min(6, cpu_count())

    scalability_results, iterations_results = experiment_2_scalability(
        maze_path, n_agents=100, max_processes=max_proc
    )

    # Построение графика масштабируемости
    plot_maze_scalability(scalability_results, save_path='experiment_scalability.png')

    # Итоговая сводка
    t1 = scalability_results[1]
    t_max = scalability_results[max_proc]
    speedup = t1 / t_max
    efficiency = (speedup / max_proc) * 100

    print(f"\nИтого:")
    print(f"  Время на 1 процессе: {t1:.2f}с")
    print(f"  Время на {max_proc} процессах: {t_max:.2f}с")
    print(f"  Ускорение: {speedup:.2f}x")
    print(f"  Эффективность: {efficiency:.1f}%")

    # Находим оптимальное количество процессов
    best_speedup = max(t1 / scalability_results[n] for n in scalability_results.keys())
    best_n_proc = [n for n in scalability_results.keys()
                   if abs((t1 / scalability_results[n]) - best_speedup) < 0.01][0]
    print(f"  Оптимальное число процессов: {best_n_proc} (ускорение {best_speedup:.2f}x)")

    print(f"\nВсе эксперименты завершены!")
