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

    def __init__(self, maze_path: str, n_agents: int = 100, seed: int = 42):
        """
        Инициализация алгоритма

        Args:
            maze_path: путь к изображению лабиринта
            n_agents: количество агентов
            seed: seed для воспроизводимости
        """
        np.random.seed(seed)

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

        print(f"Лабиринт загружен: {self.maze.shape}")
        print(f"Проходимых клеток: {np.sum(self.maze)}")
        print(f"Стен: {np.sum(1 - self.maze)}")

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

        print(f"Вход найден: {entry}")
        print(f"Выход найден: {exit_point}")

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

        print(f"Метрика альфа вычислена")
        print(f"Диапазон альфа: [{np.min(valid_alpha):.2f}, {np.max(valid_alpha):.2f}]")
        print(f"Достижимых клеток: {len(valid_alpha)}")

    def initialize_agents(self):
        """Инициализация агентов у входа в лабиринт"""
        entry_x, entry_y = self.entry

        # ВСЕ агенты стартуют строго в точке входа
        self.agents_x = np.full(self.n_agents, entry_x, dtype=np.int32)
        self.agents_y = np.full(self.n_agents, entry_y, dtype=np.int32)
        self.agents_finished = np.zeros(self.n_agents, dtype=bool)

        # Вычисляем альфа для стартовой позиции
        start_alpha = self.alpha[entry_y, entry_x]

        print(f"Агенты инициализированы: {self.n_agents}")
        print(f"Стартовая позиция: ({entry_x}, {entry_y})")
        print(f"Альфа на старте: {start_alpha:.2f}")

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
            self.successful_agents += 1
            return x, y

        current_alpha = self.alpha[y, x]

        # Собираем все возможные направления с их альфами
        possible_moves = []

        for dir_idx in range(8):
            dx, dy = self.directions[dir_idx]
            new_x, new_y = x + dx, y + dy

            # Проверка границ и проходимости
            if (0 <= new_x < self.width and 0 <= new_y < self.height and
                    self.maze[new_y, new_x] == 1):
                alpha_val = self.alpha[new_y, new_x]
                possible_moves.append((new_x, new_y, alpha_val))

        if not possible_moves:
            # Застряли - остаемся на месте
            return x, y

        # Сортируем по альфе (по убыванию)
        possible_moves.sort(key=lambda m: m[2], reverse=True)

        # Выбираем из топ-K лучших направлений (K=3 или меньше если вариантов мало)
        top_k = min(3, len(possible_moves))
        candidates = possible_moves[:top_k]

        # Случайный выбор среди топ-K с уникальным seed
        np.random.seed(self.total_steps * 1000 + agent_idx)
        best_x, best_y, _ = candidates[np.random.randint(len(candidates))]

        return best_x, best_y

    def step(self, n_steps: int = 1):
        """Выполнить n_steps шагов алгоритма"""
        for _ in range(n_steps):
            # Перемещаем всех агентов последовательно
            for agent_idx in range(self.n_agents):
                if not self.agents_finished[agent_idx]:
                    new_x, new_y = self.move_agent(agent_idx)

                    # Обновляем позицию агента
                    self.agents_x[agent_idx] = new_x
                    self.agents_y[agent_idx] = new_y

            self.total_steps += 1

    def get_statistics(self) -> dict:
        """Получить статистику о прогрессе"""
        return {
            'total_steps': self.total_steps,
            'successful_agents': self.successful_agents,
            'active_agents': np.sum(~self.agents_finished),
            'success_rate': self.successful_agents / self.n_agents * 100
        }

    def save_checkpoint(self, iteration: int):
        """Сохранить контрольную точку"""
        stats = self.get_statistics()

        # Вычисляем метрики альфа для активных агентов
        active_mask = ~self.agents_finished
        if np.any(active_mask):
            active_x = self.agents_x[active_mask]
            active_y = self.agents_y[active_mask]
            active_alphas = self.alpha[active_y, active_x]
            max_alpha = np.max(active_alphas)
            avg_alpha = np.mean(active_alphas)
        else:
            max_alpha = 0
            avg_alpha = 0

        self.history.append({
            'iteration': iteration,
            'agents_x': self.agents_x.copy(),
            'agents_y': self.agents_y.copy(),
            'finished': self.agents_finished.copy(),
            'max_alpha': max_alpha,
            'avg_alpha': avg_alpha,
            **stats
        })

        print(f"Checkpoint - Итерация {iteration:,}: "
              f"max_α={max_alpha:.2f}, avg_α={avg_alpha:.2f}")

    def visualize_progress(self, save_path: str = 'maze_progress.png'):
        """Визуализация прогресса навигации"""
        if not self.history:
            print("История пуста, нечего визуализировать")
            return

        n_checkpoints = len(self.history)

        # Создаем фигуру с 7 subplot'ами: 6 для лабиринта + 1 для графика альфа
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Первые 6 ячеек для визуализации лабиринта
        axes_maze = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

        # Последняя ячейка (внизу справа) для графика альфа
        ax_alpha = fig.add_subplot(gs[2, :])

        # Выбираем ключевые контрольные точки
        indices = [0, n_checkpoints // 5, 2 * n_checkpoints // 5,
                   3 * n_checkpoints // 5, 4 * n_checkpoints // 5, n_checkpoints - 1]

        for idx, ax in zip(indices, axes_maze):
            if idx >= n_checkpoints:
                idx = n_checkpoints - 1

            checkpoint = self.history[idx]

            # Создаем изображение для визуализации
            viz_img = np.ones((self.height, self.width, 3), dtype=np.float32)

            # Рисуем лабиринт (стены черные)
            viz_img[self.maze == 0] = [0, 0, 0]

            # Рисуем градиент альфа
            for y in range(self.height):
                for x in range(self.width):
                    if self.maze[y, x] == 1:
                        alpha_val = self.alpha_normalized[y, x]
                        viz_img[y, x] = [1 - alpha_val * 0.3, 1 - alpha_val * 0.3, 1]

            # Рисуем вход и выход (оба зеленые)
            viz_img[self.entry[1], self.entry[0]] = [0, 1, 0]  # Зеленый вход
            viz_img[self.exit[1], self.exit[0]] = [0, 1, 0]  # Зеленый выход

            # Рисуем агентов с градиентом цвета
            agents_x = checkpoint['agents_x']
            agents_y = checkpoint['agents_y']
            finished = checkpoint['finished']

            # Подсчитываем активных агентов для градиента
            n_active = np.sum(~finished)

            for x, y, is_finished in zip(agents_x, agents_y, finished):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if is_finished:
                        viz_img[y, x] = [0.3, 0.3, 0.3]  # Темно-серые (завершили)
                    else:
                        # Градиент от темно-красного (мало агентов) до ярко-красного (много агентов)
                        intensity = 0.5 + 0.5 * (n_active / self.n_agents)
                        viz_img[y, x] = [intensity, 0, 0]

            ax.imshow(viz_img, interpolation='nearest')
            ax.set_title(f"Итерация {checkpoint['iteration']:,}\n"
                         f"Успешных: {checkpoint['successful_agents']}, "
                         f"Активных: {checkpoint['active_agents']}",
                         fontsize=10, fontweight='bold')
            ax.axis('off')

        # График прогресса альфа-метрики
        iterations = [h['iteration'] for h in self.history]
        max_alphas = [h.get('max_alpha', 0) for h in self.history]
        avg_alphas = [h.get('avg_alpha', 0) for h in self.history]

        ax_alpha.plot(iterations, max_alphas, 'b-', linewidth=3, marker='o',
                      markersize=6, label='Максимальная α', alpha=0.8)
        ax_alpha.plot(iterations, avg_alphas, 'g-', linewidth=3, marker='s',
                      markersize=6, label='Средняя α', alpha=0.8)
        ax_alpha.fill_between(iterations, avg_alphas, max_alphas, alpha=0.2, color='cyan')

        ax_alpha.set_xlabel('Итерация', fontsize=14, fontweight='bold')
        ax_alpha.set_ylabel('Метрика α', fontsize=14, fontweight='bold')
        ax_alpha.set_title('Прогресс метрики α активных агентов', fontsize=14, fontweight='bold')
        ax_alpha.grid(True, alpha=0.3, linestyle='--')
        ax_alpha.legend(fontsize=12, loc='lower right')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация сохранена: {save_path}")
        plt.close()

    def visualize_alpha_metric(self, save_path: str = 'alpha_metric.png'):
        """Визуализация метрики альфа"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Левый график - метрика альфа
        ax = axes[0]
        alpha_display = np.where(self.maze == 1, self.alpha_normalized, np.nan)
        im = ax.imshow(alpha_display, cmap='RdYlGn', interpolation='nearest')
        ax.plot(self.entry[0], self.entry[1], 'bo', markersize=15, label='Вход', markeredgecolor='white',
                markeredgewidth=2)
        ax.plot(self.exit[0], self.exit[1], 'g*', markersize=20, label='Выход', markeredgecolor='white',
                markeredgewidth=2)
        ax.set_title('Метрика альфа (градиент к выходу)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Альфа (нормализованная)')

        # Правый график - лабиринт
        ax = axes[1]
        ax.imshow(self.maze, cmap='gray', interpolation='nearest')
        ax.plot(self.entry[0], self.entry[1], 'bo', markersize=15, label='Вход', markeredgecolor='white',
                markeredgewidth=2)
        ax.plot(self.exit[0], self.exit[1], 'g*', markersize=20, label='Выход', markeredgecolor='white',
                markeredgewidth=2)
        ax.set_title('Лабиринт', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация метрики альфа сохранена: {save_path}")
        plt.close()


# === ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ ===

# Глобальные переменные для shared memory
_shared_maze = None
_shared_alpha = None
_shared_agents_x = None
_shared_agents_y = None
_shared_agents_finished = None
_maze_shape = None
_n_agents = None
_directions = None
_random_orders = None
_exit_point = None


def init_worker_maze(maze_array, alpha_array, agents_x_array, agents_y_array,
                     agents_finished_array, maze_shape, n_agents, directions,
                     random_orders, exit_point):
    """Инициализация worker процесса"""
    global _shared_maze, _shared_alpha, _shared_agents_x, _shared_agents_y
    global _shared_agents_finished, _maze_shape, _n_agents, _directions
    global _random_orders, _exit_point

    _shared_maze = maze_array
    _shared_alpha = alpha_array
    _shared_agents_x = agents_x_array
    _shared_agents_y = agents_y_array
    _shared_agents_finished = agents_finished_array
    _maze_shape = maze_shape
    _n_agents = n_agents
    _directions = directions
    _random_orders = random_orders
    _exit_point = exit_point


def process_agent_batch_maze(args):
    """Обработка батча агентов"""
    start_idx, end_idx, step_num, exit_point = args

    # Получаем numpy массивы из shared memory
    maze = np.frombuffer(_shared_maze, dtype=np.int32).reshape(_maze_shape)
    alpha = np.frombuffer(_shared_alpha, dtype=np.float64).reshape(_maze_shape)
    agents_x = np.frombuffer(_shared_agents_x, dtype=np.int32)
    agents_y = np.frombuffer(_shared_agents_y, dtype=np.int32)
    agents_finished = np.frombuffer(_shared_agents_finished, dtype=ctypes.c_bool)

    height, width = _maze_shape
    new_positions = []
    newly_finished = []

    for i in range(start_idx, end_idx):
        if agents_finished[i]:
            new_positions.append((agents_x[i], agents_y[i]))
            newly_finished.append(False)
            continue

        x, y = agents_x[i], agents_y[i]

        # Проверка достижения выхода
        exit_x, exit_y = exit_point
        if abs(x - exit_x) <= 2 and abs(y - exit_y) <= 2:
            new_positions.append((x, y))
            newly_finished.append(True)
            continue

        # Собираем все возможные направления с их альфами
        possible_moves = []

        for dir_idx in range(8):
            dx, dy = _directions[dir_idx]
            new_x, new_y = x + dx, y + dy

            if (0 <= new_x < width and 0 <= new_y < height and
                    maze[new_y, new_x] == 1):
                alpha_val = alpha[new_y, new_x]
                possible_moves.append((new_x, new_y, alpha_val))

        if not possible_moves:
            # Застряли - остаемся на месте
            new_positions.append((x, y))
            newly_finished.append(False)
            continue

        # Сортируем по альфе (по убыванию)
        possible_moves.sort(key=lambda m: m[2], reverse=True)

        # Выбираем из топ-K лучших направлений
        top_k = min(3, len(possible_moves))
        candidates = possible_moves[:top_k]

        # Случайный выбор с уникальным seed
        np.random.seed(step_num * 1000 + i)
        best_x, best_y, _ = candidates[np.random.randint(len(candidates))]

        new_positions.append((best_x, best_y))
        newly_finished.append(False)

    return new_positions, newly_finished


class ParallelMazeNavigator(MazeNavigationAgent):
    """Параллельная версия навигатора по лабиринту"""

    def __init__(self, maze_path: str, n_agents: int = 100,
                 n_processes: int = None, seed: int = 42):
        super().__init__(maze_path, n_agents, seed)

        if n_processes is None:
            n_processes = cpu_count()
        self.n_processes = n_processes

        # Создаем shared memory
        self.setup_shared_memory()

        # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: создаем Pool один раз!
        self.pool = None
        if self.n_processes >= 1:
            self.pool = Pool(
                processes=self.n_processes,
                initializer=init_worker_maze,
                initargs=(
                    self.shared_maze, self.shared_alpha,
                    self.shared_agents_x, self.shared_agents_y,
                    self.shared_agents_finished,
                    self.maze.shape, self.n_agents,
                    self.directions, self.random_orders,
                    self.exit
                )
            )
            # Регистрируем закрытие пула при завершении программы
            atexit.register(self.close_pool)

    def close_pool(self):
        """Закрытие процессного пула"""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        """Деструктор - закрываем пул"""
        self.close_pool()

    def setup_shared_memory(self):
        """Настройка shared memory для параллельной обработки"""
        # Shared memory для лабиринта и альфа (read-only для workers)
        self.shared_maze = RawArray(ctypes.c_int32, self.maze.flatten())
        self.shared_alpha = RawArray(ctypes.c_double, self.alpha.flatten())

        # Shared memory для агентов (read-write)
        self.shared_agents_x = RawArray(ctypes.c_int32, self.agents_x)
        self.shared_agents_y = RawArray(ctypes.c_int32, self.agents_y)
        self.shared_agents_finished = RawArray(ctypes.c_bool, self.agents_finished)

    def step_parallel(self, n_steps: int = 1):
        """Параллельное выполнение шагов"""
        # if self.n_processes <= 1 or self.pool is None:
        #     # Если 1 процесс - используем обычный метод
        #     self.step(n_steps)
        #     return

        for step in range(n_steps):
            # Разбиваем агентов на батчи
            batch_size = max(1, self.n_agents // self.n_processes)
            batches = []

            for i in range(self.n_processes):
                start = i * batch_size
                end = min((i + 1) * batch_size, self.n_agents)
                if start < end:
                    batches.append((start, end, self.total_steps, self.exit))

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

            # Обновляем только если нужно сохранить checkpoint
            # Убираем постоянное копирование для производительности


def plot_maze_scalability(results: dict, save_path: str = 'experiment2_scalability.png'):
    """График масштабируемости"""
    n_processes = sorted(results.keys())
    times = [results[n] for n in n_processes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # График 1: Время выполнения
    ax1.plot(n_processes, times, 'b-', linewidth=3, marker='o', markersize=10, label='Время выполнения')

    # Идеальная масштабируемость
    t1 = times[0]
    ideal_times = [t1 / n for n in n_processes]
    ax1.plot(n_processes, ideal_times, 'r--', linewidth=2, alpha=0.5, label='Идеальная масштабируемость')

    ax1.set_xlabel('Количество процессов', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Время выполнения (секунды)', fontsize=14, fontweight='bold')
    ax1.set_title('Время выполнения vs Количество процессов', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    # График 2: Ускорение и эффективность
    speedups = [t1 / t for t in times]
    efficiencies = [(s / n) * 100 for s, n in zip(speedups, n_processes)]

    ax2_twin = ax2.twinx()

    line1 = ax2.plot(n_processes, speedups, 'g-', linewidth=3, marker='o', markersize=10, label='Ускорение')
    line2 = ax2.plot(n_processes, n_processes, 'r--', linewidth=2, alpha=0.5, label='Линейное ускорение')
    line3 = ax2_twin.plot(n_processes, efficiencies, 'purple', linewidth=3, marker='s', markersize=8,
                          label='Эффективность')

    ax2.set_xlabel('Количество процессов', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Ускорение', fontsize=14, fontweight='bold', color='g')
    ax2_twin.set_ylabel('Эффективность (%)', fontsize=14, fontweight='bold', color='purple')
    ax2.set_title('Ускорение и Эффективность', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='purple')

    # Объединяем легенды
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nГрафик масштабируемости сохранен: {save_path}")
    plt.close()

    # Выводим таблицу
    speedups = [t1 / t for t in times]
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ МАСШТАБИРУЕМОСТИ")
    print("=" * 70)
    print(f"{'Процессы':<12} {'Время (с)':<12} {'Ускорение':<12} {'Эффективность':<15}")
    print("-" * 70)
    for n, t, s in zip(n_processes, times, speedups):
        efficiency = (s / n) * 100
        print(f"{n:<12} {t:>10.2f}   {s:>10.2f}x   {efficiency:>12.1f}%")
    print("=" * 70)


def experiment_2_scalability(maze_path: str, n_agents: int = 100, max_processes: int = 24):
    """
    Эксперимент 2: Масштабируемость от 1 до 24 процессов
    Задача - первый агент достигает выхода
    """
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 2: Масштабируемость (1-24 процесса)")
    print("Критерий остановки: первый агент достигает выхода")
    print("=" * 70)

    results = {}
    iterations_per_run = {}

    for n_proc in range(1, max_processes + 1):
        print(f"\n{'─' * 60}")
        print(f"Тест с {n_proc} процесс{'ом' if n_proc == 1 else 'ами'}")
        print(f"{'─' * 60}")

        algo = ParallelMazeNavigator(maze_path, n_agents=n_agents,
                                     n_processes=n_proc, seed=42)

        iteration = 0
        start_time = time.time()
        last_print = 0

        while algo.successful_agents < 1:
            algo.step_parallel(1)
            iteration += 1

            # Печатаем прогресс каждые 100 итераций
            if iteration - last_print >= 100:
                # Синхронизируем массивы для отображения статистики
                algo.agents_x = np.array(np.frombuffer(algo.shared_agents_x, dtype=np.int32))
                algo.agents_y = np.array(np.frombuffer(algo.shared_agents_y, dtype=np.int32))
                algo.agents_finished = np.array(np.frombuffer(algo.shared_agents_finished, dtype=ctypes.c_bool))

                active_mask = ~algo.agents_finished
                if np.any(active_mask):
                    active_alphas = algo.alpha[algo.agents_y[active_mask], algo.agents_x[active_mask]]
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
        print(f"✓ Время до первого успеха: {elapsed:.2f}с")
        print(f"✓ Итераций: {iteration}")
        print(f"✓ Успешных агентов на момент остановки: {stats['successful_agents']}")
        print(f"✓ Скорость: {iteration / elapsed:.1f} итераций/с")

        # Закрываем пул
        algo.close_pool()

    return results, iterations_per_run


if __name__ == "__main__":
    import sys
    import os

    maze_path = "maze2223.png"

    # Проверка существования файла
    if not os.path.exists(maze_path):
        print(f"Файл {maze_path} не найден!")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("НАВИГАЦИЯ АГЕНТОВ ПО ЛАБИРИНТУ (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
    print("=" * 70)

    # Загрузка лабиринта и визуализация метрики альфа
    print("\nЗагрузка лабиринта и вычисление метрики альфа...")
    algo_init = MazeNavigationAgent(maze_path, n_agents=100, seed=42)
    algo_init.visualize_alpha_metric('outputs/alpha_metric.png')

    # ЭКСПЕРИМЕНТ 1: Масштабируемость
    # max_proc = min(24, cpu_count())
    # print(f"\nМаксимальное количество процессов для тестирования: {max_proc}")
    #
    # scalability_results, iterations_results = experiment_2_scalability(
    #     maze_path, n_agents=100, max_processes=max_proc
    # )

    # Построение графика масштабируемости
    # plot_maze_scalability(scalability_results, save_path='outputs/experiment_scalability.png')

    # # Итоговая сводка
    # print("\n" + "=" * 70)
    # print("ИТОГОВАЯ СВОДКА")
    # print("=" * 70)
    #
    # t1 = scalability_results[1]
    # t_max = scalability_results[max_proc]
    # speedup = t1 / t_max
    # efficiency = (speedup / max_proc) * 100
    #
    # print(f"\nЭксперимент 1. Масштабируемость:")
    # print(f"  Время на 1 процессе: {t1:.2f}с")
    # print(f"  Время на {max_proc} процессах: {t_max:.2f}с")
    # print(f"  Ускорение: {speedup:.2f}x")
    # print(f"  Эффективность: {efficiency:.1f}%")
    #
    # # Находим оптимальное количество процессов
    # best_speedup = max(t1 / scalability_results[n] for n in scalability_results.keys())
    # best_n_proc = [n for n in scalability_results.keys()
    #                if abs((t1 / scalability_results[n]) - best_speedup) < 0.01][0]
    # print(f"  Оптимальное число процессов: {best_n_proc} (ускорение {best_speedup:.2f}x)")
    #
    # print("\n" + "=" * 70)
    # print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    # print("=" * 70)
    # print("\nСохраненные файлы:")
    # print("  - alpha_metric.png - визуализация метрики альфа")
    # print("  - experiment2_scalability.png - график масштабируемости")
    # print("=" * 70)
