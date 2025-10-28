import numpy as np
import matplotlib.pyplot as plt
import time
import random
from PIL import Image
from multiprocessing import Pool, cpu_count, RawArray
from typing import Tuple, List
import ctypes
import atexit


class MazeSolver:
    def __init__(self, maze_path: str):
        """Инициализация решателя лабиринта"""
        self.load_maze(maze_path)
        self.find_start_exit_positions()

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

    def find_start_exit_positions(self):
        """Автоматическое нахождение старта и выхода"""
        start_candidates = []
        exit_candidates = []

        # Ищем проходы на границах
        # Левая граница
        for i in range(self.height):
            if self.maze[i, 0] == 1:
                start_candidates.append((0, i))

        # Правая граница
        for i in range(self.height):
            if self.maze[i, self.width - 1] == 1:
                exit_candidates.append((self.width - 1, i))

        # Верхняя граница
        for j in range(self.width):
            if self.maze[0, j] == 1:
                start_candidates.append((j, 0))

        # Нижняя граница
        for j in range(self.height):
            if self.maze[self.height - 1, j] == 1:
                exit_candidates.append((j, self.height - 1))

        # Выбираем позиции
        self.start_pos = start_candidates[0] if start_candidates else (0, 1)
        self.exit_pos = exit_candidates[-1] if exit_candidates else (self.width - 1, self.height - 2)

        print(f"Стартовая позиция: {self.start_pos}")
        print(f"Выход: {self.exit_pos}")
        print(f"Найдено стартовых позиций: {len(start_candidates)}")
        print(f"Найдено выходов: {len(exit_candidates)}")


# Глобальные переменные для shared memory
_shared_maze = None
_shared_dynamic_dist = None
_shared_random_sequences = None
_maze_shape = None
_num_sequences = None


def init_worker(maze_data, maze_shape, dynamic_dist_data, random_seq_data, num_sequences):
    """Инициализация глобальных переменных для worker процессов"""
    global _shared_maze, _shared_dynamic_dist, _shared_random_sequences
    global _maze_shape, _num_sequences
    
    _maze_shape = maze_shape
    _num_sequences = num_sequences
    
    # Создаем numpy массивы из shared memory
    _shared_maze = np.frombuffer(maze_data, dtype=np.int32).reshape(maze_shape)
    _shared_dynamic_dist = np.frombuffer(dynamic_dist_data, dtype=np.float64).reshape(maze_shape)
    _shared_random_sequences = np.frombuffer(random_seq_data, dtype=np.int32).reshape((num_sequences, 4, 2))


def calculate_K(x, y):
    """Упрощенный расчет показателя K"""
    target_value = x
    visits = _shared_dynamic_dist[y, x]
    return target_value / (visits + 1)


def get_valid_moves(x, y):
    """Получение допустимых ходов"""
    valid_moves = []
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if (0 <= new_x < _maze_shape[1] and 0 <= new_y < _maze_shape[0] and
                _shared_maze[new_y, new_x] == 1):
            K_val = calculate_K(new_x, new_y)
            if K_val > -5:
                valid_moves.append((new_x, new_y))
    
    return valid_moves


def move_agent_parallel(args):
    """
    Выполнение одного шага для агента (параллельная версия)
    
    Args:
        args: (agent_id, current_x, current_y, exit_x, exit_y, sequence_idx)
    
    Returns:
        (agent_id, new_x, new_y, success, new_sequence_idx)
    """
    agent_id, x, y, exit_x, exit_y, sequence_idx = args
    
    # Проверка достижения выхода
    if (x, y) == (exit_x, exit_y):
        return (agent_id, x, y, True, sequence_idx)
    
    # Получение случайной последовательности
    seq_idx = sequence_idx % _num_sequences
    random_seq = _shared_random_sequences[seq_idx]
    
    # Получение допустимых ходов
    valid_moves = get_valid_moves(x, y)
    
    if not valid_moves:
        # Тупик - возвращаемся на шаг назад (упрощенная версия)
        return (agent_id, x, y, False, sequence_idx + 1)
    
    best_K = -np.inf
    best_pos = (x, y)
    best_candidates = []
    
    # Оценка ходов
    for direction in random_seq:
        new_x, new_y = x + direction[0], y + direction[1]
        
        if (new_x, new_y) not in valid_moves:
            continue
        
        K_val = calculate_K(new_x, new_y)
        
        if K_val > best_K:
            best_K = K_val
            best_pos = (new_x, new_y)
            best_candidates = [(new_x, new_y)]
        elif K_val == best_K:
            best_candidates.append((new_x, new_y))
    
    # Случайный выбор из лучших
    if len(best_candidates) > 1:
        best_pos = random.choice(best_candidates)
    
    return (agent_id, best_pos[0], best_pos[1], False, sequence_idx + 1)


class ParallelMazeAgent:
    def __init__(self, maze, start_pos, exit_pos, n_agents=10, n_processes=None):
        """
        Параллельный агент для обучения в лабиринте
        
        Args:
            maze: матрица лабиринта
            start_pos: начальная позиция
            exit_pos: позиция выхода
            n_agents: количество агентов
            n_processes: количество процессов (по умолчанию - количество CPU)
        """
        self.maze = maze
        self.height, self.width = maze.shape
        self.start_pos = start_pos
        self.exit_pos = exit_pos
        self.n_agents = n_agents
        self.n_processes = n_processes if n_processes else cpu_count()
        
        # Инициализация агентов
        self.agents_x = np.full(n_agents, start_pos[0], dtype=np.int32)
        self.agents_y = np.full(n_agents, start_pos[1], dtype=np.int32)
        self.agents_finished = np.zeros(n_agents, dtype=bool)
        self.agents_sequence_idx = np.zeros(n_agents, dtype=np.int32)
        
        # Динамическое распределение (shared)
        self.dynamic_distribution = np.zeros((self.height, self.width), dtype=np.float64)
        
        # Генерация случайных последовательностей
        self._generate_random_sequences()
        
        # Создание shared memory
        self._create_shared_memory()
        
        # Создание пула процессов
        self.pool = Pool(
            processes=self.n_processes,
            initializer=init_worker,
            initargs=(
                self.shared_maze_data,
                (self.height, self.width),
                self.shared_dynamic_dist_data,
                self.shared_random_seq_data,
                self.num_sequences
            )
        )
        
        # Статистика
        self.successful_agents = 0
        self.total_steps = 0
        self.iteration = 0
        
        print(f"\nПараллельный агент инициализирован:")
        print(f"  Количество агентов: {n_agents}")
        print(f"  Количество процессов: {self.n_processes}")
        print(f"  Стартовая позиция: {start_pos}")
        print(f"  Выход: {exit_pos}")

    def _generate_random_sequences(self, num_sequences=10001):
        """Генерация случайных последовательностей направлений"""
        self.num_sequences = num_sequences
        self.random_sequences = np.zeros((num_sequences, 4, 2), dtype=np.int32)
        
        directions = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)], dtype=np.int32)
        
        for i in range(num_sequences):
            self.random_sequences[i] = directions[np.random.permutation(4)]

    def _create_shared_memory(self):
        """Создание shared memory для данных"""
        # Лабиринт (read-only)
        self.shared_maze = RawArray(ctypes.c_int32, int(self.height * self.width))
        self.shared_maze_data = np.frombuffer(self.shared_maze, dtype=np.int32).reshape(
            (self.height, self.width)
        )
        self.shared_maze_data[:] = self.maze
        
        # Динамическое распределение (read-write)
        self.shared_dynamic_dist = RawArray(ctypes.c_double, int(self.height * self.width))
        self.shared_dynamic_dist_data = np.frombuffer(
            self.shared_dynamic_dist, dtype=np.float64
        ).reshape((self.height, self.width))
        self.shared_dynamic_dist_data[:] = 0
        
        # Случайные последовательности (read-only)
        self.shared_random_seq = RawArray(ctypes.c_int32, int(self.num_sequences * 4 * 2))
        self.shared_random_seq_data = np.frombuffer(
            self.shared_random_seq, dtype=np.int32
        ).reshape((self.num_sequences, 4, 2))
        self.shared_random_seq_data[:] = self.random_sequences

    def step_parallel(self, n_steps=1):
        """Выполнение n шагов параллельно для всех агентов"""
        for _ in range(n_steps):
            # Подготовка задач для активных агентов
            tasks = []
            active_agents = []
            
            for agent_id in range(self.n_agents):
                if not self.agents_finished[agent_id]:
                    tasks.append((
                        agent_id,
                        self.agents_x[agent_id],
                        self.agents_y[agent_id],
                        self.exit_pos[0],
                        self.exit_pos[1],
                        self.agents_sequence_idx[agent_id]
                    ))
                    active_agents.append(agent_id)
            
            if not tasks:
                break
            
            # Параллельное выполнение
            results = self.pool.map(move_agent_parallel, tasks)
            
            # Обновление состояния агентов
            for agent_id, new_x, new_y, success, new_seq_idx in results:
                self.agents_x[agent_id] = new_x
                self.agents_y[agent_id] = new_y
                self.agents_sequence_idx[agent_id] = new_seq_idx
                
                # Обновление динамического распределения
                self.shared_dynamic_dist_data[new_y, new_x] += 1
                
                if success and not self.agents_finished[agent_id]:
                    self.agents_finished[agent_id] = True
                    self.successful_agents += 1
            
            self.total_steps += 1
            self.iteration += 1

    def train(self, max_iterations=100000):
        """Обучение агентов"""
        start_time = time.time()
        
        print("\nНачало обучения...")
        last_print = 0
        
        while self.successful_agents < 1 and self.iteration < max_iterations:
            self.step_parallel(1)
            
            # Прогресс каждые 100 итераций
            if self.iteration - last_print >= 100:
                active_count = np.sum(~self.agents_finished)
                print(f"  Итерация {self.iteration}: "
                      f"успешных={self.successful_agents}, "
                      f"активных={active_count}")
                last_print = self.iteration
        
        training_time = time.time() - start_time
        
        return {
            'success': self.successful_agents > 0,
            'iterations': self.iteration,
            'time': training_time,
            'successful_agents': self.successful_agents
        }

    def get_statistics(self):
        """Получение статистики"""
        return {
            'total_agents': self.n_agents,
            'successful_agents': self.successful_agents,
            'total_steps': self.total_steps,
            'iterations': self.iteration
        }

    def close_pool(self):
        """Закрытие пула процессов"""
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()

    def __del__(self):
        """Деструктор"""
        self.close_pool()


def experiment_scalability(maze_path: str, n_agents: int = 100, max_processes: int = 24):
    """
    Эксперимент: тестирование масштабируемости от 1 до max_processes процессов
    """
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ: Масштабируемость параллелизации")
    print("Критерий остановки: первый агент достигает выхода")
    print("=" * 70)
    
    # Загрузка лабиринта
    solver = MazeSolver(maze_path)
    
    results = {}
    
    for n_proc in range(1, max_processes + 1):
        print(f"\n{'─' * 60}")
        print(f"Тест с {n_proc} процесс{'ом' if n_proc == 1 else 'ами'}")
        print(f"{'─' * 60}")
        
        agent = ParallelMazeAgent(
            solver.maze,
            solver.start_pos,
            solver.exit_pos,
            n_agents=n_agents,
            n_processes=n_proc
        )
        
        start_time = time.time()
        train_results = agent.train(max_iterations=100000)
        elapsed = time.time() - start_time
        
        results[n_proc] = elapsed
        
        print(f"✓ Время: {elapsed:.2f}с")
        print(f"✓ Итераций: {train_results['iterations']}")
        print(f"✓ Успешных агентов: {train_results['successful_agents']}")
        print(f"✓ Скорость: {train_results['iterations'] / elapsed:.1f} итераций/с")
        
        agent.close_pool()
    
    return results


def plot_scalability_results(results: dict, save_path: str = 'scalability_results.png'):
    """Построение графиков масштабируемости"""
    n_processes = sorted(results.keys())
    times = [results[n] for n in n_processes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # График 1: Время выполнения
    ax1.plot(n_processes, times, 'b-', linewidth=3, marker='o', markersize=10, 
             label='Время выполнения')
    
    # Идеальная масштабируемость
    t1 = times[0]
    ideal_times = [t1 / n for n in n_processes]
    ax1.plot(n_processes, ideal_times, 'r--', linewidth=2, alpha=0.5, 
             label='Идеальная масштабируемость')
    
    ax1.set_xlabel('Количество процессов', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Время выполнения (секунды)', fontsize=14, fontweight='bold')
    ax1.set_title('Время выполнения vs Количество процессов', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # График 2: Ускорение и эффективность
    speedups = [t1 / t for t in times]
    efficiencies = [(s / n) * 100 for s, n in zip(speedups, n_processes)]
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(n_processes, speedups, 'g-', linewidth=3, marker='o', 
                     markersize=10, label='Ускорение')
    line2 = ax2.plot(n_processes, n_processes, 'r--', linewidth=2, alpha=0.5, 
                     label='Линейное ускорение')
    line3 = ax2_twin.plot(n_processes, efficiencies, 'purple', linewidth=3, 
                         marker='s', markersize=8, label='Эффективность')
    
    ax2.set_xlabel('Количество процессов', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Ускорение', fontsize=14, fontweight='bold', color='g')
    ax2_twin.set_ylabel('Эффективность (%)', fontsize=14, fontweight='bold', color='purple')
    ax2.set_title('Ускорение и Эффективность', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='purple')
    
    # Легенда
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nГрафик сохранен: {save_path}")
    plt.close()
    
    # Таблица результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ МАСШТАБИРУЕМОСТИ")
    print("=" * 70)
    print(f"{'Процессы':<12} {'Время (с)':<12} {'Ускорение':<12} {'Эффективность':<15}")
    print("-" * 70)
    for n, t, s in zip(n_processes, times, speedups):
        efficiency = (s / n) * 100
        print(f"{n:<12} {t:>10.2f}   {s:>10.2f}x   {efficiency:>12.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    import os
    
    maze_path = "maze.png"
    
    # Проверка существования файла
    if not os.path.exists(maze_path):
        print(f"Файл {maze_path} не найден!")
        print("Укажите путь к файлу лабиринта в переменной maze_path")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("ПАРАЛЛЕЛЬНОЕ РЕШЕНИЕ ЛАБИРИНТА")
    print("Алгоритм из new1.py + Параллелизация из lab_07_4.py")
    print("=" * 70)
    
    # Определение максимального количества процессов
    max_proc = min(24, cpu_count())
    print(f"\nДоступно процессоров: {cpu_count()}")
    print(f"Тестирование от 1 до {max_proc} процессов")
    
    # Запуск эксперимента
    scalability_results = experiment_scalability(
        maze_path, 
        n_agents=100, 
        max_processes=max_proc
    )
    
    # Построение графиков
    plot_scalability_results(
        scalability_results, 
        save_path='scalability_results.png'
    )
    
    # Итоговая сводка
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 70)
    
    t1 = scalability_results[1]
    t_max = scalability_results[max_proc]
    speedup = t1 / t_max
    efficiency = (speedup / max_proc) * 100
    
    print(f"\nВремя на 1 процессе: {t1:.2f}с")
    print(f"Время на {max_proc} процессах: {t_max:.2f}с")
    print(f"Ускорение: {speedup:.2f}x")
    print(f"Эффективность: {efficiency:.1f}%")
    
    # Оптимальное количество процессов
    best_speedup = max(t1 / scalability_results[n] for n in scalability_results.keys())
    best_n_proc = [n for n in scalability_results.keys()
                   if abs((t1 / scalability_results[n]) - best_speedup) < 0.01][0]
    print(f"Оптимальное число процессов: {best_n_proc} (ускорение {best_speedup:.2f}x)")
    
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН!")
    print("=" * 70)
