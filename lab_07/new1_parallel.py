import numpy as np
import matplotlib.pyplot as plt
import time
import random
from PIL import Image
from multiprocessing import Process, Manager, Value, Lock, cpu_count
import multiprocessing as mp


class MazeSolver:
    def __init__(self, maze_path: str):
        """Инициализация решателя лабиринта"""
        self.load_maze(maze_path)
        self.find_start_exit_positions()

    def load_maze(self, maze_path: str):
        """Загрузка лабиринта из изображения"""
        img = Image.open(maze_path).convert('L')
        img_array = np.array(img, dtype=np.uint8)
        self.maze = (img_array > 128).astype(np.int32)
        self.height, self.width = self.maze.shape

        print(f"Лабиринт загружен: {self.maze.shape}")
        print(f"Проходимых клеток: {np.sum(self.maze)}")
        print(f"Стен: {np.sum(1 - self.maze)}")

    def find_start_exit_positions(self):
        """Автоматическое нахождение старта и выхода"""
        start_candidates = []
        exit_candidates = []

        for i in range(self.height):
            if self.maze[i, 0] == 1:
                start_candidates.append((0, i))
        for i in range(self.height):
            if self.maze[i, self.width - 1] == 1:
                exit_candidates.append((self.width - 1, i))
        for j in range(self.width):
            if self.maze[0, j] == 1:
                start_candidates.append((j, 0))
        for j in range(self.width):
            if self.maze[self.height - 1, j] == 1:
                exit_candidates.append((j, self.height - 1))

        self.start_pos = start_candidates[0] if start_candidates else (0, 1)
        self.exit_pos = exit_candidates[-1] if exit_candidates else (self.width - 1, self.height - 2)

        print(f"Стартовая позиция: {self.start_pos}")
        print(f"Выход: {self.exit_pos}")


class MazeAgent:
    def __init__(self, maze, start_pos, exit_pos, shared_K_dict=None, K_lock=None, exit_found=None):
        """
        Инициализация агента для обучения в лабиринте
        
        Args:
            maze: матрица лабиринта (1 - проходимо, 0 - стена)
            start_pos: начальная позиция (x, y)
            exit_pos: позиция выхода (x, y)
            shared_K_dict: общий словарь для синхронизации значений K между процессами
            K_lock: блокировка для синхронизации доступа к shared_K_dict
            exit_found: флаг для остановки всех агентов при нахождении выхода
        """
        self.maze = maze
        self.height, self.width = maze.shape
        self.start_pos = start_pos
        self.exit_pos = exit_pos
        self.shared_K_dict = shared_K_dict
        self.K_lock = K_lock
        self.exit_found = exit_found
        
        self.reset_state()
        self._generate_random_sequences()

    def reset_state(self):
        """Сброс состояния агента"""
        self.position = list(self.start_pos)
        self.dynamic_distribution = np.zeros((self.height, self.width))
        self.visited_positions = [list(self.start_pos)]
        self.steps = 0
        self.sequence_index = 0
        self.backtrack_path = []

    def _generate_random_sequences(self, num_sequences=1001):
        """Генерация случайных последовательностей для направлений"""
        self.random_sequences = []
        for _ in range(num_sequences):
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            random.shuffle(directions)
            self.random_sequences.append(directions)

    def _get_valid_moves(self, x, y, avoid_traps=True):
        """Получение допустимых ходов из текущей позиции"""
        valid_moves = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.width and 0 <= new_y < self.height and
                    self.maze[new_y, new_x] == 1):
                if avoid_traps:
                    K_val = self.calculate_K(new_x, new_y)
                    if K_val > -5:
                        valid_moves.append((new_x, new_y))
                else:
                    valid_moves.append((new_x, new_y))

        return valid_moves

    def calculate_K(self, x, y):
        """
        Расчет показателя K с синхронизацией между процессами
        K = x / (n(x,y) + 1)
        """
        target_value = x
        
        # Получаем количество посещений из локального распределения
        local_visits = self.dynamic_distribution[y, x]
        
        # Если есть общий словарь, синхронизируем значения
        if self.shared_K_dict is not None and self.K_lock is not None:
            key = (x, y)
            with self.K_lock:
                if key in self.shared_K_dict:
                    shared_visits = self.shared_K_dict[key]
                else:
                    shared_visits = 0
                
                # Берем максимальное значение посещений
                visits = max(local_visits, shared_visits)
        else:
            visits = local_visits
            
        return target_value / (visits + 1)

    def update_shared_K(self, x, y):
        """Обновление общего распределения K"""
        if self.shared_K_dict is not None and self.K_lock is not None:
            key = (x, y)
            with self.K_lock:
                if key in self.shared_K_dict:
                    self.shared_K_dict[key] = self.shared_K_dict[key] + 1
                else:
                    self.shared_K_dict[key] = 1

    def mark_trap_path(self, trap_position, junction_position):
        """Помечаем путь от тупика до развилки как менее привлекательный"""
        path_to_mark = []
        current_pos = trap_position

        for i in range(len(self.visited_positions) - 2, -1, -1):
            path_to_mark.append(tuple(self.visited_positions[i]))
            if tuple(self.visited_positions[i]) == junction_position:
                break

        for pos in path_to_mark:
            x, y = pos
            self.dynamic_distribution[y, x] += 2
            self.update_shared_K(x, y)

    def find_junction(self):
        """Находим последнюю развилку"""
        for i in range(len(self.visited_positions) - 1, -1, -1):
            x, y = self.visited_positions[i]
            valid_moves = self._get_valid_moves(x, y, avoid_traps=False)
            if len(valid_moves) > 1:
                return (x, y)
        return self.start_pos

    def run_iteration(self):
        """Выполнение одной итерации обучения агента"""
        # Проверяем, не нашел ли кто-то уже выход
        if self.exit_found is not None and self.exit_found.value == 1:
            return True
            
        x, y = self.position

        if (x, y) == self.exit_pos:
            if self.exit_found is not None:
                self.exit_found.value = 1
            return True

        seq_idx = self.sequence_index % len(self.random_sequences)
        random_seq = self.random_sequences[seq_idx]
        self.sequence_index += 1

        valid_moves = self._get_valid_moves(x, y)

        if not valid_moves:
            junction = self.find_junction()
            self.mark_trap_path((x, y), junction)

            if junction != (x, y):
                self.position = list(junction)
                self.visited_positions.append(list(junction))
                self.steps += 1
            return False

        best_K = -np.inf
        best_pos = (x, y)
        best_candidates = []

        for direction in random_seq:
            new_x, new_y = x + direction[0], y + direction[1]

            if (new_x, new_y) not in valid_moves:
                continue

            K_val = self.calculate_K(new_x, new_y)

            if K_val > best_K:
                best_K = K_val
                best_pos = (new_x, new_y)
                best_candidates = [(new_x, new_y)]
            elif K_val == best_K:
                best_candidates.append((new_x, new_y))

        if len(best_candidates) > 1:
            best_pos = random.choice(best_candidates)

        self.position = list(best_pos)
        self.visited_positions.append(list(best_pos))
        self.dynamic_distribution[best_pos[1], best_pos[0]] += 1
        self.update_shared_K(best_pos[0], best_pos[1])
        self.steps += 1

        return False

    def train(self, max_iterations=10000):
        """Обучение агента в лабиринте"""
        start_time = time.time()
        success = False

        for iteration in range(max_iterations):
            success = self.run_iteration()
            if success:
                break

        training_time = time.time() - start_time

        return {
            'success': success,
            'iterations': iteration + 1,
            'steps': self.steps,
            'time': training_time,
            'final_position': self.position
        }


def worker_process(process_id, maze, start_pos, exit_pos, num_agents, 
                   shared_K_dict, K_lock, exit_found, results_queue):
    """
    Рабочий процесс, обрабатывающий группу агентов
    
    Args:
        process_id: идентификатор процесса
        maze: матрица лабиринта
        start_pos: начальная позиция
        exit_pos: позиция выхода
        num_agents: количество агентов для обработки
        shared_K_dict: общий словарь для K
        K_lock: блокировка для синхронизации
        exit_found: флаг нахождения выхода
        results_queue: очередь для результатов
    """
    print(f"[Процесс {process_id}] Запуск с {num_agents} агентами")
    
    process_start_time = time.time()
    
    for agent_id in range(num_agents):
        # Проверяем, не нашел ли кто-то уже выход
        if exit_found.value == 1:
            print(f"[Процесс {process_id}] Остановка - выход найден другим процессом")
            break
            
        agent = MazeAgent(maze, start_pos, exit_pos, shared_K_dict, K_lock, exit_found)
        result = agent.train(max_iterations=1000000)
        
        if result['success']:
            print(f"[Процесс {process_id}] Агент {agent_id} нашел выход!")
            process_time = time.time() - process_start_time
            results_queue.put({
                'process_id': process_id,
                'agent_id': agent_id,
                'success': True,
                'time': process_time,
                'result': result
            })
            break
    
    process_time = time.time() - process_start_time
    print(f"[Процесс {process_id}] Завершен за {process_time:.2f} сек")


def run_parallel_experiment(maze_path, num_processes, agents_per_process=100):
    """
    Запуск эксперимента с заданным количеством процессов
    
    Args:
        maze_path: путь к файлу лабиринта
        num_processes: количество процессов
        agents_per_process: количество агентов на процесс
        
    Returns:
        dict: результаты эксперимента
    """
    print(f"\n{'='*70}")
    print(f"ЭКСПЕРИМЕНТ С {num_processes} ПРОЦЕССАМИ")
    print(f"{'='*70}")
    
    # Загружаем лабиринт
    solver = MazeSolver(maze_path)
    
    # Создаем менеджер для общих данных
    manager = Manager()
    shared_K_dict = manager.dict()
    K_lock = manager.Lock()
    exit_found = manager.Value('i', 0)
    results_queue = manager.Queue()
    
    # Засекаем время
    start_time = time.time()
    
    # Создаем и запускаем процессы
    processes = []
    for i in range(num_processes):
        p = Process(target=worker_process, 
                   args=(i, solver.maze, solver.start_pos, solver.exit_pos, 
                        agents_per_process, shared_K_dict, K_lock, 
                        exit_found, results_queue))
        processes.append(p)
        p.start()
    
    # Ждем завершения всех процессов
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # Собираем результаты
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    print(f"\nОбщее время выполнения: {total_time:.2f} сек")
    print(f"Найдено решений: {len(results)}")
    
    return {
        'num_processes': num_processes,
        'total_time': total_time,
        'success': len(results) > 0,
        'results': results
    }


def run_scalability_test(maze_path, max_processes=None, agents_per_process=100):
    """
    Проведение теста масштабируемости
    
    Args:
        maze_path: путь к файлу лабиринта
        max_processes: максимальное количество процессов (по умолчанию = количество CPU)
        agents_per_process: количество агентов на процесс
        
    Returns:
        dict: результаты тестирования
    """
    if max_processes is None:
        max_processes = cpu_count()
    
    print(f"\n{'#'*70}")
    print(f"ТЕСТ МАСШТАБИРУЕМОСТИ")
    print(f"Максимальное количество процессов: {max_processes}")
    print(f"Агентов на процесс: {agents_per_process}")
    print(f"{'#'*70}")
    
    results = []
    
    # Запускаем эксперименты с разным количеством процессов
    for num_proc in range(1, max_processes + 1):
        result = run_parallel_experiment(maze_path, num_proc, agents_per_process)
        results.append(result)
        
        # Небольшая пауза между экспериментами
        time.sleep(1)
    
    return results


def plot_scalability(results):
    """
    Построение графика масштабируемости
    
    Args:
        results: результаты тестирования
    """
    num_processes = [r['num_processes'] for r in results]
    times = [r['total_time'] for r in results]
    
    # Идеальное время (T1 / n)
    T1 = times[0]
    ideal_times = [T1 / n for n in num_processes]
    
    # Построение графика
    plt.figure(figsize=(12, 7))
    
    plt.plot(num_processes, times, 'bo-', linewidth=2, markersize=8, label='Реальное время')
    plt.plot(num_processes, ideal_times, 'r--', linewidth=2, label='Идеальное время (T₁/n)')
    
    plt.xlabel('Количество процессов', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.title('График масштабируемости распараллеливания', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Добавляем значения на точки
    for i, (n, t) in enumerate(zip(num_processes, times)):
        plt.annotate(f'{t:.1f}s', 
                    xy=(n, t), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/scalability_graph.png', dpi=300, bbox_inches='tight')
    print("\nГрафик сохранен в outputs/scalability_graph.png")
    
    # Вычисляем эффективность
    print("\n" + "="*70)
    print("АНАЛИЗ МАСШТАБИРУЕМОСТИ")
    print("="*70)
    print(f"{'Процессы':<12} {'Время (с)':<12} {'Ускорение':<12} {'Эффективность':<15}")
    print("-"*70)
    
    for i, (n, t) in enumerate(zip(num_processes, times)):
        speedup = T1 / t
        efficiency = speedup / n * 100
        print(f"{n:<12} {t:<12.2f} {speedup:<12.2f} {efficiency:<15.1f}%")
    
    plt.show()


def main():
    """Главная функция"""
    maze_path = 'maze.png'
    
    # Параметры эксперимента
    max_processes = min(24, cpu_count())  # Ограничиваем до 8 процессов или количества CPU
    agents_per_process = 1
    
    print(f"Доступно CPU: {cpu_count()}")
    print(f"Будет использовано процессов: от 1 до {max_processes}")
    print(f"Агентов на процесс: {agents_per_process}")
    
    # Запускаем тест масштабируемости
    results = run_scalability_test(maze_path, max_processes, agents_per_process)
    
    # Строим график
    plot_scalability(results)
    
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*70)


if __name__ == "__main__":
    # Важно для Windows
    mp.set_start_method('spawn', force=True)
    main()
