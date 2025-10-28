import numpy as np
import matplotlib.pyplot as plt
import time
import random
from PIL import Image
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


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
        for j in range(self.width):
            if self.maze[self.height - 1, j] == 1:
                exit_candidates.append((j, self.height - 1))

        # Выбираем позиции
        self.start_pos = start_candidates[0] if start_candidates else (0, 1)
        self.exit_pos = exit_candidates[-1] if exit_candidates else (self.width - 1, self.height - 2)

        print(f"Стартовая позиция: {self.start_pos}")
        print(f"Выход: {self.exit_pos}")
        print(f"Найдено стартовых позиций: {len(start_candidates)}")
        print(f"Найдено выходов: {len(exit_candidates)}")


class MazeAgent:
    def __init__(self, maze, start_pos, exit_pos):
        """
        Инициализация агента для обучения в лабиринте

        Args:
            maze: матрица лабиринта (1 - проходимо, 0 - стена)
            start_pos: начальная позиция (x, y)
            exit_pos: позиция выхода (x, y)
        """
        self.maze = maze
        self.height, self.width = maze.shape
        self.start_pos = start_pos
        self.exit_pos = exit_pos

        # Инициализация распределений
        self.reset_state()

        # Генерация случайных последовательностей для направлений
        self._generate_random_sequences()

    def reset_state(self):
        """Сброс состояния агента"""
        self.position = list(self.start_pos)
        self.dynamic_distribution = np.zeros((self.height, self.width))
        self.visited_positions = [list(self.start_pos)]
        self.steps = 0
        self.sequence_index = 0
        self.backtrack_path = []  # Путь для возврата из тупика

    def _generate_random_sequences(self, num_sequences=1001):
        """Генерация случайных последовательностей для направлений"""
        self.random_sequences = []
        for _ in range(num_sequences):
            # 4 направления: вверх, вниз, влево, вправо
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            random.shuffle(directions)
            self.random_sequences.append(directions)

    def _get_valid_moves(self, x, y, avoid_traps=True):
        """Получение допустимых ходов из текущей позиции"""
        valid_moves = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Вверх, вниз, влево, вправо

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.width and 0 <= new_y < self.height and
                    self.maze[new_y, new_x] == 1):
                # Если включено избегание ловушек, проверяем K-value
                if avoid_traps:
                    K_val = self.calculate_K(new_x, new_y)
                    if K_val > -5:  # Порог для избегания плохих клеток
                        valid_moves.append((new_x, new_y))
                else:
                    valid_moves.append((new_x, new_y))

        return valid_moves

    def calculate_K(self, x, y):
        """
        Упрощенный расчет показателя K по формуле: K = x / (n(x,y) + 1)
        где x - координата (целевое распределение ñ(x,y) = x)
        n(x,y) - количество посещений клетки

        Агент будет стремиться к клеткам с БОЛЬШИМ K!
        """
        target_value = x  # ñ(x,y) = x (движение вправо)
        visits = self.dynamic_distribution[y, x]  # n(x,y)
        return target_value / (visits + 1)

    def mark_trap_path(self, trap_position, junction_position):
        """
        Помечаем путь от тупика до развилки как менее привлекательный
        """
        # Находим путь между тупиком и развилкой
        path_to_mark = []
        current_pos = trap_position

        # Поднимаемся по истории до развилки
        for i in range(len(self.visited_positions) - 2, -1, -1):
            path_to_mark.append(tuple(self.visited_positions[i]))
            if tuple(self.visited_positions[i]) == junction_position:
                break

        # Увеличиваем счетчики посещений для этого пути
        for pos in path_to_mark:
            x, y = pos
            self.dynamic_distribution[y, x] += 2  # Штраф

    def find_junction(self):
        """
        Находим последнюю развилку (клетку с более чем 1 вариантом хода)
        """
        for i in range(len(self.visited_positions) - 1, -1, -1):
            x, y = self.visited_positions[i]
            valid_moves = self._get_valid_moves(x, y, avoid_traps=False)
            if len(valid_moves) > 1:
                return (x, y)
        return self.start_pos  # Если развилка не найдена, возвращаемся к старту

    def run_iteration(self):
        """
        Выполнение одной итерации обучения агента
        Returns:
            bool: True если агент достиг выхода, False в противном случае
        """
        x, y = self.position

        # Проверка достижения выхода
        if (x, y) == self.exit_pos:
            return True

        # Получение случайной последовательности направлений
        seq_idx = self.sequence_index % len(self.random_sequences)
        random_seq = self.random_sequences[seq_idx]
        self.sequence_index += 1

        # Получение допустимых ходов
        valid_moves = self._get_valid_moves(x, y)

        # Если нет допустимых ходов - тупик
        if not valid_moves:
            # Находим развилку и помечаем путь как тупиковый
            junction = self.find_junction()
            self.mark_trap_path((x, y), junction)

            # Возвращаемся к развилке
            if junction != (x, y):  # Если мы не уже в развилке
                self.position = list(junction)
                self.visited_positions.append(list(junction))
                self.steps += 1
            return False

        best_K = -np.inf
        best_pos = (x, y)
        best_candidates = []

        # Оценка допустимых ходов в случайном порядке
        for direction in random_seq:
            new_x, new_y = x + direction[0], y + direction[1]

            # Пропускаем недопустимые ходы
            if (new_x, new_y) not in valid_moves:
                continue

            # Вычисление K по упрощенной формуле
            K_val = self.calculate_K(new_x, new_y)

            # ИЩЕМ МАКСИМАЛЬНОЕ K!
            if K_val > best_K:
                best_K = K_val
                best_pos = (new_x, new_y)
                best_candidates = [(new_x, new_y)]
            elif K_val == best_K:
                best_candidates.append((new_x, new_y))

        # Если несколько кандидатов с одинаковым K - случайный выбор
        if len(best_candidates) > 1:
            best_pos = random.choice(best_candidates)

        # Перемещение агента
        self.position = list(best_pos)
        self.visited_positions.append(list(best_pos))

        # Обновление динамического распределения
        self.dynamic_distribution[best_pos[1], best_pos[0]] += 1
        self.steps += 1

        return False

    def train(self, max_iterations=10000):
        """
        Обучение агента в лабиринте

        Args:
            max_iterations: максимальное количество итераций

        Returns:
            dict: результаты обучения
        """
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

    def get_performance_metrics(self):
        """Расчет метрик производительности"""
        if not self.visited_positions:
            return {}

        path = np.array(self.visited_positions)
        unique_cells = len(set(map(tuple, self.visited_positions)))
        efficiency = unique_cells / len(self.visited_positions) if self.visited_positions else 0

        return {
            'total_steps': self.steps,
            'unique_cells_visited': unique_cells,
            'path_efficiency': efficiency,
            'final_x': self.position[0],
            'max_x_reached': max(pos[0] for pos in self.visited_positions)
        }


def run_agents_batch(args):
    """
    Запуск пакета агентов в отдельном процессе
    """
    maze, start_pos, exit_pos, num_agents, max_iterations, process_id = args

    success_count = 0
    fastest_time = float('inf')
    fastest_agent = None

    for i in range(num_agents):
        agent = MazeAgent(maze, start_pos, exit_pos)
        results = agent.train(max_iterations)

        if results['success']:
            success_count += 1
            if results['time'] < fastest_time:
                fastest_time = results['time']
                fastest_agent = {
                    'agent_id': i,
                    'time': results['time'],
                    'iterations': results['iterations'],
                    'steps': results['steps']
                }
            # Как только нашли успешного агента, можно завершать
            break

    return {
        'process_id': process_id,
        'success_count': success_count,
        'fastest_time': fastest_time if fastest_time != float('inf') else None,
        'fastest_agent': fastest_agent
    }


def parallel_maze_solving(maze_path, max_processes=None, agents_per_process=100, max_iterations=10000):
    """
    Параллельное решение лабиринта с масштабированием количества процессов

    Args:
        maze_path: путь к файлу лабиринта
        max_processes: максимальное количество процессов (по умолчанию - количество ядер CPU)
        agents_per_process: количество агентов на процесс
        max_iterations: максимальное количество итераций на агента
    """
    print("ПАРАЛЛЕЛЬНОЕ РЕШЕНИЕ ЛАБИРИНТА")
    print("=" * 60)

    # Загружаем лабиринт
    solver = MazeSolver(maze_path)

    if max_processes is None:
        max_processes = mp.cpu_count()

    print(f"Количество ядер CPU: {mp.cpu_count()}")
    print(f"Максимальное количество процессов: {max_processes}")
    print(f"Агентов на процесс: {agents_per_process}")
    print(f"Всего агентов: {max_processes * agents_per_process}")

    results = []
    execution_times = []
    ideal_times = []

    # Запускаем для разного количества процессов
    for num_processes in range(1, max_processes + 1):
        print(f"\nЗапуск с {num_processes} процессами...")

        start_time = time.time()

        # Подготавливаем аргументы для каждого процесса
        process_args = [
            (solver.maze, solver.start_pos, solver.exit_pos, agents_per_process, max_iterations, i)
            for i in range(num_processes)
        ]

        # Запускаем процессы
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(run_agents_batch, arg) for arg in process_args]

            # Собираем результаты по мере завершения
            batch_results = []
            for future in as_completed(futures):
                result = future.result()
                batch_results.append(result)

                # Если нашли успешного агента, прерываем выполнение
                if result['success_count'] > 0:
                    # Отменяем остальные задачи
                    for f in futures:
                        f.cancel()
                    break

        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        # Расчет идеального времени (линейное масштабирование)
        if num_processes == 1:
            base_time = execution_time
        ideal_time = base_time / num_processes
        ideal_times.append(ideal_time)

        # Анализ результатов
        total_success = sum(r['success_count'] for r in batch_results)
        successful_processes = [r for r in batch_results if r['success_count'] > 0]

        results.append({
            'num_processes': num_processes,
            'execution_time': execution_time,
            'ideal_time': ideal_time,
            'total_success': total_success,
            'successful_processes': successful_processes,
            'efficiency': ideal_time / execution_time if execution_time > 0 else 0
        })

        print(f"Результат: время = {execution_time:.3f} сек, успешных агентов = {total_success}")
        if successful_processes:
            fastest = min((p['fastest_time'] for p in successful_processes if p['fastest_time'] is not None),
                          default=None)
            if fastest:
                print(f"Лучшее время агента: {fastest:.3f} сек")

    return results, base_time


def plot_scalability(results, base_time):
    """
    Построение графика масштабируемости
    """
    num_processes = [r['num_processes'] for r in results]
    execution_times = [r['execution_time'] for r in results]
    ideal_times = [r['ideal_time'] for r in results]
    efficiencies = [r['efficiency'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # График времени выполнения
    ax1.plot(num_processes, execution_times, 'bo-', label='Фактическое время', linewidth=2)
    ax1.plot(num_processes, ideal_times, 'r--', label='Идеальное время', linewidth=2)
    ax1.set_xlabel('Количество процессов')
    ax1.set_ylabel('Время выполнения (сек)')
    ax1.set_title('Масштабируемость параллельного решения')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График эффективности
    ax2.plot(num_processes, efficiencies, 'go-', linewidth=2)
    ax2.set_xlabel('Количество процессов')
    ax2.set_ylabel('Эффективность')
    ax2.set_title('Эффективность масштабирования')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)

    # Добавляем линию идеальной эффективности
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Идеальная эффективность')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Вывод результатов в таблицу
    print("\nРЕЗУЛЬТАТЫ МАСШТАБИРУЕМОСТИ:")
    print("Процессы | Время (сек) | Идеальное | Эффективность | Успешные агенты")
    print("-" * 65)
    for r in results:
        print(f"{r['num_processes']:8} | {r['execution_time']:11.3f} | {r['ideal_time']:9.3f} | "
              f"{r['efficiency']:13.3f} | {r['total_success']:14}")


def solve_maze_with_analysis(maze_path):
    """
    Полное решение лабиринта с анализом (последовательная версия)
    """
    print("ПОСЛЕДОВАТЕЛЬНОЕ РЕШЕНИЕ ЛАБИРИНТА")
    print("=" * 60)

    # Загружаем и анализируем лабиринт
    solver = MazeSolver(maze_path)

    # Создаем агента
    agent = MazeAgent(solver.maze, solver.start_pos, solver.exit_pos)

    # Обучаем агента
    print("\nЗАПУСК ОБУЧЕНИЯ АГЕНТА...")
    results = agent.train(max_iterations=30000)

    # Результаты
    print("\nРЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
    print(f"Успех: {'ДА' if results['success'] else 'НЕТ'}")
    print(f"Итерации: {results['iterations']}")
    print(f"Шаги: {results['steps']}")
    print(f"Время: {results['time']:.3f} сек")
    print(f"Финальная позиция: {results['final_position']}")

    # Метрики
    metrics = agent.get_performance_metrics()
    print(f"Уникальных клеток посещено: {metrics['unique_cells_visited']}")
    print(f"Эффективность пути: {metrics['path_efficiency']:.3f}")

    # Визуализация
    agent.visualize_learning()

    # Анализ стратегии K
    print("\nСТРАТЕГИЯ ВЫБОРА ПУТИ:")
    print("• Агент выбирает клетки с МАКСИМАЛЬНЫМ значением K")
    print("• K = x / (n + 1), где:")
    print("  - x: координата X (поощряет движение вправо)")
    print("  - n: количество посещений (штрафует повторные посещения)")
    print("• Чем БОЛЬШЕ K, тем привлекательнее клетка")


if __name__ == "__main__":
    maze_path = 'maze0.png'

    # Параллельное решение с масштабированием
    results, base_time = parallel_maze_solving(
        maze_path,
        max_processes=mp.cpu_count(),
        agents_per_process=100,
        max_iterations=10000
    )

    # Построение графиков масштабируемости
    plot_scalability(results, base_time)