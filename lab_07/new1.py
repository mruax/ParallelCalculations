import numpy as np
import matplotlib.pyplot as plt
import time
import random
from PIL import Image


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

    def visualize_learning(self, figsize=(15, 5)):
        """Визуализация процесса обучения"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # 1. Лабиринт с путем агента
        ax1.imshow(self.maze, cmap='binary', alpha=0.7)
        path = np.array(self.visited_positions)
        if len(path) > 1:
            ax1.plot(path[:, 0], path[:, 1], 'r-', linewidth=1, alpha=0.7)
            ax1.scatter(path[:, 0], path[:, 1], c=range(len(path)),
                        cmap='viridis', s=10, alpha=0.6)

        # Отметки старта и финиша
        ax1.scatter(*self.start_pos, color='green', s=50, marker='s', label='Старт')
        ax1.scatter(*self.exit_pos, color='blue', s=50, marker='s', label='Выход')
        ax1.set_title('Путь агента в лабиринте')
        ax1.legend()

        # 2. Динамическое распределение (посещения)
        visits_plot = ax2.imshow(self.dynamic_distribution, cmap='hot', alpha=0.8)
        ax2.imshow(self.maze, cmap='binary', alpha=0.3)
        plt.colorbar(visits_plot, ax=ax2, label='Количество посещений (n)')
        ax2.set_title('Динамическое распределение (n)')
        ax2.scatter(*self.start_pos, color='green', s=50, marker='s')
        ax2.scatter(*self.exit_pos, color='blue', s=50, marker='s')

        # 3. Целевое распределение и значения K
        K_distribution = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y, x] == 1:
                    K_distribution[y, x] = self.calculate_K(x, y)

        K_plot = ax3.imshow(K_distribution, cmap='viridis', alpha=0.8)
        ax3.imshow(self.maze, cmap='binary', alpha=0.3)
        plt.colorbar(K_plot, ax=ax3, label='Значение K')
        ax3.set_title('Распределение K = x/(n+1)')
        ax3.scatter(*self.start_pos, color='green', s=50, marker='s')
        ax3.scatter(*self.exit_pos, color='blue', s=50, marker='s')

        plt.tight_layout()
        plt.show()

        # Вывод примера расчета K для нескольких клеток
        print("\nПРИМЕР РАСЧЕТА K (агент выбирает БОЛЬШИЕ значения):")
        print("Формула: K = x / (n + 1)")
        test_cells = [self.start_pos, (self.width // 2, self.height // 2), self.exit_pos]
        for cell in test_cells:
            x, y = cell
            if self.maze[y, x] == 1:
                n_val = self.dynamic_distribution[y, x]
                K_val = self.calculate_K(x, y)
                print(f"Клетка ({x:2f},{y:2f}): ñ={x:2f}, n={n_val:2f}, K={K_val:6.2f}")


def solve_maze_with_analysis(maze_path):
    """
    Полное решение лабиринта с анализом
    """
    print("РЕШЕНИЕ ЛАБИРИНТА")
    print("=" * 60)

    # Загружаем и анализируем лабиринт
    solver = MazeSolver(maze_path)

    # Создаем агента
    agent = MazeAgent(solver.maze, solver.start_pos, solver.exit_pos)

    # Обучаем агента
    print("\nЗАПУСК ОБУЧЕНИЯ АГЕНТА...")
    results = agent.train(max_iterations=30000000)

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
    maze_path = 'maze.png'
    solve_maze_with_analysis(maze_path)
    