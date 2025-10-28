import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from multiprocessing import Pool, RawArray
import time
from typing import Tuple, List
import os
import ctypes

# Глобальные переменные для shared memory
shared_dynamic = None
shared_target = None
shared_maze = None
shared_shape = None
shared_M_array = None


def init_worker(dyn_arr, target_arr, maze_arr, shape, M_arr):
    """Инициализация worker процесса с shared memory"""
    global shared_dynamic, shared_target, shared_maze, shared_shape, shared_M_array
    shared_dynamic = dyn_arr
    shared_target = target_arr
    shared_maze = maze_arr
    shared_shape = shape
    shared_M_array = M_arr


def worker_task(args):
    """Задача для worker процесса"""
    agent_indices, n_iterations, directions, random_orders, start_order_idx = args

    # Получаем numpy массивы из shared memory
    height, width = shared_shape

    # Read-only доступ к maze и target
    maze = np.frombuffer(shared_maze, dtype=np.float64).reshape(height, width)
    target_dist = np.frombuffer(shared_target, dtype=np.float64).reshape(height, width)

    # Read-write доступ к динамическому распределению (локальная копия, которую потом вернем)
    local_dynamic = np.zeros((height, width), dtype=np.float64)
    local_M = 0

    # Позиции агентов (локальные для этого процесса)
    agents_x = []
    agents_y = []

    # Инициализация агентов на входе (левая сторона)
    for _ in agent_indices:
        # Находим вход слева (первая пустая клетка в первом столбце)
        entry_y = None
        for y in range(height):
            if maze[y, 0] == 1:
                entry_y = y
                break

        if entry_y is None:
            # Если нет входа, берем центр левой стороны
            entry_y = height // 2

        agents_x.append(0)
        agents_y.append(entry_y)
        local_dynamic[entry_y, 0] += 1
        local_M += 1

    current_order_idx = start_order_idx
    n_orders = len(random_orders)

    # Выполняем итерации
    for iteration in range(n_iterations):
        for agent_idx in range(len(agent_indices)):
            x, y = agents_x[agent_idx], agents_y[agent_idx]

            # Получаем случайный порядок проверки направлений
            order = random_orders[current_order_idx]
            current_order_idx = (current_order_idx + 1) % n_orders

            best_k = float('-inf')
            best_x, best_y = x, y

            # Проверяем все направления в случайном порядке
            for dir_idx in order:
                dx, dy = directions[dir_idx]
                new_x, new_y = x + dx, y + dy

                # Проверка границ и что это не стена
                if 0 <= new_x < width and 0 <= new_y < height and maze[new_y, new_x] == 1:
                    # Вычисляем K = n' / (n + 1)
                    n_visits = local_dynamic[new_y, new_x]
                    k = target_dist[new_y, new_x] / (n_visits + 1)

                    if k > best_k:
                        best_k = k
                        best_x, best_y = new_x, new_y

            # Обновляем позицию агента
            agents_x[agent_idx] = best_x
            agents_y[agent_idx] = best_y

            # Обновляем локальное динамическое распределение
            local_dynamic[best_y, best_x] += 1
            local_M += 1

    return local_dynamic, local_M


class MazeSolver:
    """
    Решатель лабиринта с использованием мультиагентного подхода
    """

    def __init__(self, maze_image_path: str, n_agents: int = 100, seed: int = 42):
        """
        Инициализация решателя

        Args:
            maze_image_path: путь к изображению лабиринта
            n_agents: количество агентов
            seed: seed для воспроизводимости
        """
        np.random.seed(seed)

        # Загрузка лабиринта
        self.load_maze(maze_image_path)

        # Параметры
        self.n_agents = n_agents
        self.height, self.width = self.maze.shape

        # Создание эталонного распределения (градиент альфа)
        self.create_alpha_gradient()

        # Инициализация динамического распределения
        self.dynamic_distribution = np.zeros((self.height, self.width), dtype=np.float64)
        self.M = 0

        # 8 направлений: (dx, dy)
        self.directions = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1)
        ]

        # Подготовка случайных порядков проверки направлений
        self.prepare_random_orders(10000)

        # История метрик
        self.metric_history = []

    def load_maze(self, maze_image_path: str):
        """Загрузка лабиринта из изображения"""
        if not os.path.exists(maze_image_path):
            print(f"Изображение {maze_image_path} не найдено. Создаем тестовый лабиринт...")
            self.create_test_maze(maze_image_path)

        img = Image.open(maze_image_path).convert('L')

        # Изменяем размер если нужно
        max_size = 200
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            try:
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            except AttributeError:
                img = img.resize(new_size, Image.LANCZOS)

        # Преобразуем в numpy массив
        img_array = np.array(img, dtype=np.float64)

        # Бинаризация: светлые пиксели = 1 (пустота), темные = 0 (стена)
        threshold = 128
        self.maze = (img_array > threshold).astype(np.float64)

        print(f"Лабиринт загружен: {self.maze.shape}")
        print(f"Пустых клеток: {np.sum(self.maze == 1):.0f}")
        print(f"Стен: {np.sum(self.maze == 0):.0f}")

    def create_test_maze(self, save_path: str):
        """Создание тестового лабиринта"""
        width, height = 200, 200
        img = Image.new('L', (width, height), color=255)  # белый фон
        draw = ImageDraw.Draw(img)

        # Рисуем стены (черные линии)
        wall_color = 0

        # Внешние границы
        draw.rectangle([0, 0, width - 1, 5], fill=wall_color)  # верх
        draw.rectangle([0, height - 6, width - 1, height - 1], fill=wall_color)  # низ

        # Создаем простой лабиринт с явным путем
        # Вертикальные стены с проходами
        for x in [40, 80, 120, 160]:
            # Случайные проходы
            gap_y = np.random.randint(20, height - 20)
            gap_size = 30

            # Верхняя часть стены
            draw.rectangle([x, 5, x + 5, gap_y - gap_size // 2], fill=wall_color)
            # Нижняя часть стены
            draw.rectangle([x, gap_y + gap_size // 2, x + 5, height - 6], fill=wall_color)

        # Горизонтальные стены с проходами
        for y in [40, 80, 120, 160]:
            gap_x = np.random.randint(20, width - 20)
            gap_size = 30

            # Левая часть стены
            draw.rectangle([5, y, gap_x - gap_size // 2, y + 5], fill=wall_color)
            # Правая часть стены
            draw.rectangle([gap_x + gap_size // 2, y, width - 6, y + 5], fill=wall_color)

        # Обеспечиваем вход слева и выход справа
        entry_y = height // 2
        exit_y = height // 2

        # Вход (слева)
        draw.rectangle([0, entry_y - 10, 10, entry_y + 10], fill=255)

        # Выход (справа)
        draw.rectangle([width - 10, exit_y - 10, width, exit_y + 10], fill=255)

        img.save(save_path)
        print(f"Тестовый лабиринт сохранен в {save_path}")

    def create_alpha_gradient(self):
        """
        Создание градиента альфа - метрики, которая растет слева направо
        Это эталонное распределение n'(x,y)
        """
        # Линейный градиент от левого края к правому
        gradient = np.zeros((self.height, self.width), dtype=np.float64)

        for x in range(self.width):
            # Альфа растет линейно слева направо
            alpha = (x / (self.width - 1)) * 100 + 1  # от 1 до 101
            gradient[:, x] = alpha

        # Применяем маску лабиринта: только в пустых клетках
        self.target_distribution = gradient * self.maze

        # Норма эталонного распределения
        self.N_target = np.sum(self.target_distribution)

        print(f"Градиент альфа создан")
        print(f"Норма эталонного распределения: {self.N_target:.2f}")

    def prepare_random_orders(self, n_orders: int = 10000):
        """Подготовка случайных порядков проверки направлений"""
        self.random_orders = np.zeros((n_orders, 8), dtype=np.int32)
        for i in range(n_orders):
            self.random_orders[i] = np.random.permutation(8)

    def run_sequential(self, n_iterations: int) -> float:
        """Последовательное выполнение (без параллелизации)"""
        start_time = time.time()

        # Инициализация агентов на входе
        agents_x = []
        agents_y = []

        # Находим вход (левая сторона)
        entry_y = None
        for y in range(self.height):
            if self.maze[y, 0] == 1:
                entry_y = y
                break

        if entry_y is None:
            entry_y = self.height // 2

        for _ in range(self.n_agents):
            agents_x.append(0)
            agents_y.append(entry_y)
            self.dynamic_distribution[entry_y, 0] += 1
            self.M += 1

        current_order_idx = 0

        # Выполняем итерации
        for iteration in range(n_iterations):
            for agent_idx in range(self.n_agents):
                x, y = agents_x[agent_idx], agents_y[agent_idx]

                # Получаем случайный порядок проверки направлений
                order = self.random_orders[current_order_idx]
                current_order_idx = (current_order_idx + 1) % len(self.random_orders)

                best_k = float('-inf')
                best_x, best_y = x, y

                # Проверяем все направления
                for dir_idx in order:
                    dx, dy = self.directions[dir_idx]
                    new_x, new_y = x + dx, y + dy

                    # Проверка границ и что это не стена
                    if 0 <= new_x < self.width and 0 <= new_y < self.height and self.maze[new_y, new_x] == 1:
                        # Вычисляем K = n' / (n + 1)
                        n_visits = self.dynamic_distribution[new_y, new_x]
                        k = self.target_distribution[new_y, new_x] / (n_visits + 1)

                        if k > best_k:
                            best_k = k
                            best_x, best_y = new_x, new_y

                # Обновляем позицию агента
                agents_x[agent_idx] = best_x
                agents_y[agent_idx] = best_y

                # Обновляем динамическое распределение
                self.dynamic_distribution[best_y, best_x] += 1
                self.M += 1

        elapsed_time = time.time() - start_time
        return elapsed_time

    def run_parallel(self, n_iterations: int, n_processes: int) -> float:
        """Параллельное выполнение"""
        start_time = time.time()

        # Создаем shared memory для массивов
        height, width = self.height, self.width

        # Динамическое распределение (изменяемое)
        shared_dynamic_arr = RawArray(ctypes.c_double, height * width)
        shared_dynamic_np = np.frombuffer(shared_dynamic_arr, dtype=np.float64).reshape(height, width)
        shared_dynamic_np[:] = 0

        # Эталонное распределение (read-only)
        shared_target_arr = RawArray(ctypes.c_double, height * width)
        shared_target_np = np.frombuffer(shared_target_arr, dtype=np.float64).reshape(height, width)
        shared_target_np[:] = self.target_distribution

        # Лабиринт (read-only)
        shared_maze_arr = RawArray(ctypes.c_double, height * width)
        shared_maze_np = np.frombuffer(shared_maze_arr, dtype=np.float64).reshape(height, width)
        shared_maze_np[:] = self.maze

        # M counter
        shared_M_arr = RawArray(ctypes.c_longlong, 1)

        # Распределяем агентов по процессам
        agents_per_process = self.n_agents // n_processes
        remainder = self.n_agents % n_processes

        tasks = []
        start_idx = 0
        for i in range(n_processes):
            # Количество агентов для этого процесса
            n_agents_proc = agents_per_process + (1 if i < remainder else 0)
            agent_indices = list(range(start_idx, start_idx + n_agents_proc))
            start_idx += n_agents_proc

            # Стартовый индекс для random orders (разный для каждого процесса)
            start_order_idx = i * 1000 % len(self.random_orders)

            tasks.append((agent_indices, n_iterations, self.directions,
                          self.random_orders, start_order_idx))

        # Запускаем pool процессов
        with Pool(processes=n_processes,
                  initializer=init_worker,
                  initargs=(shared_dynamic_arr, shared_target_arr, shared_maze_arr,
                            (height, width), shared_M_arr)) as pool:
            results = pool.map(worker_task, tasks)

        # Собираем результаты
        self.dynamic_distribution = np.zeros((height, width), dtype=np.float64)
        self.M = 0

        for local_dynamic, local_M in results:
            self.dynamic_distribution += local_dynamic
            self.M += local_M

        elapsed_time = time.time() - start_time
        return elapsed_time

    def visualize_gradient(self, save_path: str = 'gradient_alpha.png'):
        """Визуализация градиента альфа и текущего распределения"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Градиент альфа (эталонное распределение)
        im1 = axes[0].imshow(self.target_distribution, cmap='hot', aspect='auto')
        axes[0].set_title('Градиент альфа (эталон)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Динамическое распределение (пути агентов)
        im2 = axes[1].imshow(self.dynamic_distribution, cmap='hot', aspect='auto')
        axes[1].set_title(f'Пути агентов (M={self.M:,})', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Лабиринт с путями
        # Создаем RGB изображение
        maze_rgb = np.stack([self.maze] * 3, axis=2)

        # Накладываем пути (красным)
        if self.M > 0:
            normalized_paths = self.dynamic_distribution / np.max(self.dynamic_distribution)
            maze_rgb[:, :, 0] = np.maximum(maze_rgb[:, :, 0], normalized_paths)

        axes[2].imshow(maze_rgb, aspect='auto')
        axes[2].set_title('Лабиринт с путями', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация сохранена: {save_path}")
        plt.close()


def experiment_scalability(maze_path: str, max_processes: int = 24, n_iterations: int = 100000):
    """
    Эксперимент 1: Исследование масштабируемости
    """
    print(f"\nЗапуск эксперимента масштабируемости...")
    print(f"Итерации: {n_iterations:,}")
    print(f"Диапазон процессов: 1-{max_processes}")

    times = {}

    # Тестируем разное количество процессов
    for n_proc in range(1, max_processes + 1):
        print(f"\n{'=' * 60}")
        print(f"Тестирование с {n_proc} процесс(ов)...")
        print(f"{'=' * 60}")

        solver = MazeSolver(maze_path, n_agents=100, seed=42)

        if n_proc == 1:
            elapsed = solver.run_sequential(n_iterations)
        else:
            elapsed = solver.run_parallel(n_iterations, n_proc)

        times[n_proc] = elapsed

        print(f"✓ Завершено за {elapsed:.2f} секунд")
        print(f"  Всего шагов агентов: {solver.M:,}")

        # Сохраняем визуализацию для некоторых конфигураций
        if n_proc in [1, 4, 8, 12, 24]:
            solver.visualize_gradient(f'scalability_{n_proc}proc.png')

    return times


def plot_scalability(times: dict, save_path: str = 'scalability_graph.png'):
    """Построение графика масштабируемости"""
    n_processes = sorted(times.keys())
    times_list = [times[n] for n in n_processes]

    # Идеальная масштабируемость
    t1 = times[1]
    ideal_times = [t1 / n for n in n_processes]

    # Создаем график
    fig, ax = plt.subplots(figsize=(14, 8))

    # Реальное время
    ax.plot(n_processes, times_list, 'b-', linewidth=3, marker='o',
            markersize=8, label='Реальное время')

    # Идеальная масштабируемость (пунктир)
    ax.plot(n_processes, ideal_times, 'r--', linewidth=2,
            label='Идеальная масштабируемость')

    ax.set_xlabel('Количество процессов', fontsize=14, fontweight='bold')
    ax.set_ylabel('Время выполнения (секунды)', fontsize=14, fontweight='bold')
    ax.set_title('Масштабируемость: время выполнения vs количество процессов',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Делаем метки осей жирными
    ax.tick_params(axis='both', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Добавляем значения на график
    for i, (n, t) in enumerate(zip(n_processes, times_list)):
        if i % 3 == 0 or n == n_processes[-1]:
            ax.annotate(f'{t:.1f}с',
                        xy=(n, t),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=10,
                        fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nГрафик масштабируемости сохранен: {save_path}")
    plt.close()

    # Выводим таблицу результатов
    speedups = [t1 / t for t in times_list]
    efficiencies = [(s / n) * 100 for s, n in zip(speedups, n_processes)]

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА: МАСШТАБИРУЕМОСТЬ")
    print("=" * 80)
    print(f"{'Процессы':<12} {'Время (с)':<15} {'Ускорение':<15} {'Эффективность':<15}")
    print("-" * 80)
    for n, t, s, e in zip(n_processes, times_list, speedups, efficiencies):
        print(f"{n:<12} {t:>13.2f}   {s:>13.2f}x   {e:>13.1f}%")
    print("=" * 80)


def plot_gradient_profile(solver: MazeSolver, save_path: str = 'gradient_profile.png'):
    """График профиля градиента к выходу"""
    # Усредняем значения по каждому столбцу
    target_profile = np.mean(solver.target_distribution, axis=0)
    dynamic_profile = np.mean(solver.dynamic_distribution, axis=0)

    # Нормализуем динамический профиль
    if solver.M > 0:
        dynamic_profile = dynamic_profile * (np.sum(target_profile) / np.sum(dynamic_profile))

    x_coords = np.arange(solver.width)

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(x_coords, target_profile, 'b-', linewidth=3, label='Целевой градиент (альфа)', alpha=0.7)
    ax.plot(x_coords, dynamic_profile, 'r-', linewidth=2, label='Распределение агентов', alpha=0.7)

    ax.set_xlabel('Позиция X (слева направо)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Среднее значение метрики', fontsize=14, fontweight='bold')
    ax.set_title('Профиль градиента к выходу лабиринта',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Отмечаем вход и выход
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Вход', alpha=0.5)
    ax.axvline(x=solver.width - 1, color='orange', linestyle='--', linewidth=2, label='Выход', alpha=0.5)

    ax.tick_params(axis='both', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"График градиента сохранен: {save_path}")
    plt.close()


if __name__ == "__main__":
    maze_path = "maze.png"

    # Проверяем наличие изображения лабиринта
    if not os.path.exists(maze_path):
        print("Создаем тестовый лабиринт...")
        solver = MazeSolver(maze_path, n_agents=100)

    # ЭКСПЕРИМЕНТ: Масштабируемость
    print("\n" + "#" * 80)
    print("# ЭКСПЕРИМЕНТ: МАСШТАБИРУЕМОСТЬ АЛГОРИТМА ПОИСКА ПУТИ")
    print("#" * 80)

    times = experiment_scalability(maze_path, max_processes=24, n_iterations=100000)

    # Строим график масштабируемости
    plot_scalability(times, 'scalability_graph.png')

    # Строим график градиента для финального состояния
    print("\nСоздание графика градиента...")
    final_solver = MazeSolver(maze_path, n_agents=100, seed=42)
    final_solver.run_parallel(100000, 24)
    plot_gradient_profile(final_solver, 'gradient_profile.png')

    # Итоговая сводка
    print("\n\n" + "=" * 80)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 80)

    t1 = times[1]
    t24 = times[24]
    speedup = t1 / t24
    efficiency = (speedup / 24) * 100

    print(f"\nПараметры эксперимента:")
    print(f"  - Итерации: 100,000")
    print(f"  - Агенты: 100")
    print(f"  - Диапазон процессов: 1-24")

    print(f"\nРезультаты масштабируемости:")
    print(f"  - Время на 1 процессе: {t1:.2f}с")
    print(f"  - Время на 24 процессах: {t24:.2f}с")
    print(f"  - Ускорение: {speedup:.2f}x")
    print(f"  - Эффективность: {efficiency:.1f}%")

    # Находим оптимальное количество процессов
    speedups = {n: t1 / times[n] for n in times.keys()}
    best_speedup = max(speedups.values())
    best_n_proc = [n for n, s in speedups.items() if abs(s - best_speedup) < 0.01][0]
    print(f"  - Оптимальное число процессов: {best_n_proc} (ускорение {best_speedup:.2f}x)")

    # Анализ эффективности на разных диапазонах
    all_procs = sorted(times.keys())
    all_speedups = [t1 / times[n] for n in all_procs]
    all_efficiencies = [(s / n) * 100 for s, n in zip(all_speedups, all_procs)]

    avg_eff_small = np.mean([all_efficiencies[n - 1] for n in range(1, 5)])  # 1-4
    avg_eff_medium = np.mean([all_efficiencies[n - 1] for n in range(5, 9)])  # 5-8
    avg_eff_large = np.mean([all_efficiencies[n - 1] for n in range(9, 25)])  # 9-24

    print(f"\nСредняя эффективность:")
    print(f"  - 1-4 процесса: {avg_eff_small:.1f}%")
    print(f"  - 5-8 процессов: {avg_eff_medium:.1f}%")
    print(f"  - 9-24 процесса: {avg_eff_large:.1f}%")

    # Порог эффективности
    threshold_70 = next((n for n in all_procs if all_efficiencies[n - 1] < 70), 24)
    threshold_50 = next((n for n in all_procs if all_efficiencies[n - 1] < 50), 24)

    print(f"\nПороги эффективности:")
    print(f"  - Эффективность >70% до {threshold_70} процессов")
    print(f"  - Эффективность >50% до {threshold_50} процессов")

    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН!")
    print("=" * 80)

    print("\nСохраненные файлы:")
    print("  - maze.png - тестовый лабиринт")
    print("  - scalability_graph.png - график масштабируемости")
    print("  - gradient_profile.png - профиль градиента к выходу")
    print("  - scalability_*proc.png - визуализации для разных конфигураций")

    print("\n" + "=" * 80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    print("\n1. Масштабируемость:")
    print(f"   Алгоритм показывает {'хорошую' if efficiency > 50 else 'умеренную'} масштабируемость.")
    print(f"   При увеличении числа процессов с 1 до 24 достигнуто ускорение в {speedup:.1f}x.")

    print("\n2. Эффективность:")
    if avg_eff_small > 80:
        print("   На малом числе процессов (1-4) эффективность высокая (>80%).")
    if avg_eff_large > 40:
        print("   На большом числе процессов (9-24) сохраняется приемлемая эффективность.")
    else:
        print("   На большом числе процессов эффективность снижается из-за overhead'а синхронизации.")

    print("\n3. Рекомендации:")
    print(f"   Оптимальное число процессов для данной задачи: {best_n_proc}")
    print(f"   Это обеспечивает наилучший баланс между скоростью и эффективностью использования ресурсов.")

    print("\n" + "=" * 80)