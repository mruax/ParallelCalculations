import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, RawArray
import time
from typing import Tuple, List, Dict
import os
import ctypes

# КОНСТАНТЫ

# 8 направлений движения агента: диагонали и прямые (dx, dy)
DIRECTIONS = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1)
]


# РАБОТА С ИЗОБРАЖЕНИЯМИ

# Загрузка изображения и преобразование в целевое распределение [1, 256]
def load_image(image_path: str, max_size: int = 400) -> Tuple[np.ndarray, float]:
    img = Image.open(image_path).convert('L')

    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        try:
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        except AttributeError:
            img = img.resize(new_size, Image.LANCZOS)

    img_array = np.array(img, dtype=np.float64)
    target_distribution = img_array + 1.0  # Нормализуем к [1, 256]
    N_target = np.sum(target_distribution)  # Вычисляем норму

    return target_distribution, N_target


# Преобразование динамического распределения в изображение [0, 255]
def get_current_image(dynamic_distribution: np.ndarray, N_target: float, M: int) -> np.ndarray:
    if M == 0:
        return np.zeros_like(dynamic_distribution)

    # Нормализуем с учетом отношения норм
    normalized = (N_target / M) * dynamic_distribution
    result = np.clip(normalized - 1.0, 0, 255)

    return result


# ИНИЦИАЛИЗАЦИЯ

# Размещение агентов в случайных позициях
def initialize_agents(n_agents: int, width: int, height: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    agents_x = np.random.randint(0, width, n_agents)
    agents_y = np.random.randint(0, height, n_agents)
    return agents_x, agents_y


# Создание начального динамического распределения с размещенными агентами
def initialize_dynamic_distribution(agents_x: np.ndarray, agents_y: np.ndarray,
                                    height: int, width: int) -> Tuple[np.ndarray, int]:
    dynamic_distribution = np.zeros((height, width), dtype=np.float64)
    M = 0

    # Размещаем каждого агента на поле
    for x, y in zip(agents_x, agents_y):
        dynamic_distribution[y, x] += 1
        M += 1

    return dynamic_distribution, M


# Предварительная генерация случайных порядков проверки направлений
def prepare_random_orders(n_orders: int = 10000, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    random_orders = np.zeros((n_orders, 8), dtype=np.int32)

    # Генерируем случайную перестановку [0..7] для каждого порядка
    for i in range(n_orders):
        random_orders[i] = np.random.permutation(8)

    return random_orders


# ОСНОВНАЯ ЛОГИКА АЛГОРИТМА

# Вычисление коэффициента K = n_target(x,y) - (N_target/M) * m(x,y)
# K показывает, насколько нужны агенты в данной точке
def calculate_k(x: int, y: int, target_distribution: np.ndarray,
                dynamic_distribution: np.ndarray, N_target: float, M: int) -> float:
    if M == 0:
        return target_distribution[y, x]

    normalized_dynamic = (N_target / M) * dynamic_distribution[y, x]
    return target_distribution[y, x] - normalized_dynamic


# Перемещение агента: выбирает направление с максимальным K
def move_agent(agent_x: int, agent_y: int, target_distribution: np.ndarray,
               dynamic_distribution: np.ndarray, N_target: float, M: int,
               random_order: np.ndarray, width: int, height: int) -> Tuple[int, int]:
    best_k = float('-inf')
    best_x, best_y = agent_x, agent_y

    # Проверяем все 8 направлений в случайном порядке
    for dir_idx in random_order:
        dx, dy = DIRECTIONS[dir_idx]
        new_x, new_y = agent_x + dx, agent_y + dy

        # Проверка границ
        if 0 <= new_x < width and 0 <= new_y < height:
            k = calculate_k(new_x, new_y, target_distribution,
                            dynamic_distribution, N_target, M)

            if k > best_k:
                best_k = k
                best_x, best_y = new_x, new_y

    return best_x, best_y


# Последовательное выполнение n_steps шагов алгоритма (1 процесс)
def step_sequential(agents_x: np.ndarray, agents_y: np.ndarray,
                    dynamic_distribution: np.ndarray, M: int,
                    target_distribution: np.ndarray, N_target: float,
                    random_orders: np.ndarray, order_idx: int,
                    n_steps: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    height, width = dynamic_distribution.shape
    n_agents = len(agents_x)

    for _ in range(n_steps):
        for agent_idx in range(n_agents):
            # Получаем случайный порядок проверки
            order = random_orders[order_idx % len(random_orders)]
            order_idx += 1

            # Перемещаем агента
            new_x, new_y = move_agent(
                agents_x[agent_idx], agents_y[agent_idx],
                target_distribution, dynamic_distribution,
                N_target, M, order, width, height
            )

            # Обновляем позицию
            agents_x[agent_idx] = new_x
            agents_y[agent_idx] = new_y

            # Агент оставляет "след" в новой клетке
            dynamic_distribution[new_y, new_x] += 1
            M += 1

    return agents_x, agents_y, dynamic_distribution, M, order_idx


# МЕТРИКИ

# Вычисление метрики ошибки: среднее относительное отклонение * 100
def calculate_metric(dynamic_distribution: np.ndarray, target_distribution: np.ndarray,
                     N_target: float, M: int) -> float:
    if M == 0:
        return float('inf')

    normalized_dynamic = (N_target / M) * dynamic_distribution
    diff = np.abs(target_distribution - normalized_dynamic)
    relative_error = np.mean(diff / (target_distribution + 1e-10))

    return relative_error * 100


# ЧЕКПОИНТЫ И ИСТОРИЯ


# Сохранение чекпоинта: текущее состояние и метрика
def save_checkpoint(iteration: int, dynamic_distribution: np.ndarray,
                    target_distribution: np.ndarray, N_target: float, M: int,
                    history: List[Dict], metric_history: List[Dict],
                    save_to_history: bool = True) -> None:
    current_img = get_current_image(dynamic_distribution, N_target, M)
    metric = calculate_metric(dynamic_distribution, target_distribution, N_target, M)

    # Сохраняем полный чекпоинт с изображением
    if save_to_history:
        history.append({
            'iteration': iteration,
            'image': current_img.copy(),
            'metric': metric,
            'M': M
        })

    # Всегда сохраняем метрику для графиков
    metric_history.append({
        'iteration': iteration,
        'metric': metric
    })

    print(f"Итерация {iteration}: Метрика = {metric:.4f}, С = {M}")


# Визуализация прогресса: сетка с оригиналом и промежуточными результатами
def visualize_progress(history: List[Dict], target_distribution: np.ndarray,
                       save_path: str = 'progress.png') -> None:
    n_checkpoints = len(history)
    if n_checkpoints == 0:
        print("Нет данных для визуализации")
        return

    # Размер сетки (добавляем +1 для оригинала)
    n_cols = min(4, n_checkpoints + 1)
    n_rows = (n_checkpoints + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()

    # Первое изображение - оригинал
    original_img = target_distribution - 1.0
    axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Оригинал', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Промежуточные результаты
    for idx, checkpoint in enumerate(history):
        axes[idx + 1].imshow(checkpoint['image'], cmap='gray', vmin=0, vmax=255)
        axes[idx + 1].set_title(
            f"Итераций: {checkpoint['iteration']:,}\nМетрика: {checkpoint['metric']:.4f}",
            fontsize=14, fontweight='bold'
        )
        axes[idx + 1].axis('off')

    # Скрываем неиспользуемые оси
    for idx in range(n_checkpoints + 1, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ


# Глобальные переменные для shared memory между процессами
shared_dynamic = None
shared_target = None
shared_random_orders = None
shared_directions = None
shared_M_counter = None


# Инициализация воркера: создание numpy массивов из shared memory
def init_worker_shared(dynamic_base, target_base, orders_base, M_counter_base,
                       shape, n_orders, directions_tuple):
    global shared_dynamic, shared_target, shared_random_orders, shared_directions, shared_M_counter

    shared_dynamic = np.frombuffer(dynamic_base, dtype=np.float64).reshape(shape)
    shared_target = np.frombuffer(target_base, dtype=np.float64).reshape(shape)
    shared_random_orders = np.frombuffer(orders_base, dtype=np.int32).reshape((n_orders, 8))
    shared_M_counter = np.frombuffer(M_counter_base, dtype=np.int64)
    shared_directions = directions_tuple


# Воркер: обрабатывает батч агентов в отдельном процессе
def worker_step_batch(args):
    agent_start, agent_end, agents_x, agents_y, n_steps, order_start_idx, N_target, height, width = args

    # Локальные копии координат для батча
    local_agents_x = agents_x[agent_start:agent_end].copy()
    local_agents_y = agents_y[agent_start:agent_end].copy()

    # Локальный словарь изменений: (y, x) -> количество следов
    local_dynamic_updates = {}

    current_M = int(shared_M_counter[0])
    order_idx = order_start_idx

    # Обрабатываем n_steps для наших агентов
    for step in range(n_steps):
        for local_idx in range(len(local_agents_x)):
            x = local_agents_x[local_idx]
            y = local_agents_y[local_idx]

            order = shared_random_orders[order_idx % len(shared_random_orders)]
            order_idx += 1

            best_k = float('-inf')
            best_x, best_y = x, y

            # Проверяем все направления
            for dir_idx in order:
                dx, dy = shared_directions[dir_idx]
                new_x, new_y = x + dx, y + dy

                if 0 <= new_x < width and 0 <= new_y < height:
                    # Вычисляем K с учетом локальных изменений
                    if current_M == 0:
                        k = shared_target[new_y, new_x]
                    else:
                        dynamic_val = shared_dynamic[new_y, new_x]
                        if (new_y, new_x) in local_dynamic_updates:
                            dynamic_val += local_dynamic_updates[(new_y, new_x)]

                        normalized_dynamic = (N_target / current_M) * dynamic_val
                        k = shared_target[new_y, new_x] - normalized_dynamic

                    if k > best_k:
                        best_k = k
                        best_x, best_y = new_x, new_y

            # Обновляем локальную позицию
            local_agents_x[local_idx] = best_x
            local_agents_y[local_idx] = best_y

            # Накапливаем изменения
            key = (best_y, best_x)
            local_dynamic_updates[key] = local_dynamic_updates.get(key, 0) + 1
            current_M += 1

    return agent_start, agent_end, local_agents_x, local_agents_y, local_dynamic_updates


# Создание shared memory для всех процессов
def create_shared_memory(dynamic_distribution: np.ndarray, target_distribution: np.ndarray,
                         random_orders: np.ndarray, M: int):
    height, width = dynamic_distribution.shape

    # Динамическое распределение в shared memory
    shared_dynamic_base = RawArray(ctypes.c_double, int(height * width))
    shared_dynamic_arr = np.frombuffer(shared_dynamic_base, dtype=np.float64).reshape((height, width))
    shared_dynamic_arr[:] = dynamic_distribution

    # Целевое распределение в shared memory
    shared_target_base = RawArray(ctypes.c_double, target_distribution.flatten().tolist())

    # Случайные порядки в shared memory
    shared_orders_base = RawArray(ctypes.c_int32, random_orders.flatten().tolist())

    # Счетчик M в shared memory
    shared_M_base = RawArray(ctypes.c_int64, 1)
    shared_M_base[0] = M

    return shared_dynamic_base, shared_target_base, shared_orders_base, shared_M_base, shared_dynamic_arr


# Параллельное выполнение n_steps шагов на нескольких процессах
def step_parallel(agents_x: np.ndarray, agents_y: np.ndarray,
                  dynamic_distribution: np.ndarray, M: int,
                  N_target: float, random_orders: np.ndarray, order_idx: int,
                  pool: Pool, shared_M_base, n_steps: int, n_processes: int) -> Tuple:
    height, width = dynamic_distribution.shape
    n_agents = len(agents_x)

    # Обновляем M в shared memory
    shared_M_base[0] = M

    # Разделяем агентов между процессами
    agents_per_process = n_agents // n_processes
    tasks = []

    for i in range(n_processes):
        agent_start = i * agents_per_process
        if i == n_processes - 1:
            agent_end = n_agents
        else:
            agent_end = (i + 1) * agents_per_process

        order_start = (order_idx + i * 1000) % len(random_orders)

        tasks.append((
            agent_start, agent_end, agents_x, agents_y, n_steps,
            order_start, N_target, height, width
        ))

    # Выполняем параллельно
    results = pool.map(worker_step_batch, tasks)

    # Применяем обновления из всех процессов
    for agent_start, agent_end, new_agents_x, new_agents_y, dynamic_updates in results:
        agents_x[agent_start:agent_end] = new_agents_x
        agents_y[agent_start:agent_end] = new_agents_y

        for (y, x), count in dynamic_updates.items():
            dynamic_distribution[y, x] += count
            M += count

    order_idx = (order_idx + n_steps * n_agents) % len(random_orders)

    return agents_x, agents_y, dynamic_distribution, M, order_idx


# ЭКСПЕРИМЕНТЫ


# Запуск одного эксперимента: последовательная или параллельная версия
def run_single_experiment(image_path: str, n_processes: int, max_iterations: int,
                          checkpoint_iterations: List[int] = None,
                          verbose: bool = True, n_agents: int = 100) -> Tuple[float, Dict]:
    # Загрузка изображения
    target_distribution, N_target = load_image(image_path)
    height, width = target_distribution.shape

    # Инициализация агентов и распределения
    agents_x, agents_y = initialize_agents(n_agents, width, height)
    dynamic_distribution, M = initialize_dynamic_distribution(agents_x, agents_y, height, width)
    random_orders = prepare_random_orders()

    # История для чекпоинтов
    history = []
    metric_history = []

    # Начальный чекпоинт
    save_checkpoint(0, dynamic_distribution, target_distribution, N_target, M,
                    history, metric_history, save_to_history=True)

    # Определяем чекпоинты
    if checkpoint_iterations is None:
        checkpoint_iterations = [max_iterations]
    checkpoint_set = set(checkpoint_iterations)
    sorted_checkpoints = sorted([cp for cp in checkpoint_set if cp > 0])

    # Выбор между последовательной и параллельной версией
    if n_processes > 1:
        # Создаем shared memory
        shared_dynamic_base, shared_target_base, shared_orders_base, shared_M_base, shared_dynamic_arr = \
            create_shared_memory(dynamic_distribution, target_distribution, random_orders, M)

        # Создаем Pool процессов
        pool = Pool(
            processes=n_processes,
            initializer=init_worker_shared,
            initargs=(shared_dynamic_base, shared_target_base, shared_orders_base,
                      shared_M_base, (height, width), len(random_orders), DIRECTIONS)
        )

        dynamic_distribution = shared_dynamic_arr
        step_func = step_parallel
        batch_size = 5000
    else:
        pool = None
        shared_M_base = None
        step_func = step_sequential
        batch_size = 1000

    # Основной цикл
    start_time = time.time()
    current_iteration = 0
    order_idx = 0

    while current_iteration < max_iterations:
        remaining = max_iterations - current_iteration
        step_size = min(batch_size, remaining)

        # Уменьшаем шаг если приближаемся к чекпоинту
        for cp in sorted_checkpoints:
            if cp > current_iteration and cp <= current_iteration + step_size:
                step_size = cp - current_iteration
                break

        # Выполняем шаги
        if n_processes > 1:
            agents_x, agents_y, dynamic_distribution, M, order_idx = step_parallel(
                agents_x, agents_y, dynamic_distribution, M, N_target,
                random_orders, order_idx, pool, shared_M_base, step_size, n_processes
            )
        else:
            agents_x, agents_y, dynamic_distribution, M, order_idx = step_sequential(
                agents_x, agents_y, dynamic_distribution, M, target_distribution,
                N_target, random_orders, order_idx, step_size
            )

        current_iteration += step_size

        # Сохраняем чекпоинт если достигли нужной итерации
        if current_iteration in checkpoint_set:
            save_checkpoint(current_iteration, dynamic_distribution, target_distribution,
                            N_target, M, history, metric_history, save_to_history=True)

    total_time = time.time() - start_time

    if verbose:
        print(f"\nЗавершено за {total_time:.2f} секунд")
        print(f"Средняя скорость: {max_iterations / total_time:.0f} итераций/с")

    # Закрываем Pool если использовали параллельную версию
    if pool is not None:
        pool.close()
        pool.join()

    results = {
        'history': history,
        'metric_history': metric_history,
        'target_distribution': target_distribution,
        'dynamic_distribution': dynamic_distribution,
        'M': M,
        'N_target': N_target
    }

    return total_time, results


# Эксперимент 1: масштабируемость на 1-6 процессах
def experiment_1_scalability(image_path: str, max_processes: int = 6):
    iterations = 200000
    checkpoints = [0, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000]
    results_times = {}

    for n_proc in range(1, max_processes + 1):
        print(f"\n{'─' * 70}")
        print(f"{'ЗАПУСК: ' + str(n_proc) + ' процесс' + ('ов' if n_proc > 1 else ''):^70}")
        print(f"{'─' * 70}")

        exec_time, results = run_single_experiment(
            image_path, n_proc, iterations,
            checkpoint_iterations=checkpoints, verbose=True
        )
        results_times[n_proc] = exec_time

        # Сохраняем визуализацию
        visualize_progress(results['history'], results['target_distribution'],
                           f'exp1_progress_{n_proc}proc.png')

    return results_times


# ПОСТРОЕНИЕ ГРАФИКОВ

# График масштабируемости: реальное vs идеальное время
def build_scalability_plot(results: dict, iterations: int, save_path: str = 'experiment1_scalability.png'):
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

    return t1, times


# График сходимости: зависимость метрики от итераций
def build_convergence_plot(metric_history: list, save_path: str = 'experiment2_convergence.png'):
    iterations = [item['iteration'] for item in metric_history]
    metrics = [item['metric'] for item in metric_history]

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, metrics, label="Метрика ошибки",
             marker="o", color="blue", linewidth=2, markersize=6)
    plt.xlabel("Итерации", fontsize=16, fontweight='bold')
    plt.ylabel("Метрика ошибки", fontsize=16, fontweight='bold')
    plt.title("Сходимость алгоритма", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    image_path = "pigs.jpg"

    # Запуск эксперимента масштабируемости
    print("\nЭКСПЕРИМЕНТ: МАСШТАБИРУЕМОСТЬ (6 процессов, 200.000 итераций)")

    scalability_results = experiment_1_scalability(image_path, max_processes=6)

    # Сбор данных для графика сходимости
    print("\nСБОР ДАННЫХ ДЛЯ ГРАФИКА СХОДИМОСТИ")

    _, last_results = run_single_experiment(
        image_path, 6, 200000,
        checkpoint_iterations=[0, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000],
        verbose=False
    )

    # Построение графиков
    print("\nПОСТРОЕНИЕ ГРАФИКОВ")

    build_scalability_plot(scalability_results, 200000, 'experiment1_scalability.png')
    print("Сохранен: experiment1_scalability.png")

    build_convergence_plot(last_results['metric_history'], 'experiment2_convergence.png')
    print("Сохранен: experiment2_convergence.png")

    print("\nЭКСПЕРИМЕНТ ЗАВЕРШЕН")
