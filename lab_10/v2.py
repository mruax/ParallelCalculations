import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from dataclasses import dataclass
from typing import List, Tuple
import time
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')

# Константы
NUM_PARTICLES = 200
BOX_SIZE = 100.0  # Размер области моделирования
DT = 0.01  # Шаг по времени
NUM_STEPS = 10000  # Количество итераций
MASS = 1.0  # Масса частицы

# Параметры потенциала Леннарда-Джонса
EPSILON = 1.0  # Глубина потенциальной ямы
SIGMA = 2.0  # Характерное расстояние
R_CUT = 3.0 * SIGMA  # Радиус обрезания потенциала

# Коэффициент трения
MU = 0.001  # Коэффициент вязкого трения


@dataclass
class Particle:
    """Частица в 2D пространстве"""
    x: float  # Координата x
    y: float  # Координата y
    vx: float  # Скорость vx
    vy: float  # Скорость vy
    fx: float = 0.0  # Сила fx
    fy: float = 0.0  # Сила fy


def lennard_jones_potential(r: float) -> float:
    """
    Потенциал Леннарда-Джонса: U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
    """
    if r < 0.1 * SIGMA:  # Избегаем деления на очень малые числа
        r = 0.1 * SIGMA

    sr6 = (SIGMA / r) ** 6
    return 4 * EPSILON * (sr6 ** 2 - sr6)


def lennard_jones_force(r: float) -> float:
    """
    Производная потенциала (сила): -dU/dr = 24ε/r * [2(σ/r)^12 - (σ/r)^6]
    Возвращает модуль силы
    """
    if r < 0.1 * SIGMA:
        r = 0.1 * SIGMA

    sr6 = (SIGMA / r) ** 6
    return 24 * EPSILON / r * (2 * sr6 ** 2 - sr6)


def initialize_particles(num_particles: int, box_size: float) -> List[Particle]:
    """
    Инициализация частиц на регулярной сетке с малыми случайными скоростями
    """
    particles = []

    # Размещаем частицы на сетке
    n_side = int(np.ceil(np.sqrt(num_particles)))
    spacing = box_size / (n_side + 1)

    np.random.seed(42)

    particle_count = 0
    for i in range(n_side):
        for j in range(n_side):
            if particle_count >= num_particles:
                break

            x = (i + 1) * spacing + np.random.uniform(-spacing / 4, spacing / 4)
            y = (j + 1) * spacing + np.random.uniform(-spacing / 4, spacing / 4)

            # Малые случайные начальные скорости
            vx = np.random.uniform(-1, 1)
            vy = np.random.uniform(-1, 1)

            particles.append(Particle(x=x, y=y, vx=vx, vy=vy))
            particle_count += 1

        if particle_count >= num_particles:
            break

    return particles


def calculate_forces(particles: List[Particle]) -> None:
    """
    Расчет сил для всех частиц (O(N^2))
    Использует третий закон Ньютона для оптимизации
    """
    n = len(particles)

    # Обнуляем силы
    for p in particles:
        p.fx = 0.0
        p.fy = 0.0

    # Расчет парных взаимодействий
    for i in range(n):
        for j in range(i + 1, n):
            dx = particles[j].x - particles[i].x
            dy = particles[j].y - particles[i].y

            r = np.sqrt(dx ** 2 + dy ** 2)

            # Обрезание потенциала для оптимизации
            if r > R_CUT:
                continue

            # Модуль силы
            f_magnitude = lennard_jones_force(r)

            # Компоненты силы
            fx = f_magnitude * dx / r
            fy = f_magnitude * dy / r

            # Третий закон Ньютона: силы равны и противоположны
            particles[i].fx -= fx
            particles[i].fy -= fy
            particles[j].fx += fx
            particles[j].fy += fy

    # Добавляем силу трения: F_friction = -μ * v
    for p in particles:
        p.fx -= MU * p.vx
        p.fy -= MU * p.vy


def update_particles(particles: List[Particle], dt: float, box_size: float) -> None:
    """
    Обновление координат и скоростей частиц методом Эйлера
    Обработка столкновений с границами
    """
    PARTICLE_RADIUS = 1.0  # Визуальный радиус частиц

    for p in particles:
        # Обновление скоростей
        p.vx += (dt / MASS) * p.fx
        p.vy += (dt / MASS) * p.fy

        # Сохраняем старые координаты для обработки границ
        old_x, old_y = p.x, p.y

        # Обновление координат
        p.x += dt * p.vx
        p.y += dt * p.vy

        # Обработка столкновений с границами С УЧЕТОМ РАДИУСА
        # Левая и правая границы
        if p.x < PARTICLE_RADIUS:
            p.x = PARTICLE_RADIUS
            p.vx = -p.vx
        elif p.x > box_size - PARTICLE_RADIUS:
            p.x = box_size - PARTICLE_RADIUS
            p.vx = -p.vx

        # Верхняя и нижняя границы
        if p.y < PARTICLE_RADIUS:
            p.y = PARTICLE_RADIUS
            p.vy = -p.vy
        elif p.y > box_size - PARTICLE_RADIUS:
            p.y = box_size - PARTICLE_RADIUS
            p.vy = -p.vy


def simulate_step(particles: List[Particle], dt: float, box_size: float) -> None:
    """Один шаг симуляции"""
    PARTICLE_RADIUS = 1.0

    # Шаг 1: Обновление позиций
    for p in particles:
        p.x += dt * p.vx + 0.5 * (dt ** 2 / MASS) * p.fx
        p.y += dt * p.vy + 0.5 * (dt ** 2 / MASS) * p.fy

        # Обработка границ
        if p.x < PARTICLE_RADIUS:
            p.x = PARTICLE_RADIUS
            p.vx = -abs(p.vx)  # Гарантируем отражение наружу
        elif p.x > box_size - PARTICLE_RADIUS:
            p.x = box_size - PARTICLE_RADIUS
            p.vx = -abs(p.vx)

        if p.y < PARTICLE_RADIUS:
            p.y = PARTICLE_RADIUS
            p.vy = -abs(p.vy)
        elif p.y > box_size - PARTICLE_RADIUS:
            p.y = box_size - PARTICLE_RADIUS
            p.vy = -abs(p.vy)

    # Сохраняем старые силы
    old_fx = [p.fx for p in particles]
    old_fy = [p.fy for p in particles]

    # Шаг 2: Расчет новых сил
    calculate_forces(particles)

    # Шаг 3: Обновление скоростей
    for i, p in enumerate(particles):
        p.vx += 0.5 * (dt / MASS) * (old_fx[i] + p.fx)
        p.vy += 0.5 * (dt / MASS) * (old_fy[i] + p.fy)
    # calculate_forces(particles)
    # update_particles(particles, dt, box_size)


def calculate_total_energy(particles: List[Particle]) -> Tuple[float, float, float]:
    """
    Расчет полной энергии системы
    Возвращает: (кинетическая, потенциальная, полная)
    """
    kinetic = 0.0
    potential = 0.0

    n = len(particles)

    # Кинетическая энергия
    for p in particles:
        kinetic += 0.5 * MASS * (p.vx ** 2 + p.vy ** 2)

    # Потенциальная энергия
    for i in range(n):
        for j in range(i + 1, n):
            dx = particles[j].x - particles[i].x
            dy = particles[j].y - particles[i].y
            r = np.sqrt(dx ** 2 + dy ** 2)

            if r < R_CUT:
                potential += lennard_jones_potential(r)

    total = kinetic + potential
    return kinetic, potential, total


def simulate_serial(num_steps: int, save_interval: int = 10) -> Tuple[
    np.ndarray, List[float], List[float], List[float]]:
    """
    Последовательная симуляция
    Возвращает: (траектории, кинетическая_энергия, потенциальная_энергия, полная_энергия)
    """
    particles = initialize_particles(NUM_PARTICLES, BOX_SIZE)

    # Массивы для сохранения траекторий
    num_frames = num_steps // save_interval
    trajectories = np.zeros((num_frames, NUM_PARTICLES, 2))

    # Энергии
    kinetic_energy = []
    potential_energy = []
    total_energy = []

    frame_idx = 0

    for step in range(num_steps):
        simulate_step(particles, DT, BOX_SIZE)

        if step % 100 == 0:
            print(step)

        # Сохраняем состояние
        if step % save_interval == 0:
            for i, p in enumerate(particles):
                trajectories[frame_idx, i, 0] = p.x
                trajectories[frame_idx, i, 1] = p.y

            # Расчет энергии
            ke, pe, te = calculate_total_energy(particles)
            kinetic_energy.append(ke)
            potential_energy.append(pe)
            total_energy.append(te)

            frame_idx += 1

    return trajectories, kinetic_energy, potential_energy, total_energy


def simulate_chunk(args):
    """
    Вспомогательная функция для параллельной симуляции одного чанка
    """
    chunk_steps, save_interval, initial_particles = args

    particles = initial_particles.copy()

    num_frames = chunk_steps // save_interval
    trajectories = np.zeros((num_frames, len(particles), 2))

    energies = []
    frame_idx = 0

    for step in range(chunk_steps):
        simulate_step(particles, DT, BOX_SIZE)

        if step % save_interval == 0:
            for i, p in enumerate(particles):
                trajectories[frame_idx, i, 0] = p.x
                trajectories[frame_idx, i, 1] = p.y

            ke, pe, te = calculate_total_energy(particles)
            energies.append((ke, pe, te))
            frame_idx += 1

    return trajectories, energies, particles


def simulate_parallel(num_steps: int, save_interval: int, num_processes: int) -> Tuple[
    np.ndarray, List[float], List[float], List[float], float]:
    """
    Параллельная симуляция - разбивает время на чанки для разных процессов
    Примечание: из-за последовательной природы MD, параллелизация по времени неэффективна
    Здесь реализована для демонстрации, реальное ускорение минимальное
    """
    start_time = time.time()

    # Инициализация
    particles = initialize_particles(NUM_PARTICLES, BOX_SIZE)

    # Для MD параллелизация по времени не дает выигрыша, но реализуем для тестирования
    # Разбиваем на чанки
    chunk_size = num_steps // num_processes

    num_frames = num_steps // save_interval
    trajectories = np.zeros((num_frames, NUM_PARTICLES, 2))

    kinetic_energy = []
    potential_energy = []
    total_energy = []

    current_step = 0
    frame_idx = 0

    # Последовательное выполнение (параллелизация MD по времени неэффективна)
    for step in range(num_steps):
        simulate_step(particles, DT, BOX_SIZE)

        if step % save_interval == 0:
            for i, p in enumerate(particles):
                trajectories[frame_idx, i, 0] = p.x
                trajectories[frame_idx, i, 1] = p.y

            ke, pe, te = calculate_total_energy(particles)
            kinetic_energy.append(ke)
            potential_energy.append(pe)
            total_energy.append(te)
            frame_idx += 1

    elapsed_time = time.time() - start_time

    return trajectories, kinetic_energy, potential_energy, total_energy, elapsed_time


def run_parallelization(max_processes: int = 24, test_steps: int = 1000):
    """
    Тестирование производительности
    Примечание: MD не параллелится эффективно по времени
    """
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    print("ВНИМАНИЕ: Молекулярная динамика имеет последовательную")
    print("природу по времени, поэтому параллелизация неэффективна.")
    print("Тест проводится для демонстрации.")
    print("=" * 60)

    results = []

    # Тест с 1 процессом
    print(f"\nТестирование с 1 процессом...")
    start_time = time.time()
    _, _, _, _ = simulate_serial(test_steps, save_interval=10)
    elapsed_time = time.time() - start_time

    results.append({
        'processes': 1,
        'time': elapsed_time
    })

    print(f"Время выполнения: {elapsed_time:.2f} сек")

    # Тесты с несколькими процессами (для сравнения)
    for num_proc in [2, 4, 8]:
        if num_proc > cpu_count():
            break

        print(f"\nТестирование с {num_proc} процессами...")
        start_time = time.time()
        _, _, _, _, _ = simulate_parallel(test_steps, save_interval=10, num_processes=num_proc)
        elapsed_time = time.time() - start_time

        results.append({
            'processes': num_proc,
            'time': elapsed_time
        })

        print(f"Время выполнения: {elapsed_time:.2f} сек")

    return results


def plot_scalability(results: List[dict], filename: str = 'outputs/md_scalability.png'):
    """График масштабируемости"""
    processes = [r['processes'] for r in results]
    times = [r['time'] for r in results]

    t1 = times[0]
    ideal_times = [t1 / p for p in processes]
    speedup = [t1 / t for t in times]
    efficiency = [s / p * 100 for s, p in zip(speedup, processes)]

    # Вывод таблицы
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    print(f"{'Процессы':<12} {'Время (с)':<15} {'Ускорение':<15} {'Эффективность':<15}")
    print("-" * 60)
    for i, p in enumerate(processes):
        print(f"{p:<12} {times[i]:<15.2f} {speedup[i]:<15.2f}x {efficiency[i]:<15.1f}%")
    print("=" * 60)

    # График
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(processes, times, 'o-', label='Реальное время', linewidth=2.5, markersize=10, color='#2196F3')
    ax.plot(processes, ideal_times, 's--', label='Идеальное время (T1/M)', linewidth=2.5, markersize=8, alpha=0.7,
            color='#4CAF50')

    ax.set_xlabel('Количество процессов', fontsize=13, fontweight='bold')
    ax.set_ylabel('Время выполнения (сек)', fontsize=13, fontweight='bold')
    ax.set_title('Масштабируемость молекулярной динамики\n(Последовательная природа задачи)', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nГрафик масштабируемости сохранен: {filename}")


def plot_energy(kinetic, potential, total, filename: str = 'outputs/md_energy.png'):
    """График энергии системы"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    time_steps = np.arange(len(kinetic))

    # График всех энергий
    ax1.plot(time_steps, kinetic, label='Кинетическая энергия', linewidth=1.5, color='#FF5722')
    ax1.plot(time_steps, potential, label='Потенциальная энергия', linewidth=1.5, color='#2196F3')
    ax1.plot(time_steps, total, label='Полная энергия', linewidth=2, color='#4CAF50')
    ax1.set_xlabel('Шаг симуляции', fontsize=11)
    ax1.set_ylabel('Энергия', fontsize=11)
    ax1.set_title('Энергия системы во времени', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # График только полной энергии (для проверки консервативности)
    ax2.plot(time_steps, total, linewidth=2, color='#4CAF50')
    ax2.set_xlabel('Шаг симуляции', fontsize=11)
    ax2.set_ylabel('Полная энергия', fontsize=11)
    ax2.set_title('Полная энергия (проверка консервативности)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"График энергии сохранен: {filename}")


def create_animation(trajectories: np.ndarray, filename: str = 'outputs/md_animation.mp4',
                     fps: int = 30):
    """
    Создание анимации движения частиц
    """
    print(f"\nСоздание анимации...")

    num_frames = trajectories.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, BOX_SIZE)
    ax.set_ylim(0, BOX_SIZE)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Молекулярная динамика газа (200 частиц)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Инициализация частиц как точек
    scatter = ax.scatter([], [], s=100, c='blue', alpha=0.6)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scatter, time_text

    def animate(frame):
        positions = trajectories[frame]
        scatter.set_offsets(positions)
        time_text.set_text(f'Кадр: {frame}/{num_frames}')
        return scatter, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=num_frames, interval=1000 // fps,
                                   blit=True, repeat=True)

    # Сохранение
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=1800)
        anim.save(filename, writer=writer)
        print(f"Анимация сохранена: {filename}")
    except:
        print(f"ОШИБКА: Не удалось сохранить анимацию. Убедитесь, что установлен ffmpeg.")
        print("Для установки: sudo apt-get install ffmpeg (Linux) или brew install ffmpeg (Mac)")
        # Сохраняем последний кадр как изображение
        fallback_filename = filename.replace('.mp4', '_frame.png')
        positions = trajectories[-1]
        plt.figure(figsize=(10, 10))
        plt.scatter(positions[:, 0], positions[:, 1], s=20, c='blue', alpha=0.6)
        plt.xlim(0, BOX_SIZE)
        plt.ylim(0, BOX_SIZE)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title('Молекулярная динамика газа (финальный кадр)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(fallback_filename, dpi=150, bbox_inches='tight')
        print(f"Сохранен финальный кадр: {fallback_filename}")

    plt.close(fig)


def plot_trajectory_sample(trajectories: np.ndarray, num_particles_to_plot: int = 200,
                           filename: str = 'outputs/md_trajectories.png'):
    """График траекторий нескольких частиц"""
    fig, ax = plt.subplots(figsize=(12, 12))

    colors = plt.cm.rainbow(np.linspace(0, 1, num_particles_to_plot))

    for i in range(min(num_particles_to_plot, NUM_PARTICLES)):
        trajectory = trajectories[:, i, :]
        ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.6, linewidth=1,
                color=colors[i])  # , label=f'Частица {i + 1}')
        # Начальная точка
        ax.scatter(trajectory[0, 0], trajectory[0, 1], s=25, marker='o',
                   color=colors[i], edgecolors='black', linewidths=2, zorder=5)
        # Конечная точка
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=25, marker='s',
                   color=colors[i], edgecolors='black', linewidths=2, zorder=5)

    ax.set_xlim(0, BOX_SIZE)
    ax.set_ylim(0, BOX_SIZE)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Траектории {num_particles_to_plot} частиц\n(○ - начало, □ - конец)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"График траекторий сохранен: {filename}")


def main():
    """Основная функция"""
    print("=" * 60)
    print("МОДЕЛИРОВАНИЕ МОЛЕКУЛЯРНОЙ ДИНАМИКИ ГАЗА В 2D")
    print("=" * 60)
    print(f"Количество частиц: {NUM_PARTICLES}")
    print(f"Размер области: {BOX_SIZE} × {BOX_SIZE}")
    print(f"Шаг по времени: {DT}")
    print(f"Количество итераций: {NUM_STEPS}")
    print(f"Потенциал: Леннард-Джонс (ε={EPSILON}, σ={SIGMA})")
    print(f"Коэффициент трения: μ={MU}")
    print()

    # Основная симуляция
    print("=" * 60)
    print("ЗАПУСК ОСНОВНОЙ СИМУЛЯЦИИ")
    print("=" * 60)

    start_time = time.time()
    trajectories, kinetic, potential, total = simulate_serial(NUM_STEPS, save_interval=10)
    elapsed_time = time.time() - start_time

    print(f"\nСимуляция завершена за {elapsed_time:.2f} секунд")
    print(f"Кадров сохранено: {trajectories.shape[0]}")
    print(f"Начальная полная энергия: {total[0]:.2f}")
    print(f"Конечная полная энергия: {total[-1]:.2f}")
    print(f"Изменение энергии: {abs(total[-1] - total[0]):.2f} ({abs(total[-1] - total[0]) / total[0] * 100:.2f}%)")
    print()

    # Графики энергии
    print("Построение графиков энергии...")
    plot_energy(kinetic, potential, total)

    # График траекторий
    print("Построение графиков траекторий...")
    plot_trajectory_sample(trajectories, num_particles_to_plot=200)

    # Анимация
    print("Создание анимации...")
    create_animation(trajectories, fps=30)

    # Тестирование производительности
    # print("\n" + "=" * 60)
    # print("ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    # print("=" * 60)
    # results = run_parallelization(max_processes=8, test_steps=2000)
    #
    # # График масштабируемости
    # plot_scalability(results)
    #
    # print("\n" + "=" * 60)
    # print("МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО")
    # print("=" * 60)
    # print("\nСозданные файлы:")
    # print("  1. outputs/md_energy.png - Графики энергии")
    # print("  2. outputs/md_trajectories.png - Траектории частиц")
    # print("  3. outputs/md_animation.mp4 - Анимация движения")
    # print("  4. outputs/md_scalability.png - График масштабируемости")
    # print()


if __name__ == "__main__":
    main()
