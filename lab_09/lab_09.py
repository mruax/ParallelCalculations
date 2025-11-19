import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import time
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')

# Константы
INITIAL_CAPITAL = 1_000_000  # 1 млн рублей
QUARTERS = 40  # 10 лет * 4 квартала
INFLATION_QUARTER = 0.025  # 2.5% инфляция за квартал
MIN_PROJECTS = 20  # Минимум параллельных проектов


@dataclass
class Project:
    """Инвестиционный проект"""
    start_quarter: int  # Квартал начала (0-39)
    duration_quarters: int  # Длительность в кварталах (4, 8, 12, 16)
    annual_return: float  # Годовая доходность (10-30%)
    name: str

    @property
    def end_quarter(self) -> int:
        return self.start_quarter + self.duration_quarters

    @property
    def quarterly_return(self) -> float:
        """Квартальная доходность"""
        return self.annual_return / 4


def generate_projects(num_projects: int = 20) -> List[Project]:
    """Генерация инвестиционных проектов с пересечениями"""
    projects = []
    np.random.seed(42)

    # Возможные длительности: 1, 2, 3, 4 года = 4, 8, 12, 16 кварталов
    durations = [4, 8, 12, 16]

    for i in range(num_projects):
        # Выбираем длительность
        duration = np.random.choice(durations)

        # Стартовый квартал - так чтобы проект не выходил за 10 лет
        max_start = QUARTERS - duration
        start = np.random.randint(0, max_start + 1)

        # Годовая доходность 10-30%, со смещением в зависимости от длительности
        # Более длинные проекты имеют немного выше доходность в среднем
        base_return = np.random.uniform(0.10, 0.30)
        duration_bonus = (duration / 16) * 0.03  # До 3% бонус за длинные проекты
        annual_return = base_return + duration_bonus
        annual_return = min(annual_return, 0.30)  # Ограничение 30%

        project = Project(
            start_quarter=start,
            duration_quarters=duration,
            annual_return=annual_return,
            name=f"P{i + 1}"
        )
        projects.append(project)

    return projects


def visualize_projects(projects: List[Project], filename: str = 'outputs/projects_timeline.png'):
    """Визуализация временной шкалы проектов"""
    fig, ax = plt.subplots(figsize=(16, 10))

    colors = plt.cm.tab20(np.linspace(0, 1, len(projects)))

    for idx, project in enumerate(projects):
        ax.barh(idx, project.duration_quarters, left=project.start_quarter,
                height=0.8, color=colors[idx], alpha=0.7,
                label=f'{project.name}: {project.annual_return * 100:.1f}% годовых')

        # Добавляем текст с информацией
        mid_point = project.start_quarter + project.duration_quarters / 2
        ax.text(mid_point, idx, f'{project.annual_return * 100:.1f}%',
                ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Квартал', fontsize=12)
    ax.set_ylabel('Проект', fontsize=12)
    ax.set_title('Временная шкала инвестиционных проектов', fontsize=14, fontweight='bold')
    ax.set_xlim(0, QUARTERS)
    ax.set_ylim(-1, len(projects))
    ax.grid(True, alpha=0.3, axis='x')

    # Легенда в две колонки
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


class Individual:
    """Особь в генетическом алгоритме"""

    def __init__(self, projects: List[Project], random_init: bool = True):
        self.projects = projects
        self.num_projects = len(projects)

        if random_init:
            # Гены - доля капитала в каждом проекте для каждого квартала
            # Размерность: QUARTERS x num_projects
            self.genes = np.random.random((QUARTERS, self.num_projects))
            # Нормализуем так, чтобы в каждом квартале сумма долей <= 1
            for q in range(QUARTERS):
                total = self.genes[q].sum()
                if total > 0:
                    self.genes[q] /= (total * np.random.uniform(1.0, 2.0))  # Оставляем часть денег свободной
        else:
            self.genes = np.zeros((QUARTERS, self.num_projects))

        self.fitness = 0.0

    def calculate_fitness(self) -> float:
        """Расчет эффективности инвестиционной стратегии"""
        capital = INITIAL_CAPITAL

        for quarter in range(QUARTERS):
            # Применяем инфляцию
            inflation_factor = (1 - INFLATION_QUARTER)
            capital *= inflation_factor

            # Определяем активные проекты в этом квартале
            active_projects = []
            for i, project in enumerate(self.projects):
                if project.start_quarter <= quarter < project.end_quarter:
                    active_projects.append(i)

            if not active_projects:
                continue

            # Инвестируем согласно генам
            investments = {}
            total_allocated = 0

            for proj_idx in active_projects:
                # Начало проекта - инвестируем
                if quarter == self.projects[proj_idx].start_quarter:
                    allocation = self.genes[quarter][proj_idx]
                    invest_amount = capital * allocation
                    invest_amount = min(invest_amount, capital - total_allocated)

                    if invest_amount > 0:
                        investments[proj_idx] = invest_amount
                        total_allocated += invest_amount

            # Вычитаем инвестированное
            capital -= total_allocated

            # Получаем доход от активных проектов (проинвестированных ранее)
            # Упрощенная модель: считаем, что доход приходит каждый квартал
            for proj_idx in active_projects:
                project = self.projects[proj_idx]
                if quarter > project.start_quarter:  # Доход со следующего квартала
                    # Вычисляем, сколько было инвестировано в этот проект
                    invested = self.genes[project.start_quarter][proj_idx] * INITIAL_CAPITAL
                    # Квартальный доход
                    quarterly_income = invested * project.quarterly_return
                    capital += quarterly_income

        self.fitness = capital
        return self.fitness


def mutate(individual: Individual, mutation_rate: float = 0.1) -> Individual:
    """Мутация особи"""
    new_individual = Individual(individual.projects, random_init=False)
    new_individual.genes = individual.genes.copy()

    # Мутируем случайные гены
    num_mutations = max(1, int(mutation_rate * QUARTERS * individual.num_projects))
    for _ in range(num_mutations):
        q = np.random.randint(0, QUARTERS)
        p = np.random.randint(0, individual.num_projects)
        new_individual.genes[q, p] = np.random.random()

    # Нормализуем
    for q in range(QUARTERS):
        total = new_individual.genes[q].sum()
        if total > 0:
            new_individual.genes[q] /= (total * np.random.uniform(1.0, 2.0))

    return new_individual


def crossover_blend(parent1: Individual, parent2: Individual, alpha: float = None) -> Individual:
    """Скрещивание смешиванием"""
    if alpha is None:
        alpha = np.random.random()

    new_individual = Individual(parent1.projects, random_init=False)
    new_individual.genes = alpha * parent1.genes + (1 - alpha) * parent2.genes

    return new_individual


def crossover_split(parent1: Individual, parent2: Individual) -> Individual:
    """Скрещивание разделением (половина генов от каждого родителя)"""
    new_individual = Individual(parent1.projects, random_init=False)
    new_individual.genes = parent1.genes.copy()

    # Половина по кварталам
    split_point = QUARTERS // 2
    new_individual.genes[split_point:] = parent2.genes[split_point:]

    return new_individual


def evaluate_individual_simple(genes_and_projects):
    """Упрощенная функция для параллельного вычисления fitness"""
    genes, projects = genes_and_projects
    # Создаем временную особь для вычисления fitness
    ind = Individual(projects, random_init=False)
    ind.genes = genes
    return ind.calculate_fitness()


def evaluate_population_parallel(population: List[Individual], num_processes: int, pool=None) -> List[float]:
    """
    Параллельное вычисление fitness для популяции
    Каждый процесс обрабатывает свою часть популяции (population_size / num_processes)
    """
    if num_processes <= 1:
        # Последовательное вычисление
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.calculate_fitness())
        return fitnesses

    # Подготовка данных для параллельной обработки
    # Передаем только гены и проекты, чтобы избежать проблем с сериализацией
    tasks = [(ind.genes, ind.projects) for ind in population]

    # Параллельное вычисление
    if pool is not None:
        fitnesses = pool.map(evaluate_individual_simple, tasks)
    else:
        with Pool(processes=num_processes) as temp_pool:
            fitnesses = temp_pool.map(evaluate_individual_simple, tasks)

    return fitnesses


def genetic_algorithm(projects: List[Project],
                      population_size: int = 2000,
                      generations: int = 100,
                      survival_rate: float = 0.3,
                      num_processes: int = 1,
                      verbose: bool = True,
                      use_persistent_pool: bool = True,
                      return_time: bool = False) -> Tuple[Individual, List[float]]:
    """
    Генетический алгоритм оптимизации с параллелизацией

    Args:
        projects: Список проектов
        population_size: Размер популяции (K)
        generations: Количество поколений
        survival_rate: Доля выживших (M/K)
        num_processes: Количество процессов для параллелизации
                      Каждый процесс обрабатывает population_size / num_processes особей
        verbose: Выводить прогресс
        use_persistent_pool: Использовать один Pool для всех поколений (быстрее)
        return_time: Возвращать время вычислений
    """
    # Инициализация популяции
    population = [Individual(projects) for _ in range(population_size)]

    best_fitness_history = []
    best_individual = None
    best_fitness = 0

    # Количество выживших
    num_survivors = max(1, int(population_size * survival_rate))

    if verbose and num_processes > 1:
        individuals_per_process = population_size // num_processes
        print(f"Параллелизация: {num_processes} процессов, каждый обрабатывает ~{individuals_per_process} особей")

    # Создаем persistent pool для всех поколений
    pool = None
    if num_processes > 1 and use_persistent_pool:
        pool = Pool(processes=num_processes)

    # Замер времени только на вычисления
    computation_start = time.time()

    try:
        for generation in range(generations):
            # Вычисляем fitness для всей популяции параллельно
            fitnesses = evaluate_population_parallel(population, num_processes, pool)

            # Присваиваем fitness каждой особи
            for ind, fit in zip(population, fitnesses):
                ind.fitness = fit

            # Сортируем по fitness
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Обновляем лучшую особь
            if population[0].fitness > best_fitness:
                best_fitness = population[0].fitness
                best_individual = population[0]

            best_fitness_history.append(best_fitness)

            if verbose and (generation % 10 == 0 or generation == generations - 1):
                print(f"Поколение {generation}: Лучший fitness = {best_fitness:,.0f} руб.")

            # Отбор выживших
            survivors = population[:num_survivors]

            # Создание нового поколения
            new_population = survivors.copy()  # Элитизм - лучшие особи переходят

            while len(new_population) < population_size:
                rand = np.random.random()

                if rand < 0.45:  # Мутация (45%)
                    parent = np.random.choice(survivors)
                    offspring = mutate(parent)
                    new_population.append(offspring)

                elif rand < 0.90:  # Скрещивание (45%)
                    parent1, parent2 = np.random.choice(survivors, size=2, replace=False)

                    if np.random.random() < 0.5:  # 50% blend
                        offspring = crossover_blend(parent1, parent2)
                    else:  # 50% split
                        offspring = crossover_split(parent1, parent2)

                    new_population.append(offspring)

                else:  # Новая случайная особь (10%)
                    new_population.append(Individual(projects))

            population = new_population[:population_size]

    finally:
        # Фиксируем время окончания вычислений
        computation_time = time.time() - computation_start

        # Закрываем pool если он был создан
        if pool is not None:
            pool.close()
            pool.join()

    if return_time:
        return best_individual, best_fitness_history, computation_time
    else:
        return best_individual, best_fitness_history


def run_parallelization_test(projects: List[Project], max_processes: int = 24):
    """Тестирование параллелизации от 1 до max_processes потоков"""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ПАРАЛЛЕЛИЗАЦИИ")
    print("=" * 60)

    # Параметры для теста (меньше для скорости тестирования)
    test_population = 500
    test_generations = 100

    results = []

    for num_proc in range(1, min(max_processes, cpu_count()) + 1):
        print(f"\nТестирование с {num_proc} процессом(ами)...")

        # Время считаем только для вычислений
        _, history, elapsed_time = genetic_algorithm(
            projects,
            population_size=test_population,
            generations=test_generations,
            num_processes=num_proc,
            verbose=False,
            return_time=True
        )

        results.append({
            'processes': num_proc,
            'time': elapsed_time,
            'best_fitness': history[-1]
        })

        print(f"Время вычислений: {elapsed_time:.2f} сек")
        print(f"Лучший результат: {history[-1]:,.0f} руб.")

    return results


def plot_scalability(results: List[dict], filename: str = 'outputs/scalability.png'):
    """Построение графика масштабируемости"""
    processes = [r['processes'] for r in results]
    times = [r['time'] for r in results]

    # Идеальное время
    t1 = times[0]
    ideal_times = [t1 / p for p in processes]

    # Ускорение
    speedup = [t1 / t for t in times]

    # Эффективность
    efficiency = [s / p * 100 for s, p in zip(speedup, processes)]

    # Вывод таблицы результатов в консоль
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ МАСШТАБИРУЕМОСТИ ПАРАЛЛЕЛИЗАЦИИ")
    print("=" * 60)
    print(f"{'Процессы':<12} {'Время (с)':<15} {'Ускорение':<15} {'Эффективность':<15}")
    print("-" * 60)
    for i, p in enumerate(processes):
        print(f"{p:<12} {times[i]:<15.2f} {speedup[i]:<15.2f}x {efficiency[i]:<15.1f}%")
    print("=" * 60)
    print(f"\nБазовое время (1 процесс): {t1:.2f} сек")
    print(f"Максимальное ускорение: {max(speedup):.2f}x при {processes[speedup.index(max(speedup))]} процессах")
    print(
        f"Максимальная эффективность: {max(efficiency):.1f}% при {processes[efficiency.index(max(efficiency))]} процессах")
    print()

    # Один график: Реальное vs Идеальное время
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(processes, times, 'o-', label='Реальное время', linewidth=2.5, markersize=10, color='#2196F3')
    ax.plot(processes, ideal_times, 's--', label='Идеальное время (T1/M)', linewidth=2.5, markersize=8, alpha=0.7,
            color='#4CAF50')

    ax.set_xlabel('Количество процессов', fontsize=13, fontweight='bold')
    ax.set_ylabel('Время выполнения (сек)', fontsize=13, fontweight='bold')
    ax.set_title('Масштабируемость параллелизации генетического алгоритма', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Добавляем значения на график
    for i, (p, t, it) in enumerate(zip(processes, times, ideal_times)):
        if i % 2 == 0 or i == len(processes) - 1:  # Показываем каждое второе значение
            ax.annotate(f'{t:.1f}s', xy=(p, t), xytext=(5, 5),
                        textcoords='offset points', fontsize=9, color='#2196F3')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"График масштабируемости сохранен: {filename}")


def plot_optimization_progress(history: List[float],
                               filename: str = 'outputs/optimization_progress.png'):
    """График прогресса оптимизации"""
    fig, ax = plt.subplots(figsize=(12, 6))

    generations = range(len(history))
    ax.plot(generations, history, linewidth=2, color='blue')
    ax.set_xlabel('Поколение', fontsize=11)
    ax.set_ylabel('Лучший fitness (руб.)', fontsize=11)
    ax.set_title('Прогресс оптимизации генетическим алгоритмом', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Добавляем аннотацию финального значения
    final_value = history[-1]
    ax.annotate(f'Финал: {final_value:,.0f} руб.',
                xy=(len(history) - 1, final_value),
                xytext=(-80, -30),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"График прогресса оптимизации сохранен: {filename}")


def main():
    """Основная функция"""
    print("=" * 60)
    print("СИСТЕМА ОПТИМИЗАЦИИ ИНВЕСТИЦИОННОГО ПОРТФЕЛЯ")
    print("=" * 60)
    print(f"Начальный капитал: {INITIAL_CAPITAL:,} руб.")
    print(f"Период инвестирования: {QUARTERS // 4} лет ({QUARTERS} кварталов)")
    print(f"Квартальная инфляция: {INFLATION_QUARTER * 100}%")
    print()

    # Генерация проектов
    print("Генерация инвестиционных проектов...")
    projects = generate_projects(num_projects=MIN_PROJECTS)
    print(f"Создано проектов: {len(projects)}")

    # Визуализация проектов
    print("Создание визуализации временной шкалы проектов...")
    visualize_projects(projects)
    print("Визуализация проектов сохранена: outputs/projects_timeline.png")

    # Основная оптимизация
    print("\n" + "=" * 60)
    print("ЗАПУСК ОСНОВНОЙ ОПТИМИЗАЦИИ")
    print("=" * 60)
    print("Параметры генетического алгоритма:")
    print(f"  - Размер популяции: 2000")
    print(f"  - Количество поколений: 100")
    print(f"  - Доля выживших: 30%")
    print(f"  - Процессы: {cpu_count()}")
    print()

    start_time = time.time()
    best_individual, history = genetic_algorithm(
        projects,
        population_size=2000,
        generations=100,
        survival_rate=0.3,
        num_processes=cpu_count(),
        verbose=True
    )
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 60)
    print(f"Время выполнения: {elapsed_time:.2f} секунд")
    print(f"Начальный капитал: {INITIAL_CAPITAL:,} руб.")
    print(f"Итоговый капитал: {best_individual.fitness:,.0f} руб.")
    print(f"Прирост: {(best_individual.fitness - INITIAL_CAPITAL):,.0f} руб.")
    print(f"Множитель: {best_individual.fitness / INITIAL_CAPITAL:.2f}x")
    print()

    # График прогресса оптимизации
    plot_optimization_progress(history)

    # Тестирование параллелизации
    print("\nЗапуск тестирования параллелизации...")
    parallel_results = run_parallelization_test(projects, max_processes=24)

    # График масштабируемости
    plot_scalability(parallel_results)

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 60)
    print("\nСозданные файлы:")
    print("  1. outputs/projects_timeline.png - Временная шкала проектов")
    print("  2. outputs/optimization_progress.png - Прогресс оптимизации")
    print("  3. outputs/scalability.png - Анализ масштабируемости")
    print()


if __name__ == "__main__":
    main()
