
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from PIL import Image, ImageOps
import time
from skimage.metrics import structural_similarity


# индекс структурного сходства (SSIM) для сравнения изображений
def similarity(image1, image2):
    similarity_index, _ = structural_similarity(image1, image2, full=True)
    return similarity_index


# приведение матрицы к целым числам от 0 до 255
def scaling(agent_matrix):
    if np.all(agent_matrix == 0):
        return agent_matrix

    min_val = np.min(agent_matrix)
    max_val = np.max(agent_matrix)
    scaled_agent_matrix = np.round(255 * (agent_matrix - min_val) / (max_val - min_val))
    return scaled_agent_matrix.astype(np.uint8)


# получение соседних координат
def get_neighbor_coords(x, y, max_x, max_y):
    x = int(x)
    y = int(y)
    neighbor_coords = np.array([(x + dx, y + dy)
                                for dx in (-1, 0, 1)
                                for dy in (-1, 0, 1)
                                if (
                                            dx != 0 or dy != 0) and x + dx >= 0 and y + dy >= 0 and x + dx < max_x and y + dy < max_y])
    return neighbor_coords


# получение следующих оптимальных координат для движения агента (тех при которых разница распределений максимальна)
def get_next_step_coords(x, y, img_matrix, agent_matrix):
    neighbor_coords = get_neighbor_coords(x, y, max_x=img_matrix.shape[0], max_y=img_matrix.shape[1])

    # массив распределений в соседних точках оригинальной матрицы
    img_distributions = img_matrix[neighbor_coords[:, 0], neighbor_coords[:, 1]] / np.sum(img_matrix)

    # массив потенциальных распределений в соседних точках матрицы для агентов
    agent_potential_distributions = (1 + agent_matrix[neighbor_coords[:, 0], neighbor_coords[:, 1]]) / (
                1 + np.sum(agent_matrix))

    dist_diff = img_distributions - agent_potential_distributions

    return neighbor_coords[np.argmax(dist_diff)]


# обновление матрицы числа посещений агентами соответствующих ячеек
def update_agent_matrix(agent_coords, img_matrix, agent_matrix, num_iter):
    new_agent_matrix = np.copy(agent_matrix)
    new_agent_coords = np.copy(agent_coords)

    # назначенные агентам индексы
    agent_idxs = np.arange(new_agent_coords.shape[0])

    # установка агентов в начальные координаты
    if np.all(new_agent_matrix == 0):
        new_agent_matrix[new_agent_coords[:, 0], new_agent_coords[:, 1]] = 1

    # число итераций перед усреднением матрицы агентов между процессами
    for _ in range(num_iter):
        # перемешивание индексов агентов
        shuffled_agent_idxs = np.random.permutation(agent_idxs)

        for agent_idx in shuffled_agent_idxs:
            # поиск новой координаты в зависимости от распределения
            new_x, new_y = get_next_step_coords(*new_agent_coords[agent_idx], img_matrix, new_agent_matrix)

            # изменение количества посещений агентами позиции (255 - максимум)
            if new_agent_matrix[new_x, new_y] != 255:
                new_agent_matrix[new_x, new_y] += 1

            # обновление текущих координат каждого агента
            new_agent_coords[agent_idx] = new_x, new_y

    return new_agent_matrix, new_agent_coords


# воосстановление исходного изображения с помощью агентов
def image_reconstruction(img_matrix, agent_num, num_proc, num_iter, accuracy=0.95):
    start_time = time.time()

    agent_matrix = np.zeros_like(img_matrix, dtype=np.uint8)

    # случайные начальные координаты агентов
    agent_coords = np.empty((agent_num, 2), np.uint32)
    agent_coords[:, 0] = np.random.randint(0, img_matrix.shape[0], size=agent_num)
    agent_coords[:, 1] = np.random.randint(0, img_matrix.shape[1], size=agent_num)

    # сходство между изображениями
    ssim = similarity(img_matrix, agent_matrix)

    if num_proc == 1:
        # остановка алгоритма при достижении необходимой точности
        while ssim < accuracy:
            new_agent_matrix, agent_coords = update_agent_matrix(agent_coords, img_matrix, agent_matrix, num_iter)
            new_ssim = similarity(img_matrix, new_agent_matrix)

            if new_ssim < ssim:
                break

            ssim = new_ssim
            agent_matrix = np.copy(new_agent_matrix)

    else:
        # распределение агентов между процессами
        splitted_agent_coords = np.array_split(agent_coords, num_proc, axis=0)

        with mp.Pool(processes=num_proc) as pool:
            # остановка алгоритма при достижении необходимой точности
            while ssim < accuracy:
                # аргументы целевой функции
                args = [(splitted_agent_coords[i], img_matrix, agent_matrix, num_iter) for i in range(num_proc)]

                results = pool.starmap_async(update_agent_matrix, args).get()
                result_agent_matrix_list = np.array([res[0] for res in results])

                # сохранение координат, в которых остановились агенты на каждом процессе
                splitted_agent_coords = [res[1] for res in results]

                # усреднение матрицы
                new_agent_matrix = scaling(np.sum(result_agent_matrix_list, axis=0) / num_proc)
                new_ssim = similarity(img_matrix, new_agent_matrix)

                if new_ssim < ssim:
                    break

                ssim = new_ssim
                agent_matrix = np.copy(new_agent_matrix)

            # закрытие пула
            pool.close()

            # ожидание завершения процессов
            pool.join()

    end_time = time.time()

    return agent_matrix, ssim, end_time - start_time


def plot_time_dependence(nums_processes, times):
    plt.title('Зависимость времени вычислений\nот количества выделенных процессов')
    plt.plot(nums_processes, times, label='Изменение времени по результатам эксперимента')

    # график при начальном времени (1 процесс) и постепенном разбиении на подпроцессы (эталон: гипербола)
    plt.plot(nums_processes, [times[0] / i for i in nums_processes], label='Эталонное изменение времени')
    plt.ylabel('Время выполнения в секундах')
    plt.xlabel('Количество выделенных процессов')
    plt.legend()
    plt.show()


def show_images(images, titles):
    num_images = len(images)

    # размер сетки
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    # параметры отображения
    _, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i], cmap='gray')
            ax.set_title(titles[i])
        ax.axis('off')

    plt.show()


def main():
    # загрузка изображения
    original_img = Image.open('pigs.jpg')
    gray_img = ImageOps.grayscale(original_img)

    img_matrix = np.asarray(gray_img, dtype=np.uint8)

    # число итераций перед усреднением значений между процессами
    num_iter = 2000

    # список разного числа используемых процессов (зависит от числа доступных ЦПУ)
    nums_processes = [i + 1 for i in range(0,24)]

    # список измерений времени для разного количества используемых процессов
    times = []

    # число агентов
    agent_num = int(img_matrix.size * 0.02)

    images = [img_matrix]
    titles = ['Оригинал']

    # эксперимент для разного числа процессов
    for num_proc in nums_processes:
        agent_matrix, ssim, time = image_reconstruction(img_matrix, agent_num, num_proc, num_iter)

        images.append(agent_matrix)
        titles.append(f'CPU={num_proc}, SSIM={ssim * 100:.1f}%, t={time:.1f}с')
        times.append(time)
        print(f"{time} {num_proc}")

    show_images(images, titles)
    plot_time_dependence(nums_processes, times)


if __name__ == '__main__':
    main()

