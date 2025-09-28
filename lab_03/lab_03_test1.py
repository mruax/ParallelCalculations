# кол-во депутатов n - [20;24] -> перебор votes 2**n (то есть E)
# vd +1/-1 и все
# коэффициент ad - воля каждого депутата [-1;1]
# bnm - коэфф [-2;2] для конкретной пары депутатиков
#
# 1. Каждый депутат случайное значение - Vd +1/-1 -> бинарное число
# 2. Фиксированный коэффициент Ad = -0.2
# 3. Генерим коэффициент Bnm от [-2;2] для каждой пары депутатов - оно фиксированное для всех вариантов перебора
# 4. Vn Vm перебираем комбинацию -1/+1 все возможные 2**n депутатов
#
# 2**n массивов только представляет собой бинарные числа
# 00
# 01
# 10
# 11

# сохранить для мин е какие были б и вотесы через бинчисло
import numpy as np
import heapq
import multiprocessing

def symmetric_random_matrix(n, seed=None) -> np.ndarray:
    """
    Возвращает n x n симметричную матрицу:
      - диагональ = 0
      - элементы a[i,j] = a[j,i] = случайный float в [-2, 2], округлённый до 12 знаков
    Параметр seed для воспроизводимости.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    # верхний треугольник без диагонали
    upper = rng.uniform(-2.0, 2.0, size=(n, n))
    # обнулим нижний треугольник и диагональ (сохраняем только strict upper)
    upper = np.triu(upper, k=1)
    # зеркалим в нижний треугольник
    mat = upper + upper.T
    # диагональ нули (на всякий случай)
    np.fill_diagonal(mat, 0.0)
    # округление до 12 знаков после запятой
    mat = np.round(mat, 12)
    return mat


if __name__ == "__main__":
    N = 3

    # Фиксированные коэффициенты
    Ad = -0.2                       # Воля депутата
    B = symmetric_random_matrix(N, seed=42)  # Взаимоотношения депутатов
    # к каждому обращение через i + j * n чето такое

    votes = 0

    for i in range(2 ** N):
        votes_list = list(map(int, bin(i)[2:].rjust(N, '0')))
        E = sum([Ad * Vd for Vd in votes_list])
        sum2 = 0
        for n in range(N):
            for m in range(n + 1, N):
                sum2 += B[n][m] * votes_list[n] * votes_list[m]
        E += sum2
        print(E)
        votes += 1
