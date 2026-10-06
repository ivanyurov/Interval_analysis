# -*- coding: utf-8 -*-

import itertools
import warnings
from fractions import Fraction

import numpy as np

warnings.filterwarnings("ignore")

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

EPS = 1e-12

# Данные задачи
MID_A1 = np.array([
    [0.95, 1.00],
    [1.05, 1.00],
    [1.10, 1.00],
])

RAD_DIR_REG  = np.array([[1., 0.], [1., 0.], [1., 0.]])   # регрессия
RAD_DIR_TOMO = np.ones((3, 2))                            # томография

MID_A2 = np.array([
    [1.10, 0.90, 1.10],
    [1.40, 1.00, 0.80],
    [0.80, 1.40, 1.20],
])
RAD_DIR_A2 = np.ones((3, 3))

C_GRID = np.linspace(0.5, 1.5, 200_001)   # сетка коэффициента c


# Общие утилиты
def print_matrix(m, name):
    print(f"{name}:")
    print(np.array2string(m, precision=6, suppress_small=False))


def check_membership(a_prime, mid, rad, tol=1e-9):
    """Принадлежит ли точечная матрица интервальной: |A' − mid| <= rad."""
    return bool(np.all(np.abs(a_prime - mid) <= rad + tol))


def rank_of(m):
    return int(np.linalg.matrix_rank(m))


# ЧАСТЬ 1. Прямоугольная матрица A1
def delta_curves_a1(c):
    """δ_i(c) построчно и δ(c)=max_i для обоих режимов (векторизованно)."""
    a, b = MID_A1[:, 0], MID_A1[:, 1]
    tomo = np.abs(c[:, None] * a[None, :] - b[None, :]) / (1 + np.abs(c))[:, None]
    reg  = np.abs(b[None, :] / c[:, None] - a[None, :])
    return reg.max(axis=1), tomo.max(axis=1)


def analytic_a1(mode):
    """Точные аналитические δ* и z (коэффициент col1 = z·col2)."""
    m = MID_A1[:, 0]
    c0 = MID_A1[0, 1]                      # центр второго столбца (=1)
    m_min, m_max = m.min(), m.max()
    if mode == "regression":
        # первый столбец должен стать константой z: пересечение интервалов
        delta = Fraction("0.15") / 2                     # 3/40
        z = float((m_min + m_max) / 2)                   # 1.025
    else:
        # (m_max−δ)/(1+δ) = (m_min+δ)/(1−δ)  =>  δ = (m_max−m_min)/(m_max+m_min+2·c0)
        delta = Fraction("0.15") / Fraction("4.05")      # 1/27
        d = float(delta)
        z = (m_min + d) / (c0 - d)                       # 1.025
    return float(delta), delta, z


def grid_solve_a1(mode):
    """Численный минимум δ(c) = max_i δ_i(c) по сетке c."""
    d_reg, d_tomo = delta_curves_a1(C_GRID)
    curve = d_reg if mode == "regression" else d_tomo
    i = int(np.argmin(curve))
    return float(curve[i]), float(C_GRID[i])


def build_a1(mode, delta, c):
    """Построение точечной матрицы ранга 1 при заданных δ и c (col2 = c·col1)."""
    a_prime = np.zeros((3, 2))
    if mode == "regression":
        a_prime[:, 0] = MID_A1[:, 1] / c      # = 1/c = z во всех строках
        a_prime[:, 1] = MID_A1[:, 1]
    else:
        for i in range(3):
            a, b = MID_A1[i]
            lo1, hi1 = a - delta, a + delta
            if c > 0:
                lo2, hi2 = (b - delta) / c, (b + delta) / c
            else:
                lo2, hi2 = (b + delta) / c, (b - delta) / c
            x = 0.5 * (max(lo1, lo2) + min(hi1, hi2))   # середина пересечения
            a_prime[i] = [x, c * x]
    return a_prime


def analyze_a1(mode, title):
    print(f"\nЧАСТЬ 1. A1 - {title}\n")
    print_matrix(MID_A1, "mid(A1)")
    print_matrix(RAD_DIR_REG if mode == "regression" else RAD_DIR_TOMO,
                 "Направляющая матрица радиусов (rad = δ·dir)")

    delta_ex, delta_frac, z = analytic_a1(mode)
    delta_grid, c_grid_opt = grid_solve_a1(mode)
    c_ex = 1.0 / z

    print("\nАналитическое решение:")
    print(f"  δ* = {delta_frac} = {delta_ex:.12f}")
    print(f"  z = col1/col2 = {z:.12f},  c = 1/z = {c_ex:.9f}")
    print("Численная проверка по сетке c:")
    print(f"  δ_min(сетка) = {delta_grid:.9f},  c_opt = {c_grid_opt:.9f}")
    assert abs(delta_grid - delta_ex) < 1e-4 and abs(c_grid_opt - c_ex) < 1e-3

    a_prime = build_a1(mode, delta_ex, c_ex)
    rad = delta_ex * (RAD_DIR_REG if mode == "regression" else RAD_DIR_TOMO)

    print_matrix(a_prime, "\nПостроенная точечная матрица A1'")
    r = rank_of(a_prime)
    print("\nПроверки:")
    print(f"  A1' ∈ A1            : {check_membership(a_prime, MID_A1, rad)}")
    print(f"  rank(A1') = {r} < 2   : {r < 2}")
    print(f"  max|col2 − c·col1|  = {np.max(np.abs(a_prime[:, 1] - c_ex * a_prime[:, 0])):.3e}")
    print(f"  Диапазон особенности: δ ∈ [δ*, +∞) = [{delta_ex:.6f}, +∞)")
    return delta_ex, a_prime


# ЧАСТЬ 2. Квадратная матрица A2
def radius_of_regularity(mid):
    """δ* = 1 / max_{y,z} |z^T A^{-1} y| и ранг-1 возмущение E = α·y·z^T."""
    inv = np.linalg.inv(mid)
    n = mid.shape[0]

    # Все знаковые векторы длины n (для n = 3 их 8)
    signs = list(itertools.product([-1.0, 1.0], repeat=n))

    best_abs, best_raw, best_y, best_z = -1.0, 0.0, None, None
    for y in signs:                 # 8 вариантов для y
        yv = np.array(y)
        for z in signs:             # 8 вариантов для z => 64 пары
            zv = np.array(z)
            raw = float(zv @ inv @ yv)
            if abs(raw) > best_abs:
                best_abs, best_raw, best_y, best_z = abs(raw), raw, yv, zv

    delta = 1.0 / best_abs
    alpha = -1.0 / best_raw                      # 1 + α·raw = 0  =>  det = 0
    e = alpha * np.outer(best_y, best_z)         # элементы ±δ*
    return delta, e, best_raw, best_y, best_z


def min_abs_det(delta, n_starts=8, seed=1):
    """min |det(M + δ·S)| по S ∈ [−1,1]^{3×3} (L-BFGS-B, multi-start)."""
    rng = np.random.default_rng(seed)

    def f(s):
        return abs(np.linalg.det(MID_A2 + delta * s.reshape(3, 3)))

    bounds = [(-1.0, 1.0)] * 9
    best_f, best_x = np.inf, None
    for _ in range(n_starts):
        x0 = rng.uniform(-1, 1, 9)
        res = minimize(f, x0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-16, "gtol": 1e-14, "maxiter": 5000})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x
    return best_f, best_x


def analyze_a2():
    print("\nЧАСТЬ 2. A2 - КВАДРАТНАЯ МАТРИЦА\n")
    print_matrix(MID_A2, "mid(A2)")
    det0 = np.linalg.det(MID_A2)
    print(f"\ndet(mid A2) = {det0:.6f} ≠ 0,  rank = {rank_of(MID_A2)}  (при δ=0 неособенная)")

    delta, e, raw, y, z = radius_of_regularity(MID_A2)
    a2_prime = MID_A2 + e

    print("\nТеорема о радиусе невырожденности (перебор 64 знаковых пар):")
    print(f"  max |z^T A^(-1) y| = {abs(raw):.12f}")
    print(f"  критические y = {y.astype(int)},  z = {z.astype(int)}")
    print(f"  δ* = 1/{abs(raw):.6f} = {delta:.12f}")
    print_matrix(e, "\nКритическое возмущение E = α·y·z^T")
    print_matrix(a2_prime, "\nПостроенная точечная матрица A2'")

    rad = delta * RAD_DIR_A2
    r = rank_of(a2_prime)
    print("\nПроверки:")
    print(f"  A2' ∈ A2          : {check_membership(a2_prime, MID_A2, rad)}")
    print(f"  det(A2')          = {np.linalg.det(a2_prime):.3e}")
    print(f"  rank(A2') = {r} < 3 : {r < 3}")
    print(f"  Диапазон особенности: δ ∈ [δ*, +∞) = [{delta:.6f}, +∞)")

    if HAVE_SCIPY:
        lo, hi = 0.0, 1.0
        for _ in range(30):                                   # бисекция-проверка
            mid = 0.5 * (lo + hi)
            val, _ = min_abs_det(mid, n_starts=6)
            if val < 1e-9:
                hi = mid
            else:
                lo = mid
        print(f"\nПерекрёстная проверка (бисекция + L-BFGS-B): δ_min ≈ {hi:.6f}")
    return delta, a2_prime


# Исследование поведения при δ < δ*, δ = δ*, δ > δ*
def investigate():
    print("\nИССЛЕДОВАНИЕ ПОВЕДЕНИЯ ПРИ РАЗНЫХ δ")

    d_reg, d_tomo, d_a2 = 0.075, 1.0 / 27, 0.1

    print("\n--- A1, регрессия: пересечение интервалов 1-го столбца ---")
    for d in [0.0, d_reg / 2, d_reg - 0.001, d_reg, d_reg + 0.001, 2 * d_reg]:
        lo, hi = np.max(MID_A1[:, 0] - d), np.min(MID_A1[:, 0] + d)
        ok = lo <= hi + EPS
        print(f"  δ={d:.4f}  пересечение=[{lo:.4f}, {hi:.4f}]  "
              f"{'непусто  => особая матрица ЕСТЬ' if ok else 'пусто   => особой матрицы НЕТ'}")

    print("\n--- A1, томография: пересечение интервалов z = a/b ---")
    for d in [0.0, d_tomo / 2, d_tomo - 0.001, d_tomo, d_tomo + 0.001, 2 * d_tomo]:
        lo = np.max((MID_A1[:, 0] - d) / (1 + d))
        hi = np.min((MID_A1[:, 0] + d) / (1 - d)) if d < 1 else np.inf
        ok = lo <= hi + EPS
        print(f"  δ={d:.6f}  z∈[{lo:.6f}, {hi:.6f}]  "
              f"{'непусто => особая матрица ЕСТЬ' if ok else 'пусто  => особой матрицы НЕТ'}")

    print("\n--- A2: теорема о радиусе невырожденности ---")
    for d in [0.0, d_a2 / 2, d_a2 - 0.001, d_a2, d_a2 + 0.001, 2 * d_a2]:
        verdict = "особая матрица ЕСТЬ" if d >= d_a2 - EPS else "особой матрицы НЕТ"
        line = f"  δ={d:.4f}  {verdict}"
        if HAVE_SCIPY and 0 < d:
            val, _ = min_abs_det(d, n_starts=4)   # иллюстративно
            line += f"   (min|det| численно ≈ {val:.3e})"
        print(line)


# Сводная таблица
def summary(d_reg, d_tomo, d_a2):
    print("\nСВОДНАЯ ТАБЛИЦА\n")
    print(f"  {'Матрица':<6} {'Режим':<12} {'δ_min':<12} {'Свойство'}")
    print(f"  {'A1':<6} {'томография':<12} {d_tomo:<12.6f} {'rank < 2'}")
    print(f"  {'A1':<6} {'регрессия':<12} {d_reg:<12.6f} {'rank < 2'}")
    print(f"  {'A2':<6} {'—':<12} {d_a2:<12.6f} {'det = 0'}")




if __name__ == "__main__":
    d_reg,  _ = analyze_a1("regression", "ЛИНЕЙНАЯ РЕГРЕССИЯ")
    d_tomo, _ = analyze_a1("tomography", "МАЛОРАКУРСНАЯ ТОМОГРАФИЯ")
    d_a2,   _ = analyze_a2()
    investigate()
    summary(d_reg, d_tomo, d_a2)