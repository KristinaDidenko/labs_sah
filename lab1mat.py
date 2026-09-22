import os
import matplotlib.pyplot as plt
import numpy as np

# Основні параметри
VARIANT = 100
HARMONICS_NUM = 10
DOMAIN_MIN, DOMAIN_MAX = -np.pi, np.pi


def target_function(x, n=VARIANT):
    """Цільова функція f(x) = x^n * exp(-x^2 / n)"""
    return (x**n) * np.exp(-(x**2) / n)


def calculate_fourier_coeffs(func, n_harmonics, x_min, x_max, grid_points=2000):
    """Обчислення коефіцієнтів ряду Фур'є"""
    x = np.linspace(x_min, x_max, grid_points)
    dx = x[1] - x[0]
    y = func(x)
    period = x_max - x_min

    a_coeffs = np.zeros(n_harmonics + 1)
    # Оскільки f(x) парна — всі коефіцієнти b_k (синуси) аналітично дорівнюють 0
    b_coeffs = np.zeros(n_harmonics + 1)

    for k in range(n_harmonics + 1):
        a_coeffs[k] = (2.0 / period) * np.sum(y * np.cos(k * x)) * dx

    return a_coeffs, b_coeffs


def evaluate_fourier_series(x, a_coeffs, b_coeffs):
    """Відновлення значення функції за коефіцієнтами Фур'є"""
    result = np.full_like(x, a_coeffs[0] / 2.0)
    for k in range(1, len(a_coeffs)):
        result += a_coeffs[k] * np.cos(k * x) + b_coeffs[k] * np.sin(k * x)
    return result


def compute_relative_error(y_true, y_pred):
    """Обчислення відносної L2-похибки"""
    norm_diff = np.linalg.norm(y_true - y_pred)
    norm_true = np.linalg.norm(y_true)
    return norm_diff / norm_true if norm_true != 0 else 0.0


# 1. Генерація даних
x_grid = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 1000)
y_exact = target_function(x_grid)

a_k, b_k = calculate_fourier_coeffs(
    target_function, HARMONICS_NUM, DOMAIN_MIN, DOMAIN_MAX
)
y_approx = evaluate_fourier_series(x_grid, a_k, b_k)

rel_err = compute_relative_error(y_exact, y_approx)
print(f"Відносна похибка (N = {HARMONICS_NUM}): {rel_err:.5e}")

# 2. Збереження результатів у файл
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, f"fourier_results_var{VARIANT}.txt")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(f"Кількість гармонік N: {HARMONICS_NUM}\n")
    f.write(f"Відносна похибка: {rel_err:.4%}\n\n")
    f.write("Коефіцієнти a_k:\n")
    for k, val in enumerate(a_k):
        f.write(f"  a_{k} = {val:.8f}\n")
    f.write("\nКоефіцієнти b_k:\n")
    for k, val in enumerate(b_k[1:], start=1):
        f.write(f"  b_{k} = {val:.8f}\n")

print(f"Результати збережено у файл '{file_path}'")

# 3. Побудова графіків
plt.figure(figsize=(11, 8))

# Верхній графік: Зум на відрізку [-1, 1]
plt.subplot(2, 2, (1, 2))
plt.plot(x_grid, y_exact, label="Точна f(x)", color="#1f77b4", linewidth=2)
plt.plot(
    x_grid,
    y_approx,
    label=f"Ряд Фур'є (N={HARMONICS_NUM})",
    color="#d62728",
    linestyle="--",
    linewidth=1.8,
)
plt.title(
    f"Наближення функції рядом Фур'є (Варіант {VARIANT}) — Зум [-1, 1]",
    fontsize=11,
    fontweight="bold",
)
plt.xlabel("x")
plt.ylabel("f(x)")

# Масштабування осей під діапазон [-1, 1]
plt.xlim(-1, 1)
mask = (x_grid >= -1) & (x_grid <= 1)
y_sub = y_exact[mask]
plt.ylim(y_sub.min() - 0.1, y_sub.max() + 0.2)

plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.7)

# Нижній лівий графік: Спектр a_k
plt.subplot(2, 2, 3)
plt.stem(range(len(a_k)), a_k, linefmt="b-", markerfmt="bo", basefmt="k-")
plt.title("Спектр коефіцієнтів a_k (косинуси)")
plt.xlabel("Індекс k")
plt.ylabel("a_k")
plt.grid(True, linestyle=":", alpha=0.7)

# Нижній правий графік: Спектр b_k
plt.subplot(2, 2, 4)
plt.stem(
    range(1, len(b_k)), b_k[1:], linefmt="r-", markerfmt="ro", basefmt="k-"
)
plt.title("Спектр коефіцієнтів b_k (синуси)")
plt.xlabel("Індекс k")
plt.ylabel("b_k")
plt.ylim(-1, 1)  # Масштаб фіксовано на 0
plt.grid(True, linestyle=":", alpha=0.7)

plt.tight_layout()
plt.show()