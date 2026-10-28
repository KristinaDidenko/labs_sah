import math
import time
import matplotlib.pyplot as plt
import numpy as np

n = 2  # Номер варіанта
USE_COMPLEX = True  # True: комплексний вхідний вектор у Частині І; False: дійсний
EPS = 1e-10  # поріг, нижче якого коефіцієнт вважається нульовим


# ЧАСТИНА І: ДПФ у тригонометричній / комплексній формі
def dft_k_term(f, k, N):
    """k-й коефіцієнт ДПФ C_k = A_k + j*B_k (з нормуванням 1/N).

    Працює і для дійсного, і для комплексного f.
    C_k = (1/N) * Sum f[n] * (cos(a) - j*sin(a)),  a = 2*pi*k*n/N.

    Для f = a + j*b:
        Re += a*cos + b*sin
        Im += b*cos - a*sin
    Підрахунок: комплексний f -> 4 множення, 4 додавання на член;
                дійсний f    -> 2 множення, 2 додавання на член.
    Плюс 2 множення (ділення) на нормування 1/N.
    Обчислення кута та виклики cos/sin в підрахунок не входять.
    """
    is_complex = np.iscomplexobj(f)
    re = 0.0
    im = 0.0
    mult_count = 0
    add_count = 0

    for n_idx in range(N):
        angle = 2 * math.pi * k * n_idx / N
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)

        a = float(np.real(f[n_idx]))
        b = float(np.imag(f[n_idx]))

        if is_complex:
            re += a * cos_val + b * sin_val
            im += b * cos_val - a * sin_val
            mult_count += 4
            add_count += 4
        else:
            re += a * cos_val
            im -= a * sin_val
            mult_count += 2
            add_count += 2

    return complex(re / N, im / N), mult_count + 2, add_count


def calculate_dft(f, N, repeats=0):
    """ДПФ усього вектора.

    Повертає (C, час, к-сть множень, к-сть додавань).
    Якщо repeats > 0, час усереднюється за `repeats` окремих запусків
    (сам результат C обчислюється один раз, поза таймером).
    """
    C = []
    total_mult = 0
    total_add = 0
    for k in range(N):
        c_k, mult, add = dft_k_term(f, k, N)
        C.append(c_k)
        total_mult += mult
        total_add += add

    calc_time = 0.0
    if repeats > 0:
        start_time = time.perf_counter()
        for _ in range(repeats):
            for k in range(N):
                dft_k_term(f, k, N)
        calc_time = (time.perf_counter() - start_time) / repeats

    return np.array(C), calc_time, total_mult, total_add


def clean_phase(C):
    """Аргументи C_k: для майже нульових коефіцієнтів фаза = 0,
    а значення -pi приводиться до +pi для єдиного вигляду графіків."""
    phase = np.angle(C)
    phase[np.abs(C) < EPS] = 0.0
    phase[np.isclose(phase, -np.pi)] = np.pi
    return phase


# ЧАСТИНА ІІ: Відтворення аналогового сигналу
def generate_part2_samples(n_variant):
    num_dec = 96 + n_variant
    bin_str = list(format(num_dec, '08b'))  # 8-бітне представлення

    # 8-й розряд (найстарший, індекс 0)
    bin_str[0] = '1' if n_variant % 2 != 0 else '0'

    bits = [int(b) for b in bin_str]
    return bits, num_dec


def reconstruct_analog_signal(C, t, T):
    """Відтворення s(t) = Sum_k C_k * exp(j*2*pi*kk*t/T).

    Частоти k > N/2 відображаються у від'ємні kk = k - N (без накладання).
    Частота Найквіста k = N/2 береться як C_k * cos(pi*N*t/T).
    Для дійсного сигналу Im s(t) ~ 0.
    """
    N = len(C)
    s_t = np.zeros_like(t, dtype=complex)

    for k in range(N):
        if k < N // 2:
            s_t += C[k] * np.exp(1j * 2 * np.pi * k * t / T)
        elif k == N // 2:
            s_t += C[k] * np.cos(2 * np.pi * k * t / T)
        else:
            s_t += C[k] * np.exp(1j * 2 * np.pi * (k - N) * t / T)

    return s_t


def print_signal_expression(C):
    """Аналітичний вираз s(t) у тригонометричній формі (Частина ІІ, п. 3).

    Справедливий для ДІЙСНОГО сигналу. Нульові гармоніки пропускаються.
    """
    N = len(C)
    terms = []
    if abs(C[0]) >= EPS:
        terms.append(f'{C[0].real:.4f}')

    for k in range(1, N // 2):
        if abs(C[k]) < EPS:
            continue
        amp = 2 * abs(C[k])
        ph = np.angle(C[k])
        terms.append(f'{amp:.4f}*cos({2 * k}*pi*t/T {ph:+.4f})')

    k = N // 2
    if abs(C[k]) >= EPS:
        terms.append(f'{C[k].real:.4f}*cos({2 * k}*pi*t/T)')

    print('\nАналітичний вираз s(t), 0 <= t < T:')
    print('s(t) = ' + ' + '.join(terms))


def print_signal_expression_complex(C):
    """Аналітичний вираз s(t) у комплексній формі (для будь-якого сигналу)."""
    N = len(C)
    terms = []
    for k in range(N):
        if abs(C[k]) < EPS:
            continue
        kk = k if k <= N // 2 else k - N
        terms.append(f'({C[k].real:.4f}{C[k].imag:+.4f}j)*exp(j*{2 * kk}*pi*t/T)')

    print('\nАналітичний вираз s(t) у комплексній формі, 0 <= t < T:')
    print('s(t) = ' + ' + '.join(terms))


# ЧАСТИНА ІІІ: Обернене ДПФ (ОДПФ)
def calculate_idft(C, N):
    """ОДПФ: s(n) = Sum_k C_k * exp(j*2*pi*k*n/N) (1/N вже враховано в прямому ДПФ).

    Повертає КОМПЛЕКСНИЙ масив (для дійсного сигналу Im ~ 0).
    """
    s_reconstructed = []
    for n_idx in range(N):
        val = 0.0 + 0.0j
        for k in range(N):
            angle = 2 * math.pi * k * n_idx / N
            val += C[k] * (math.cos(angle) + 1j * math.sin(angle))
        s_reconstructed.append(val)
    return np.array(s_reconstructed)


def print_idft_expressions(C, s_rec, orig):
    """Аналітичні вирази відліків s(nT_d) при n = 0, 1 (Частина ІІІ, п. 2)."""
    N = len(C)
    print('\nАналітичні вирази відліків (п. 2 Частини ІІІ):')

    for n_idx in (0, 1):
        symbolic = []
        for k in range(N):
            angle = 2 * math.pi * k * n_idx / N
            w_re = round(math.cos(angle), 4) + 0.0
            w_im = round(math.sin(angle), 4) + 0.0
            symbolic.append(f'C{k}*({w_re:.4f}{w_im:+.4f}j)')

        name = 's(0)' if n_idx == 0 else 's(T_d)'
        print(f'{name} = Sum_k C_k*exp(j*2*pi*k*{n_idx}/{N})')
        print('      = ' + ' + '.join(symbolic))
        print(f'      = {s_rec[n_idx].real:.4f}  (початкове значення: {orig[n_idx]})')


# ГОЛОВНА ПРОГРАМА
if __name__ == '__main__':
    print(f'=== ЛАБОРАТОРНА РОБОТА №2 (Варіант n = {n}) ===\n')

    # --- ЧАСТИНА І ---
    N1 = 10 + n
    np.random.seed(42)
    f_re = np.random.uniform(-5, 5, N1)
    if USE_COMPLEX:
        f_im = np.random.uniform(-5, 5, N1)
        f_input = f_re + 1j * f_im  # комплексний вхідний вектор
    else:
        f_input = f_re

    C_part1, time_p1, mult_p1, add_p1 = calculate_dft(f_input, N1, repeats=1000)
    amplitudes_p1 = np.abs(C_part1)
    phases_p1 = clean_phase(C_part1)

    print('--- Частина І ---')
    print(f'Розмірність сигналу N = 10 + {n} = {N1}')
    print(f'Тип вхідного вектора: {"комплексний" if USE_COMPLEX else "дійсний"}')
    print(f'Вхідний вектор f: {np.round(f_input, 3)}')
    print(f'Час обчислення ДПФ (середнє за 1000 запусків): {time_p1 * 1000:.4f} мс')
    print(f'Кількість операцій множення: {mult_p1}')
    print(f'Кількість операцій додавання: {add_p1}')
    print(f'Загальна кількість операцій: {mult_p1 + add_p1}')
    print('(Час включає виклики cos/sin, у кількість операцій вони не входять)')

    # Перевірка з numpy
    print(f'Збіг з np.fft.fft/N: {np.allclose(C_part1, np.fft.fft(f_input) / N1)}\n')

    k_indices1 = np.arange(N1)

    # ГРАФІК 0: Вхідний вектор (Re та Im)
    fig0, axes0 = plt.subplots(2 if USE_COMPLEX else 1, 1, figsize=(8, 5))
    axes0 = np.atleast_1d(axes0)
    axes0[0].stem(k_indices1, np.real(f_input), linefmt='b-', markerfmt='bo', basefmt='r-')
    axes0[0].set_title('Вхідний вектор: Re f(n)')
    axes0[0].set_xlabel('n')
    axes0[0].grid(True)
    if USE_COMPLEX:
        axes0[1].stem(k_indices1, np.imag(f_input), linefmt='m-', markerfmt='mo', basefmt='r-')
        axes0[1].set_title('Вхідний вектор: Im f(n)')
        axes0[1].set_xlabel('n')
        axes0[1].grid(True)
    plt.tight_layout()
    plt.show()

    # ГРАФІК 1: Спектр амплітуд та фаз (Частина І)
    fig1, axes1 = plt.subplots(2, 1, figsize=(8, 6))

    axes1[0].stem(k_indices1, amplitudes_p1, linefmt='b-', markerfmt='bo', basefmt='r-')
    axes1[0].set_title('Спектр амплітуд |C_k| (Частина І)')
    axes1[0].set_xlabel('k')
    axes1[0].set_ylabel('|C_k|')
    axes1[0].grid(True)

    axes1[1].stem(k_indices1, phases_p1, linefmt='g-', markerfmt='go', basefmt='r-')
    axes1[1].set_title('Спектр фаз arg(C_k) [рад] (Частина І)')
    axes1[1].set_xlabel('k')
    axes1[1].set_ylabel('Arg(C_k)')
    axes1[1].grid(True)

    plt.tight_layout()
    plt.show()

    # --- ЧАСТИНА ІІ ---
    bits, num_dec = generate_part2_samples(n)
    N2 = 8
    bits_arr = np.array(bits, dtype=float)
    C_part2, _, _, _ = calculate_dft(bits_arr, N2)

    amp_p2 = np.abs(C_part2)
    arg_p2 = clean_phase(C_part2)

    print('\n--- Частина ІІ ---')
    print(f'Число N = 96 + {n} = {num_dec} = {num_dec:08b}(2)')
    print(f'Згенеровані 8 відліків s(nT_d): {bits}')
    print('Комплексні коефіцієнти C_k, їх модулі та аргументи:')
    for idx in range(N2):
        c_show = complex(round(C_part2[idx].real, 4) + 0.0,
                         round(C_part2[idx].imag, 4) + 0.0)
        print(
            f'k = {idx}: C_{idx} = {c_show:.4f}, '
            f'|C_{idx}| = {amp_p2[idx]:.4f}, arg(C_{idx}) = {arg_p2[idx]:.4f} рад'
        )

    print_signal_expression(C_part2)
    print_signal_expression_complex(C_part2)

    # ГРАФІК 2: Відтворення аналогового сигналу (Частина ІІ)
    T_period = 1.0
    t_continuous = np.linspace(0, T_period, 500)
    s_continuous = reconstruct_analog_signal(C_part2, t_continuous, T_period)

    plt.figure(figsize=(8, 4))
    t_discrete = np.linspace(0, T_period, N2, endpoint=False)
    plt.plot(t_continuous, s_continuous.real,
             label='Відтворений аналоговий сигнал s(t)', color='blue')
    plt.stem(t_discrete, bits, linefmt='r--', markerfmt='ro',
             label='Дискретні відліки s(nT_d)')
    plt.title('Відтворення аналогового сигналу s(t) (Частина ІІ)')
    plt.xlabel('Час t')
    plt.ylabel('s(t)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- ПОРІВНЯННЯ ДПФ ТА ШПФ ---
    N_compare = 16
    np.random.seed(42)
    f_comp = np.random.uniform(-5, 5, N_compare)
    if USE_COMPLEX:
        f_comp = f_comp + 1j * np.random.uniform(-5, 5, N_compare)

    C_dft_comp, _, _, _ = calculate_dft(f_comp, N_compare)
    C_fft_comp = np.fft.fft(f_comp) / N_compare  # Нормування 1/N для збігу

    amp_dft = np.abs(C_dft_comp)
    amp_fft = np.abs(C_fft_comp)
    phase_dft = clean_phase(C_dft_comp)
    phase_fft = clean_phase(C_fft_comp)

    k_comp = np.arange(N_compare)

    # ГРАФІК 3: Порівняння ДПФ і ШПФ
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

    ax1.stem(k_comp, amp_dft, linefmt='b-', markerfmt='bo', label='ДПФ', basefmt=' ')
    ax1.stem(k_comp, amp_fft, linefmt='r--', markerfmt='rx', label='ШПФ', basefmt=' ')
    ax1.set_title('Порівняння амплітудного спектра |C_k| (ДПФ vs ШПФ)')
    ax1.set_xlabel('k')
    ax1.set_ylabel('|C_k|')
    ax1.legend()
    ax1.grid(True)

    ax2.stem(k_comp, phase_dft, linefmt='b-', markerfmt='bo', label='ДПФ', basefmt=' ')
    ax2.stem(k_comp, phase_fft, linefmt='r--', markerfmt='rx', label='ШПФ', basefmt=' ')
    ax2.set_title('Порівняння фазового спектра Arg(C_k) (ДПФ vs ШПФ)')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Радіани')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # --- ЧАСТИНА ІІІ ---
    s_reconstructed = calculate_idft(C_part2, N2)

    print('\n--- Частина ІІІ ---')
    print('Відновлені відліки через ОДПФ:')
    for idx in range(N2):
        print(
            f'n = {idx}: s({idx}T_d) = {s_reconstructed[idx].real:.4f} '
            f'(початкове значення: {bits[idx]})'
        )

    print_idft_expressions(C_part2, s_reconstructed, bits)

    # ОДПФ для вектора з Частини І (перевірка повного циклу)
    f_back = calculate_idft(C_part1, N1)
    print(f'\nОДПФ(ДПФ(f)) збігається з вхідним f (Частина І): '
          f'{np.allclose(f_back, f_input)}')