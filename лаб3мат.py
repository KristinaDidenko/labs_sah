import math
import time

import matplotlib.pyplot as plt
import numpy as np

n = 2  # Номер варіанта
EPS = 1e-10  # поріг, нижче якого коефіцієнт вважається нульовим


# ДОПОМІЖНІ ФУНКЦІЇ

def is_power_of_two(N):
    return N > 0 and (N & (N - 1)) == 0


def next_power_of_two(N):
    return 1 << (N - 1).bit_length()


def clean_phase(C):
    """Фаза arg(C_k): для майже нульових C_k фаза = 0,
    значення -pi приводиться до +pi (єдиний вигляд графіків)."""
    phase = np.angle(np.asarray(C))
    phase[np.abs(C) < EPS] = 0.0
    phase[np.isclose(phase, -np.pi)] = np.pi
    return phase


def generate_part2_samples(n_variant):
    """8 відліків Частини ІІ ЛР№2: N = 96 + n у двійковому вигляді."""
    num_dec = 96 + n_variant
    bin_str = list(format(num_dec, '08b'))
    bin_str[0] = '1' if n_variant % 2 != 0 else '0'
    return [int(b) for b in bin_str], num_dec


def measure_time(func, arg, repeats):
    """Середній час одного виклику func(arg) за `repeats` запусків (с).
    Підрахунок операцій у цей час не входить (він виконується окремо)."""
    func(arg)  # прогрів
    start = time.perf_counter()
    for _ in range(repeats):
        func(arg)
    return (time.perf_counter() - start) / repeats


# 1. ДПФ (з Лабораторної роботи №2)

def calculate_dft(f):
    """ДПФ за означенням: C_k = (1/N) * Sum f[n]*(cos a - j*sin a), a = 2*pi*k*n/N.

    Дійсний f:     2 множення + 2 додавання на член;
    комплексний f: 4 множення + 4 додавання на член.
    Плюс 2 множення на нормування 1/N для кожного k.
    Виклики cos/sin у підрахунок не входять.

    Повертає (C, к-сть множень, к-сть додавань).
    """
    N = len(f)
    is_complex = np.iscomplexobj(f)
    x = list(f)
    C = []
    for k in range(N):
        re = 0.0
        im = 0.0
        for i in range(N):
            angle = 2 * math.pi * k * i / N
            c = math.cos(angle)
            s = math.sin(angle)
            if is_complex:
                a, b = x[i].real, x[i].imag
                re += a * c + b * s
                im += b * c - a * s
            else:
                re += x[i] * c
                im -= x[i] * s
        C.append(complex(re / N, im / N))

    per_term_mult = 4 if is_complex else 2
    per_term_add = 4 if is_complex else 2
    mult = N * (per_term_mult * N + 2)
    add = N * per_term_add * N
    return np.array(C), mult, add


# 2. ШПФ (Кулі-Тьюкі, основа 2, прорідження за часом)

def fft_raw(x):
    """Ітеративне ШПФ без нормування: X_k = Sum x[n] * exp(-j*2*pi*k*n/N).

    1) бітово-інверсна перестановка вхідного масиву;
    2) log2(N) етапів «метеликів»:
           u = a[i],  v = a[i + half] * W,
           a[i] = u + v,  a[i + half] = u - v.
    Один «метелик» = комплексне множення (4 множ. + 2 дод.)
                     + 2 комплексних додавання (4 дод.)  ->  4 множ. + 6 дод.
    Кількість «метеликів»: (N/2) * log2(N).
    Таблиця поворотних множників W_N^k обчислюється один раз (входить у час,
    у кількість арифметичних операцій — ні).

    Повертає (X, к-сть множень, к-сть додавань).
    """
    N = len(x)
    if not is_power_of_two(N):
        raise ValueError('N повинно бути степенем двійки (8, 16, 32, ...)')
    M = N.bit_length() - 1  # M = log2(N)

    a = [complex(v) for v in x]

    # бітово-інверсна перестановка
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # таблиця W_N^k = exp(-j*2*pi*k/N), k = 0..N/2-1
    W = [complex(math.cos(2 * math.pi * k / N), -math.sin(2 * math.pi * k / N))
         for k in range(N // 2)]

    size = 2
    while size <= N:
        half = size // 2
        step = N // size
        for start in range(0, N, size):
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * W[k * step]
                a[start + k] = u + v
                a[start + k + half] = u - v
        size *= 2

    butterflies = (N // 2) * M
    return a, 4 * butterflies, 6 * butterflies


def calculate_fft(f):
    """ШПФ з нормуванням 1/N (щоб збігалося з ДПФ із ЛР№2).
    Нормування: N комплексних значень * 2 дійсні множення = 2N множень.

    Повертає (C, к-сть множень, к-сть додавань).
    """
    N = len(f)
    X, mult, add = fft_raw(f)
    C = np.array([v / N for v in X])
    return C, mult + 2 * N, add


def calculate_ifft(C):
    """Обернене ШПФ: s(n) = Sum C_k * exp(+j*2*pi*k*n/N)
    (множник 1/N уже враховано в прямому перетворенні).
    Використовується тотожність: s = conj( FFT( conj(C) ) ).

    Повертає (s, к-сть множень, к-сть додавань).
    """
    X, mult, add = fft_raw(np.conj(C))
    return np.conj(np.array(X)), mult, add


# Теоретичні формули кількості операцій

def theory_dft(N):
    """ДПФ дійсного сигналу (як у ЛР№2): (множення, додавання)."""
    return N * (2 * N + 2), 2 * N * N


def theory_fft(N):
    """ШПФ: 2*N*log2(N) + 2N множень, 3*N*log2(N) додавань."""
    M = int(round(math.log2(N)))
    return 2 * N * M + 2 * N, 3 * N * M


def choose_repeats(N, fast):
    """Кількість повторів для усереднення часу."""
    base = min(200, max(1, 400000 // (N * N)))
    return base * 20 if fast else base


# ГОЛОВНА ПРОГРАМА

if __name__ == '__main__':
    print(f'=== ЛАБОРАТОРНА РОБОТА №3 (Варіант n = {n}) ===')
    print('Швидке перетворення Фур\'є (ШПФ, алгоритм Кулі-Тьюкі)\n')

    # Вхідні дані з ЛР№2

    # Частина ІІ: 8 відліків (двійкове представлення 96 + n)
    bits, num_dec = generate_part2_samples(n)
    f_part2 = np.array(bits, dtype=float)

    # Частина І: N = 10 + n = 12 (не степінь двійки -> доповнення нулями до 16)
    N1 = 10 + n
    np.random.seed(42)
    f_part1 = np.random.uniform(-5, 5, N1)
    N1_pad = next_power_of_two(N1)
    f_part1_pad = np.concatenate([f_part1, np.zeros(N1_pad - N1)])

    # Порівняння ДПФ/ШПФ з ЛР№2: N = 16, seed 42
    np.random.seed(42)
    f_cmp = np.random.uniform(-5, 5, 16)

    cases = [
        (f'Частина ІІ ЛР№2 (N = {len(f_part2)})', f_part2),
        (f'Частина І ЛР№2, N = {N1} -> доповнено нулями до {N1_pad}', f_part1_pad),
        ('Порівняння ЛР№2 (N = 16, seed 42)', f_cmp),
    ]

    # 1. Порівняння ДПФ та ШПФ на даних ЛР№2

    print('=' * 80)
    print('1. ПОРІВНЯННЯ ДПФ ТА ШПФ НА ДАНИХ ЛР№2')
    print('=' * 80)

    results_cases = []
    for name, f in cases:
        N = len(f)
        C_dft, m_dft, a_dft = calculate_dft(f)
        C_fft, m_fft, a_fft = calculate_fft(f)

        t_dft = measure_time(calculate_dft, f, choose_repeats(N, False))
        t_fft = measure_time(calculate_fft, f, choose_repeats(N, True))

        ok_dft_fft = np.allclose(C_dft, C_fft)
        ok_numpy = np.allclose(C_fft, np.fft.fft(f) / N)
        max_err = np.max(np.abs(C_dft - C_fft))

        print(f'\n--- {name} ---')
        print(f'Вхідний вектор: {np.round(f, 3)}')
        print(f'{"Показник":<28} | {"ДПФ":>14} | {"ШПФ":>14}')
        print('-' * 62)
        print(f'{"Час обчислення (мс)":<28} | {t_dft * 1000:>14.4f} | {t_fft * 1000:>14.4f}')
        print(f'{"Кількість множень":<28} | {m_dft:>14} | {m_fft:>14}')
        print(f'{"Кількість додавань":<28} | {a_dft:>14} | {a_fft:>14}')
        print(f'{"Всього операцій":<28} | {m_dft + a_dft:>14} | {m_fft + a_fft:>14}')
        print('-' * 62)
        print(f'Прискорення за операціями: у {(m_dft + a_dft) / (m_fft + a_fft):.2f} раза')
        print(f'Прискорення за часом:      у {t_dft / t_fft:.2f} раза')
        print(f'ДПФ == ШПФ: {ok_dft_fft} (макс. різниця {max_err:.2e});  '
              f'ШПФ == np.fft.fft/N: {ok_numpy}')

        # перевірка лічильників за теоретичними формулами
        assert (m_dft, a_dft) == theory_dft(N), 'лічильник ДПФ не збігається з формулою'
        assert (m_fft, a_fft) == theory_fft(N), 'лічильник ШПФ не збігається з формулою'

        results_cases.append((name, f, C_dft, C_fft))

    # Графіки спектрів (ДПФ vs ШПФ) для N = 16 (дані ЛР№2)
    _, f16, C_dft16, C_fft16 = results_cases[2]
    k16 = np.arange(16)
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
    PASTEL = '#8fd19e'  # пастельний зелений для ДПФ
    m, s, _ = ax1.stem(k16, np.abs(C_dft16), linefmt='-', markerfmt='o',
                       label='ДПФ', basefmt=' ')
    plt.setp(m, color=PASTEL, markersize=13)
    plt.setp(s, color=PASTEL, linewidth=5)
    m, s, _ = ax1.stem(k16, np.abs(C_fft16), linefmt='--', markerfmt='x',
                       label='ШПФ', basefmt=' ')
    plt.setp(m, color='red', markersize=8, markeredgewidth=2)
    plt.setp(s, color='red', linewidth=1.2)
    ax1.set_title('Амплітудний спектр |C_k| (ДПФ vs ШПФ), N = 16')
    ax1.set_xlabel('k')
    ax1.set_ylabel('|C_k|')
    ax1.legend()
    ax1.grid(True)
    m, s, _ = ax2.stem(k16, clean_phase(C_dft16), linefmt='-', markerfmt='o',
                       label='ДПФ', basefmt=' ')
    plt.setp(m, color=PASTEL, markersize=13)
    plt.setp(s, color=PASTEL, linewidth=5)
    m, s, _ = ax2.stem(k16, clean_phase(C_fft16), linefmt='--', markerfmt='x',
                       label='ШПФ', basefmt=' ')
    plt.setp(m, color='red', markersize=8, markeredgewidth=2)
    plt.setp(s, color='red', linewidth=1.2)
    ax2.set_title('Фазовий спектр arg(C_k) (ДПФ vs ШПФ), N = 16')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Радіани')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    fig1.savefig('lab3_spectra_compare.png', dpi=150)

    # 2. Частина ІІ ЛР№2, обчислена через ШПФ + обернене ШПФ

    print('\n' + '=' * 80)
    print('2. ШПФ ТА ОБЕРНЕНЕ ШПФ ДЛЯ ВІДЛІКІВ ЧАСТИНИ ІІ ЛР№2')
    print('=' * 80)
    N2 = len(f_part2)
    C2, m2, a2 = calculate_fft(f_part2)
    print(f'Число N = 96 + {n} = {num_dec} = {num_dec:08b}(2)')
    print(f'Відліки s(nT_d): {bits}')
    print(f'ШПФ: {m2} множень, {a2} додавань')
    print('Коефіцієнти C_k, їх модулі та аргументи:')
    ph2 = clean_phase(C2)
    for k in range(N2):
        c_show = complex(round(C2[k].real, 4) + 0.0, round(C2[k].imag, 4) + 0.0)
        print(f'  k = {k}: C_{k} = {c_show:.4f}, |C_{k}| = {abs(C2[k]):.4f}, '
              f'arg(C_{k}) = {ph2[k]:.4f} рад')

    s_rec, m_i, a_i = calculate_ifft(C2)
    print(f'\nОбернене ШПФ: {m_i} множень, {a_i} додавань')
    for i in range(N2):
        print(f'  n = {i}: s({i}T_d) = {s_rec[i].real:.4f}  (початкове значення: {bits[i]})')
    print(f'Відновлені відліки збігаються з початковими: '
          f'{np.allclose(s_rec.real, f_part2) and np.allclose(s_rec.imag, 0)}')

    fig2, axes2 = plt.subplots(3, 1, figsize=(8, 8))
    k8 = np.arange(N2)
    axes2[0].stem(k8, np.abs(C2), linefmt='b-', markerfmt='bo', basefmt='r-')
    axes2[0].set_title('Спектр амплітуд |C_k| (ШПФ, Частина ІІ)')
    axes2[0].set_xlabel('k')
    axes2[0].grid(True)
    axes2[1].stem(k8, ph2, linefmt='g-', markerfmt='go', basefmt='r-')
    axes2[1].set_title('Спектр фаз arg(C_k) [рад] (ШПФ, Частина ІІ)')
    axes2[1].set_xlabel('k')
    axes2[1].grid(True)
    axes2[2].stem(k8 - 0.08, f_part2, linefmt='r-', markerfmt='ro', basefmt=' ',
                  label='початкові s(nT_d)')
    axes2[2].stem(k8 + 0.08, s_rec.real, linefmt='b--', markerfmt='bx', basefmt=' ',
                  label='обернене ШПФ')
    axes2[2].set_title('Відновлення відліків оберненим ШПФ')
    axes2[2].set_xlabel('n')
    axes2[2].legend()
    axes2[2].grid(True)
    plt.tight_layout()
    fig2.savefig('lab3_part2_fft.png', dpi=150)

    # 3. Залежність часу й кількості операцій від N

    print('\n' + '=' * 80)
    print('3. ЗАЛЕЖНІСТЬ ЧАСУ ТА КІЛЬКОСТІ ОПЕРАЦІЙ ВІД N')
    print('=' * 80)
    header = (f'{"N":>5} | {"Оп. ДПФ":>9} | {"Оп. ШПФ":>8} | {"Приск. оп.":>10} | '
              f'{"t ДПФ, мс":>10} | {"t ШПФ, мс":>10} | {"Приск. t":>8} | {"збіг":>5}')
    print(header)
    print('-' * len(header))

    Ns, ops_d, ops_f, t_d, t_f = [], [], [], [], []
    for p in range(3, 11):  # N = 8 ... 1024 (для швидшого запуску: range(3, 10))
        N = 2 ** p
        np.random.seed(42)
        f = np.random.uniform(-5, 5, N)
        Cd, md, ad = calculate_dft(f)
        Cf, mf, af = calculate_fft(f)
        td = measure_time(calculate_dft, f, choose_repeats(N, False))
        tf = measure_time(calculate_fft, f, choose_repeats(N, True))
        match = np.allclose(Cd, Cf)
        Ns.append(N)
        ops_d.append(md + ad)
        ops_f.append(mf + af)
        t_d.append(td)
        t_f.append(tf)
        print(f'{N:>5} | {md + ad:>9} | {mf + af:>8} | {(md + ad) / (mf + af):>9.2f}x | '
              f'{td * 1000:>10.4f} | {tf * 1000:>10.4f} | {td / tf:>7.2f}x | {str(match):>5}')

    Ns = np.array(Ns)
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
    axes3[0].loglog(Ns, np.array(t_d) * 1000, 'bo-', label='ДПФ')
    axes3[0].loglog(Ns, np.array(t_f) * 1000, 'rs-', label='ШПФ')
    axes3[0].set_title('Час обчислення')
    axes3[0].set_xlabel('N')
    axes3[0].set_ylabel('мс')
    axes3[0].legend()
    axes3[0].grid(True, which='both')
    axes3[1].loglog(Ns, ops_d, 'bo-', label='ДПФ ~ N²')
    axes3[1].loglog(Ns, ops_f, 'rs-', label='ШПФ ~ N·log₂N')
    axes3[1].set_title('Кількість операцій (множення + додавання)')
    axes3[1].set_xlabel('N')
    axes3[1].legend()
    axes3[1].grid(True, which='both')
    axes3[2].semilogx(Ns, np.array(ops_d) / np.array(ops_f), 'go-', label='за операціями')
    axes3[2].semilogx(Ns, np.array(t_d) / np.array(t_f), 'm^-', label='за часом')
    axes3[2].set_title('Прискорення ШПФ відносно ДПФ')
    axes3[2].set_xlabel('N')
    axes3[2].set_ylabel('разів')
    axes3[2].legend()
    axes3[2].grid(True, which='both')
    plt.tight_layout()
    fig3.savefig('lab3_complexity.png', dpi=150)

    # 4. Висновок (числа й твердження підставляються з виконаних обчислень)

    speed_ops = np.array(ops_d) / np.array(ops_f)
    speed_time = np.array(t_d) / np.array(t_f)
    faster_idx = [i for i, s in enumerate(speed_time) if s > 1]
    first_faster = int(Ns[faster_idx[0]]) if faster_idx else None
    N16 = int(np.where(Ns == 16)[0][0])

    if first_faster:
        time_line = f'ШПФ швидший за ДПФ починаючи з N = {first_faster}.'
    else:
        time_line = 'ШПФ не виявився швидшим на заданих N.'

    # твердження про співвідношення виграшів залежить від реальних результатів
    if speed_time[-1] > speed_ops[-1]:
        time_vs_ops = (
            f'Виграш за часом ({speed_time[-1]:.1f}x) більший за виграш за операціями\n'
            f'   ({speed_ops[-1]:.1f}x), бо ДПФ N² разів викликає cos/sin (вони не входять\n'
            f'   у підрахунок операцій), а ШПФ обчислює таблицю W лише один раз\n'
            f'   (N/2 викликів).')
    else:
        time_vs_ops = (
            f'Виграш за часом ({speed_time[-1]:.1f}x) не перевищує виграш за операціями\n'
            f'   ({speed_ops[-1]:.1f}x): накладні витрати інтерпретатора Python\n'
            f'   (цикли, роботу зі списками та комплексними числами) у ШПФ помітніші.')

    print('\n' + '=' * 80)
    print('ВИСНОВОК')
    print('=' * 80)
    print(f"""1. Реалізовано швидке перетворення Фур'є за алгоритмом Кулі-Тьюкі (основа 2,
   ітеративна схема з бітово-інверсною перестановкою). На всіх даних ЛР№2
   (Частина ІІ, N = 8; Частина І, доповнена нулями до N = 16; вектор N = 16
   із порівняння) спектри ШПФ збігаються зі спектрами ДПФ і з np.fft.fft/N.
   Обернене ШПФ точно відновлює початкові відліки s(nT_d).

2. Кількість операцій (множення + додавання):
   ДПФ: N(2N + 2) множень і 2N² додавань, складність O(N²).
   ШПФ: 2N·log2(N) + 2N множень і 3N·log2(N) додавань, складність O(N·log2 N).
   Для N = 16: ДПФ — {ops_d[N16]} операцій, ШПФ — {ops_f[N16]}, тобто у {speed_ops[N16]:.2f} раза менше.
   Для N = {Ns[-1]}: ДПФ — {ops_d[-1]}, ШПФ — {ops_f[-1]}, тобто у {speed_ops[-1]:.1f} раза менше.
   Виграш зростає з N, оскільки відношення операцій ~ N / log2(N).

3. Час обчислення (чистий Python):
   {time_line}
   Для N = {Ns[-1]} ШПФ швидший у {speed_time[-1]:.1f} раза.
   {time_vs_ops}
   Крім того, ДПФ із ЛР№2 рахує дійсний сигнал (2 множення на член), а ШПФ —
   комплексний, тобто порівняння для ШПФ «невигідне».

4. Обмеження: радикс-2 потребує N = 2^M. Вектор Частини І (N = {N1}) довелося
   доповнити нулями до {N1_pad}; спектр доповненого сигналу (більше точок на
   тій самій обвідній) відрізняється від ДПФ при N = {N1}.
   У підрахунок не входять виклики cos/sin (ДПФ) та побудова таблиці W (ШПФ).
""")

    plt.show()