import numpy as np


def fractional_derivative(signal, alpha=0.25, max_history=None):
    n = len(signal)
    y = np.zeros(n)

    for i in range(n):
        dc = 0.0
        k_max = i + 1 if max_history is None else min(i + 1, max_history)
        binom = 1.0
        for k in range(k_max):
            if k > 0:
                binom *= (alpha - (k - 1)) / k
            sign = (-1) ** k
            dc += sign * binom * signal[i - k]
        y[i] = dc
    return y


if __name__ == "__main__":
    print("Sample test")
    x = np.array([0, 1, 2, 4, 7, 11, 16], dtype=float)
    D_full = np.diff(x, prepend=x[0])  # = 1

    denom = 42
    alpha = 1 / denom
    todo = denom
    D_frac = x
    for _ in range(todo):
        D_frac = fractional_derivative(D_frac, alpha)

    np.set_printoptions(precision=5, suppress=True)
    print("Original signal:", x)
    print("Normal derivative: ", D_full)
    print(f"(1/{denom})-th derivative {todo}-times:", D_frac)
