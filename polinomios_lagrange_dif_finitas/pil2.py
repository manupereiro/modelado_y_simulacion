import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp

def lagrange_poly(x_points, y_points):
    """Devuelve el polinomio interpolante de Lagrange como np.poly1d"""
    n = len(x_points)
    P_coeff = np.zeros(n)
    for i in range(n):
        denom = np.prod([x_points[i] - x_points[j] for j in range(n) if j!=i])
        others = [x_points[j] for j in range(n) if j!=i]
        numer_coeff = np.poly(others)
        P_coeff = np.pad(P_coeff, (len(numer_coeff)-len(P_coeff),0), 'constant') \
                  + (y_points[i]/denom) * numer_coeff
    P_coeff = P_coeff[-n:]
    return np.poly1d(P_coeff)

def max_derivative_symbolic(f_expr, x_sym, order, a, b):
    """
    Calcula f^{(order)}(x) con sympy, luego lo evalúa en a y b,
    y devuelve M = max(|f^{(order)}(a)|, |f^{(order)}(b)|).
    """
    # Derivada simbólica de orden `order`
    f_k_expr = sp.diff(f_expr, x_sym, order)
    # Convertir a función numérica
    f_k_num = sp.lambdify(x_sym, f_k_expr, 'numpy')
    # Evaluar en los extremos
    val_a = abs(f_k_num(a))
    val_b = abs(f_k_num(b))
    M = max(val_a, val_b)
    return float(M), f_k_expr

def error_local_lagrange(x, x_points, M_np1, grado=None):
    """
    Cota de error local |f(x)-P(x)| ≤ M_{n+1}/(n+1)! * ∏_{i=0}^n |x - x_i|.
    """
    if grado is None:
        grado = len(x_points)-1
    producto = np.prod([x - xi for xi in x_points])
    return abs(M_np1 * producto / math.factorial(grado+1))

if __name__ == "__main__":
    # — Datos de ejemplo con f(x)=ln(x+1) y puntos [0,1,2]
    x_points = np.array([0, 1, 2], dtype=float)
    y_points = np.array([0, np.log(2), np.log(3)], dtype=float)

    # Definimos simbólicamente f
    x = sp.symbols('x')
    f_expr = sp.log(x + 1)

    # Construir polinomio de interpolación
    P = lagrange_poly(x_points, y_points)
    print("Polinomio de Lagrange:\n", P, "\n")

    # Grado del polinomio y orden de derivada para la cota
    grado = len(x_points) - 1
    orden = grado + 1

    # Calculamos M_{n+1} evaluando f^{(n+1)} en los extremos
    a, b = float(x_points[0]), float(x_points[-1])
    M_np1, f_k_expr = max_derivative_symbolic(f_expr, x, orden, a, b)
    print(f"Derivada de orden {orden}:")
    print(f_k_expr, "\n")
    print(f"M_{orden} = max(|f^{orden}({a})|, |f^{orden}({b})|) = {M_np1:.6}\n")

    # Elegimos un punto para evaluar la cota de error local
    x_eval = 0.5
    P_val = P(x_eval)
    err_cota = error_local_lagrange(x_eval, x_points, M_np1, grado=grado)

    print(f"P({x_eval}) = {P_val:.6f}")
    print(f"Cota del error local en x={x_eval}: ≲ {err_cota:.6}\n")

    # Graficar interpolante y el punto de evaluación
    xs = np.linspace(a - 0.5, b + 0.5, 400)
    plt.figure(figsize=(8,5))
    plt.scatter(x_points, y_points, color='red', label='nodos')
    plt.plot(xs, P(xs), label='P(x)')
    plt.axvline(x_eval, color='gray', linestyle='--', label=f'x={x_eval}')
    plt.title("Interpolación de Lagrange + cota de error local")
    plt.xlabel("x"); plt.ylabel("P(x)")
    plt.legend(); plt.grid(True)
    plt.show()
