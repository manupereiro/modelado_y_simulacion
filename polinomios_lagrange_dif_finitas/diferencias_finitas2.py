import numpy as np
import pandas as pd
import sympy as sp

def calcular_derivadas_finitas_comparativas(f_simbolica, x_vals, h):
    """
    Calcula derivadas numéricas (progresiva, centrada, regresiva) para
    la primera y segunda derivada, y las compara con las derivadas analíticas.
    """
    # Símbolos y derivadas simbólicas
    x = sp.symbols('x')
    f1_sim = sp.diff(f_simbolica, x)
    f2_sim = sp.diff(f1_sim, x)

    # Funciones numéricas
    f        = sp.lambdify(x, f_simbolica, modules='numpy')
    f1_exact = sp.lambdify(x, f1_sim, modules='numpy')
    f2_exact = sp.lambdify(x, f2_sim, modules='numpy')

    data = []
    n = len(x_vals)

    for i, xi in enumerate(x_vals):
        # Fórmulas según posición
        if i == 0:
            # Progresivas
            f1_aprox = (f(xi + h) - f(xi)) / h
            f2_aprox = (f(xi + 2*h) - 2*f(xi + h) + f(xi)) / (h**2)
        elif i == n - 1:
            # Regresivas
            f1_aprox = (f(xi) - f(xi - h)) / h
            f2_aprox = (f(xi) - 2*f(xi - h) + f(xi - 2*h)) / (h**2)
        else:
            # Centrales
            f1_aprox = (f(xi + h) - f(xi - h)) / (2*h)
            f2_aprox = (f(xi + h) - 2*f(xi) + f(xi - h)) / (h**2)

        # Exactas y errores
        f1_real = f1_exact(xi)
        f2_real = f2_exact(xi)
        err1    = abs(f1_aprox - f1_real)
        err2    = abs(f2_aprox - f2_real)

        data.append({
            'x': xi,
            "f'(aprox)": f1_aprox,
            "f'(exacta)": f1_real,
            "Error f'": err1,
            "f''(aprox)": f2_aprox,
            "f''(exacta)": f2_real,
            "Error f''": err2
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Ejemplo de uso: f(x) = sin(x)
    x = sp.symbols('x')
    f_expr = sp.log(x + 1)
    #x_vals = [0, 1.5, 4, 7.5, 12, 17.5, 24, 31.5, 40]
    x_vals = np.arange(1, 1.5, 2)
    h = 0.5

    df = calcular_derivadas_finitas_comparativas(f_expr, x_vals, h)
    print(df.to_string(index=False))
