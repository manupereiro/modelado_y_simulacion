import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import math

# Función para calcular la derivada numérica usando diferencias centrales con NumPy
def numpy_derivative(f, x, dx=1e-6):
    """Calcula la derivada numérica de una función en un punto usando diferencias centrales."""
    x_points = np.array([x - dx, x, x + dx])
    y_points = np.array([f(xi) for xi in x_points])
    derivative = np.gradient(y_points, x_points)[1]
    return derivative

# Método de Newton-Raphson utilizando el error relativo
def newton_raphson(f, valor_inicial, iteraciones=100, tolerancia=1e-3, precision=10):
    """
    Aplica el método de Newton-Raphson para hallar la raíz de una función usando error relativo.
    
    Parámetros:
    -----------
    - f: Función de la cual se desea encontrar la raíz.
    - valor_inicial: Aproximación inicial.
    - iteraciones: Número máximo de iteraciones.
    - tolerancia: Tolerancia del error relativo para detener la iteración.
    - precision: Cantidad de dígitos a mostrar (formato, evita notación científica).
    
    Retorna:
    --------
    - x: La aproximación final de la raíz.
    - results: La tabla de resultados de cada iteración.
    """
    x = valor_inicial
    results = []
    
    for i in range(iteraciones):
        fx = round(f(x), precision)
        dfx = round(numpy_derivative(f, x, dx=tolerancia), precision)
        if dfx == 0:
            raise ValueError("La derivada es cero. El método no puede continuar.")
        
        x_new = round(x - fx / dfx, precision)
        # Calcular error relativo: si x_new != 0 usamos la fórmula, sino usamos error absoluto.
        if x_new != 0:
            error_rel = abs(x_new - x) / abs(x_new)
        else:
            error_rel = abs(x_new - x)
        
        # Formatear cada valor para la tabla
        fila = [
            str(i + 1),
            f"{x:.{precision}f}",
            f"{fx:.{precision}f}",
            f"{dfx:.{precision}f}",
            f"{x_new:.{precision}f}",
            f"{error_rel:.{precision}f}"
        ]
        results.append(fila)
        
        if error_rel < tolerancia:
            x = x_new
            break
        x = x_new
    
    # Imprimir la tabla de resultados una sola vez, al finalizar
    headers = ["Iteración", "x", "f(x)", "f'(x)", "Resultado", "Error Relativo"]
    print(tabulate(results, headers=headers, tablefmt="grid", floatfmt=f".{precision}f"))
    
    return x

def graficar(f, raiz, precision=10):
    # Generar puntos para graficar la función
    x = np.linspace(0, 3, 100)
    y = f(x)

    plt.plot(x, y, label='$f(x)$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # Marcar la raíz encontrada
    plt.plot(raiz, f(raiz), 'ro', label=f'Raíz: x = {raiz:.{precision}f}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gráfica de la función y su raíz')
    plt.show()

# Definir la función de la cual se busca la raíz
def f(x):
    return (x - 1)**2

# Valor inicial definido directamente
valor_inicial = 0

# Parámetros: tolerancia y precisión (modifica estos valores según lo necesites)
tolerancia = 1e-3    # Tolerancia del error relativo
precision = 10       # Cantidad de dígitos a mostrar

# Ejecutar el método de Newton-Raphson con error relativo
raiz = newton_raphson(f, valor_inicial, iteraciones=100, tolerancia=tolerancia, precision=precision)

print(f"\nLa raíz encontrada es: {raiz:.{precision}f}")

# Graficar la función y la raíz encontrada
graficar(f, raiz, precision=precision)