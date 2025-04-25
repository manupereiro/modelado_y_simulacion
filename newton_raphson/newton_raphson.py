import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import math

# Función para calcular la derivada numérica usando NumPy con diferencias centrales
def numpy_derivative(f, x, dx=1e-6):
    """Calcula la derivada numérica de una función en un punto usando diferencias centrales"""
    x_points = np.array([x - dx, x, x + dx])
    y_points = np.array([f(xi) for xi in x_points])
    derivative = np.gradient(y_points, x_points)[1]
    return derivative

# Método de Newton-Raphson (modificado para imprimir la tabla una sola vez)
def newton_raphson(f, valor_inicial, iteraciones=100, tolerancia=1e-3, precision=10):
    """
    Aplica el método de Newton-Raphson para hallar la raíz de una función.
    
    Parámetros:
    -----------
    - f: Función de la cual se desea encontrar la raíz.
    - valor_inicial: Aproximación inicial.
    - iteraciones: Número máximo de iteraciones.
    - tolerancia: Criterio de tolerancia para detener la iteración.
    - precision: Cantidad de dígitos a mostrar (evita notación científica).
    
    Retorna:
    --------
    - x: La aproximación final de la raíz.
    - results: Tabla con los resultados de cada iteración.
    """
    x = valor_inicial
    results = []
    
    for i in range(iteraciones):
        fx = round(f(x), precision)
        dfx = round(numpy_derivative(f, x, dx=tolerancia), precision)
        if dfx == 0:
            raise ValueError("La derivada es cero. El método no puede continuar.")
        
        x_new = round(x - fx / dfx, precision)
        results.append([i + 1, f"{x:.{precision}f}", f"{fx:.{precision}f}", f"{dfx:.{precision}f}", f"{x_new:.{precision}f}"])
        
        if abs(x_new - x) < tolerancia:
            x = x_new
            break
        x = x_new
        
    # Imprimir la tabla completa una vez finalizado el proceso
    headers = ["Iteración", "x", "f(x)", "f'(x)", "Resultado"]
    print(tabulate(results, headers=headers, tablefmt="grid", floatfmt=f".{precision}f"))
    
    return x

def graficar(f, raiz, precision=10):
    # Generar valores para graficar la función
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

# Parámetros: tolerancia y precisión
tolerancia = 1e-3      # Puedes cambiar la tolerancia aquí
precision = 10         # Puedes cambiar la cantidad de dígitos de precisión aquí

# Encontrar la raíz usando el método de Newton-Raphson
raiz = newton_raphson(f, valor_inicial, iteraciones=100, tolerancia=tolerancia, precision=precision)

print(f"\nLa raíz encontrada es: {raiz:.{precision}f}")

# Graficar la función y la raíz encontrada
graficar(f, raiz, precision=precision)
