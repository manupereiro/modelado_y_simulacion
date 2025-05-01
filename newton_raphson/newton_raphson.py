import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Importar pandas
import math

# Función para calcular la derivada numérica usando NumPy con diferencias centrales
def numpy_derivative(f, x, dx=1e-6):
    """Calcula la derivada numérica de una función en un punto usando diferencias centrales"""
    x_eval = x 
    x_points = np.array([x_eval - dx, x_eval, x_eval + dx])
    try:
        y_points = f(x_points)
    except TypeError:
        y_points = np.array([f(xi) for xi in x_points])
    derivative = np.gradient(y_points, dx)[1] 
    return derivative

# Método de Newton-Raphson (modificado para usar pandas)
def newton_raphson(f, valor_inicial, iteraciones=100, tolerancia=1e-3, precision=10):
    """
    Aplica el método de Newton-Raphson para hallar la raíz de una función.
    Muestra los resultados de las iteraciones usando un DataFrame de pandas.
    
    Parámetros:
    -----------
    - f: Función de la cual se desea encontrar la raíz.
    - valor_inicial: Aproximación inicial.
    - iteraciones: Número máximo de iteraciones.
    - tolerancia: Criterio de tolerancia (error absoluto |x_new - x|) para detener la iteración.
    - precision: Cantidad de dígitos a mostrar en el DataFrame final (afecta la visualización).
    
    Retorna:
    --------
    - x: La aproximación final de la raíz.
    """
    x = float(valor_inicial) 
    results_data = [] # Lista para almacenar los datos de cada iteración
    
    print(f"Iniciando Newton-Raphson con tolerancia |x_new - x| < {tolerancia}")
    
    for i in range(iteraciones):
        fx = f(x) 
        dfx = numpy_derivative(f, x, dx=1e-7) 
        
        if abs(dfx) < 1e-12:
             print(f"\nAdvertencia en iteración {i+1}: Derivada cercana a cero ({dfx}). El método puede fallar.")
             if abs(fx) < tolerancia: 
                 print("f(x) es pequeño, considerando la iteración actual como resultado.")
                 break
             else:
                 raise ValueError("La derivada es cercana a cero y f(x) no es pequeño. El método no puede continuar.")

        x_new = x - fx / dfx
        error_abs = abs(x_new - x)
        
        # Guardar datos numéricos en la lista
        results_data.append({
            "Iteración": i + 1,
            "x_i": x,
            "f(x_i)": fx,
            "f'(x_i)": dfx,
            "x_{i+1}": x_new,
            "Error |xᵢ₊₁ - xᵢ|": error_abs
        })
        
        # Condición de parada basada en el error absoluto
        if error_abs < tolerancia:
            print(f"\nConvergencia alcanzada en la iteración {i+1}: Error ({error_abs:.{precision}e}) < Tolerancia ({tolerancia})")
            x = x_new 
            break
            
        x = x_new 
        
        if i == iteraciones - 1:
             print(f"\nAdvertencia: Se alcanzó el número máximo de iteraciones ({iteraciones}) sin converger a la tolerancia deseada.")
             
    # Crear DataFrame de pandas con los resultados
    results_df = pd.DataFrame(results_data)
    
    # Establecer 'Iteración' como índice para una mejor visualización
    results_df.set_index("Iteración", inplace=True)
    
    # Configurar opciones de visualización de pandas para la precisión deseada
    pd.set_option('display.float_format', f'{{:.{precision}f}}'.format) 
    # Formato específico para la columna de error en notación científica
    results_df['Error |xᵢ₊₁ - xᵢ|'] = results_df['Error |xᵢ₊₁ - xᵢ|'].apply(lambda e: f'{e:.{precision}e}')

    # Imprimir el DataFrame
    print("\n--- Tabla de Iteraciones ---")
    print(results_df)
    # Restaurar formato por defecto si es necesario para otras partes del código
    # pd.reset_option('display.float_format') 
    
    return x

def graficar(f, raiz, valor_inicial, precision=10):
    # (Código de graficar sin cambios)
    rango_min = min(valor_inicial, raiz) - abs(valor_inicial - raiz) - 0.5
    rango_max = max(valor_inicial, raiz) + abs(valor_inicial - raiz) + 0.5
    if rango_max - rango_min < 1.0:
        rango_min -= 0.5
        rango_max += 0.5
        
    x_vals = np.linspace(rango_min, rango_max, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='$f(x) = e^x - 3x^2$') 
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    
    plt.plot(raiz, f(raiz), 'ro', markersize=8, label=f'Raíz encontrada: x ≈ {raiz:.{precision}f}')
    plt.plot(valor_inicial, f(valor_inicial), 'go', markersize=6, label=f'Valor inicial: x₀ = {valor_inicial}')
    
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Método de Newton-Raphson')
    
    y_vals_clean = y_vals[np.isfinite(y_vals)] 
    if len(y_vals_clean) > 0:
        y_min = min(y_vals_clean) - 1
        y_max = max(y_vals_clean) + 1
        if y_max - y_min > 50: 
             y_min = max(y_min, -25)
             y_max = min(y_max, 25)
        plt.ylim(y_min, y_max)
        
    plt.show()

# Definir la función de la cual se busca la raíz
def f(x):
    return np.exp(x) - 3 * (x**2) 

# Valor inicial definido directamente
valor_inicial = 0.5

# Parámetros: tolerancia y precisión
tolerancia = 1e-8      
precision = 8         # Precisión para la visualización en pandas

# Encontrar la raíz usando el método de Newton-Raphson
raiz = newton_raphson(f, valor_inicial, iteraciones=100, tolerancia=tolerancia, precision=precision)

print(f"\nLa raíz encontrada es: {raiz:.{precision}f}")

# Graficar la función y la raíz encontrada
graficar(f, raiz, valor_inicial, precision=precision)