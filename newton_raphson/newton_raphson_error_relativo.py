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

# Método de Newton-Raphson utilizando el error relativo porcentual
def newton_raphson(f, valor_inicial, iteraciones=100, tolerancia_porcentaje=1.0, precision=10):
    """
    Aplica el método de Newton-Raphson para hallar la raíz de una función,
    deteniéndose cuando el error relativo porcentual es menor que la tolerancia dada.
    
    Parámetros:
    -----------
    - f: Función de la cual se desea encontrar la raíz.
    - valor_inicial: Aproximación inicial.
    - iteraciones: Número máximo de iteraciones.
    - tolerancia_porcentaje: Tolerancia del error relativo expresada en porcentaje (ej: 1.0 para 1%).
    - precision: Cantidad de dígitos a mostrar (formato, evita notación científica).
    
    Retorna:
    --------
    - x: La aproximación final de la raíz.
    """
    x = valor_inicial
    results = []
    # Convertir la tolerancia porcentual a fracción para la comparación
    tolerancia_fraccion = tolerancia_porcentaje / 100.0 
    
    print(f"Iniciando Newton-Raphson con tolerancia de error relativo < {tolerancia_porcentaje}%")
    
    for i in range(iteraciones):
        fx = f(x) # Usar valor sin redondear para cálculos internos
        dfx = numpy_derivative(f, x, dx=1e-7) # Usar una dx pequeña y fija para la derivada
        
        if abs(dfx) < 1e-12: # Evitar división por cero o derivada muy pequeña
             print("\nAdvertencia: Derivada cercana a cero. El método puede diverger o ser inestable.")
             # Opcional: podrías lanzar un error aquí si prefieres detener la ejecución
             # raise ValueError("La derivada es cercana a cero. El método no puede continuar.")
             # O continuar con precaución, como se hace aquí
             if abs(fx) < tolerancia_fraccion: # Si f(x) ya es pequeño, quizás estemos cerca
                 print("f(x) es pequeño, considerando la iteración actual como resultado.")
                 break
             else: # Si f(x) no es pequeño y dfx sí, es problemático
                 print("No se puede continuar de forma segura.")
                 # Devolver el último valor válido o manejar como error
                 # Aquí simplemente salimos y devolvemos el último x
                 break 

        x_new = x - fx / dfx
        
        # Calcular error relativo porcentual
        error_rel_porcentaje = 0.0
        if abs(x_new) > 1e-12: # Evitar división por cero para el error relativo
            error_rel = abs(x_new - x) / abs(x_new)
            error_rel_porcentaje = error_rel * 100.0
        else: # Si x_new es muy cercano a cero, usar error absoluto como fallback
              # o considerar que si x también es cercano a cero, la convergencia es buena
            error_abs = abs(x_new - x)
            # Podrías decidir detenerte si el error absoluto es muy pequeño también
            # if error_abs < tolerancia_fraccion: # Comparar error absoluto con la tolerancia fraccional
            #    error_rel_porcentaje = 0 # Considerarlo convergido
            print(f"Iteración {i+1}: x_new cercano a cero, usando error absoluto {error_abs:.{precision}f} para evaluar.")
            # Aquí mantenemos el cálculo original, pero podrías ajustar la lógica
            # Si x_new es 0, el error relativo es infinito o indefinido, así que no podemos usarlo directamente
            # Una opción es detenerse si el error absoluto es menor que la tolerancia * |x| (aproximación anterior)
            if abs(x) > 1e-12 and (abs(x_new - x) / abs(x)) * 100 < tolerancia_porcentaje:
                 print("Convergencia basada en error relativo respecto a x anterior.")
                 error_rel_porcentaje = (abs(x_new - x) / abs(x)) * 100
            elif abs(x_new - x) < tolerancia_fraccion: # O si el error absoluto es muy pequeño
                 print("Convergencia basada en error absoluto pequeño.")
                 error_rel_porcentaje = 0 # Forzar convergencia si error absoluto es pequeño
            else:
                 error_rel_porcentaje = float('inf') # Indicar que no se puede calcular bien


        # Formatear cada valor para la tabla (redondeo solo para mostrar)
        fila = [
            str(i + 1),
            f"{x:.{precision}f}",
            f"{f(x):.{precision}f}", # Mostrar f(x) evaluado en el x de esta iteración
            f"{dfx:.{precision}f}",
            f"{x_new:.{precision}f}",
            f"{error_rel_porcentaje:.{precision}f}%" # Mostrar como porcentaje
        ]
        results.append(fila)
        
        # Condición de parada basada en el error relativo porcentual
        if error_rel_porcentaje < tolerancia_porcentaje:
            print(f"\nConvergencia alcanzada en la iteración {i+1}: Error Relativo ({error_rel_porcentaje:.{precision}f}%) < Tolerancia ({tolerancia_porcentaje}%)")
            x = x_new # Actualizar x al último valor calculado antes de salir
            break
            
        x = x_new # Actualizar x para la siguiente iteración
        
        if i == iteraciones - 1:
             print("\nAdvertencia: Se alcanzó el número máximo de iteraciones sin converger a la tolerancia deseada.")
    
    # Imprimir la tabla de resultados una sola vez, al finalizar
    headers = ["Iteración", "x_i", "f(x_i)", "f'(x_i)", "x_{i+1}", "Error Relativo (%)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Devolver el último valor calculado, incluso si no se alcanzó la tolerancia exacta
    # Es importante devolver el 'x' final después del bucle o la última actualización dentro del if de convergencia
    return x

def graficar(f, raiz, precision=10):
    # Generar puntos para graficar la función en un rango adecuado
    # Ajustar el rango si es necesario basado en la función o la raíz esperada
    rango_min = min(valor_inicial, raiz) - 2
    rango_max = max(valor_inicial, raiz) + 2
    x_vals = np.linspace(rango_min, rango_max, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(10, 6)) # Ajustar tamaño de figura
    plt.plot(x_vals, y_vals, label=f'$f(x) = x^3 - 3x - 4$') # Usar expresión LaTeX si es posible
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # Marcar la raíz encontrada
    plt.plot(raiz, f(raiz), 'ro', markersize=8, label=f'Raíz encontrada: x ≈ {raiz:.{precision}f}')
    # Marcar el punto inicial
    plt.plot(valor_inicial, f(valor_inicial), 'go', markersize=6, label=f'Valor inicial: x₀ = {valor_inicial}')
    
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Método de Newton-Raphson')
    plt.ylim(min(y_vals)-1, max(y_vals)+1) # Ajustar límites y para mejor visualización
    plt.show()

# Definir la función de la cual se busca la raíz
def f(x):
    return x**3 - 3*x - 4

# Valor inicial definido directamente
valor_inicial = 2.0 # Usar float para consistencia

# Parámetros: tolerancia y precisión (modifica estos valores según lo necesites)
# tolerancia_porcentaje = 1.0    # Detenerse cuando el error relativo sea < 1.0%
tolerancia_porcentaje = 1    # Ejemplo: Detenerse cuando el error relativo sea < 0.5%
precision = 8                  # Cantidad de dígitos a mostrar en la tabla y resultado final

# Ejecutar el método de Newton-Raphson con error relativo porcentual
raiz_encontrada = newton_raphson(f, valor_inicial, iteraciones=100, tolerancia_porcentaje=tolerancia_porcentaje, precision=precision)

print(f"\nLa raíz final encontrada es: {raiz_encontrada:.{precision}f}")

# Graficar la función y la raíz encontrada
graficar(f, raiz_encontrada, precision=precision)